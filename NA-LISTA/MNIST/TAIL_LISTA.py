import torch
import torch.nn as nn


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Soft-thresholding: sign(x) * relu(|x| - theta)."""
    theta = theta.unsqueeze(0)  # [1, C, N] for broadcasting over batch
    return torch.relu(x - theta) - torch.relu(-x - theta)


def soft_threshold_with_support(x: torch.Tensor, theta: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    """
    Support-aware soft-thresholding.
    
    Args:
        x: Input tensor [B, C, N]
        theta: Threshold [C, N]
        support_mask: Binary mask [B, C, N], where 1=tail (apply threshold), 0=support (no threshold)
    
    Returns:
        Thresholded output [B, C, N]
    """
    # Weighted threshold: only apply to tail indices
    theta_weighted = theta.unsqueeze(0) * support_mask  # [B, C, N]
    return torch.relu(x - theta_weighted) - torch.relu(-x - theta_weighted)


class TailLISTALayer(nn.Module):
    """
    One Tail-LISTA layer (per-channel) with support-aware soft-thresholding.
    
    Shapes:
      A:  [C, M, N]  (used only for initialization)
      W1: [C, N, M]
      W2: [C, N, N]
      θ:  [C, N]
      Y:  [B, C, M]
      Zk: [B, C, N]
      support_mask: [B, C, N]
    """
    def __init__(self, M: int, N: int, A: torch.Tensor, C: int = 1,
                 init_noise: float = 1e-5,
                 W1_init: torch.Tensor | None = None,
                 W2_init: torch.Tensor | None = None,
                 theta_init: float = 0.1,
                 device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = A.device
        
        A = A.to(device)
        assert A.shape == (C, M, N), f"Expected A shape {(C, M, N)}, got {tuple(A.shape)}"
        assert W1_init is not None and W2_init is not None, "W1_init/W2_init must be provided"
        
        # Learnable weight matrices
        self.W1 = nn.Parameter(W1_init.clone())  # [C, N, M]
        self.W2 = nn.Parameter(W2_init.clone())  # [C, N, N]
        
        # Learnable threshold
        self.theta = nn.Parameter(torch.full((C, N), float(theta_init), dtype=A.dtype, device=device))
        
        # Add small noise to break symmetry
        if init_noise > 0:
            with torch.no_grad():
                self.W1.add_(init_noise * torch.randn_like(self.W1))
                self.W2.add_(init_noise * torch.randn_like(self.W2))
    
    def forward(self, Y: torch.Tensor, Zk: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with support-aware thresholding.
        
        Args:
            Y: Observations [B, C, M]
            Zk: Current estimate [B, C, N]
            support_mask: Binary mask [B, C, N], 1=tail, 0=support
        
        Returns:
            Updated estimate [B, C, N]
        """
        # Linear update: W1*Y + W2*Zk
        W1_Y = torch.einsum("cnm,bcm->bcn", self.W1, Y)
        W2_Zk = torch.einsum("cnk,bck->bcn", self.W2, Zk)
        
        # Support-aware soft-thresholding
        return soft_threshold_with_support(W1_Y + W2_Zk, self.theta, support_mask)


class InternalLISTA(nn.Module):
    """
    Internal LISTA network with K layers for Tail-LISTA.
    
    Shapes:
      Y: [B, C, M]
      Z: [B, C, N]
      support_mask: [B, C, N]
    """
    def __init__(self, M: int, N: int, K: int, A: torch.Tensor, C: int = 1,
                 W1_init: torch.Tensor | None = None,
                 W2_init: torch.Tensor | None = None,
                 theta_init: float = 0.1,
                 init_noise: float = 1e-5,
                 tied: bool = True,
                 device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = A.device
        
        self.K = K
        self.tied = tied
        if tied:
            # Single shared layer
            self.shared_layer = TailLISTALayer(M=M, N=N, A=A, C=C,
                                              W1_init=W1_init, W2_init=W2_init,
                                              theta_init=theta_init, init_noise=init_noise,
                                              device=device)
        else:        
            # Create K Tail-LISTA layers
            self.layers = nn.ModuleList([
                TailLISTALayer(M=M, N=N, A=A, C=C,
                            W1_init=W1_init, W2_init=W2_init,
                            theta_init=theta_init, init_noise=init_noise,
                            device=device)
                for _ in range(K)
            ])
    
    def forward(self, Y: torch.Tensor, Z: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward through K LISTA layers.
        
        Args:
            Y: Observations [B, C, M]
            Z: Initial estimate [B, C, N]
            support_mask: Binary mask [B, C, N]
        
        Returns:
            Final estimate [B, C, N]
        """
        if self.tied:
            for _ in range(self.K):
                Z = self.shared_layer(Y, Z, support_mask)
            return Z
        else:
            for layer in self.layers:
                Z = layer(Y, Z, support_mask)
            return Z


class TailLISTA(nn.Module):
    """
    Tail-LISTA: Deep unfolding of Tail-ISTA algorithm.
    
    Alternates between:
    1. Internal LISTA network updates (signal estimation)
    2. Tail/support updates (support refinement)
    
    Shapes:
      A (init):   [C, M, N]
      Y (input):  [B, C, M]
      Z (output): [B, C, N]
    """
    def __init__(self, M: int, N: int, num_tail_steps: int, K: int, 
                 A: torch.Tensor, C: int = 1,
                 lambda_init: float = 0.1, init_noise: float = 1e-5,
                 tied: bool = True,
                 device: torch.device | None = None):
        """
        Args:
            M: Observation dimension
            N: Signal dimension
            num_tail_steps: Number of tail update steps (outer iterations, called M in paper)
            K: Number of LISTA layers per tail step (inner iterations)
            A: Sensing matrix [C, M, N]
            C: Number of channels
            lambda_init: Initial regularization parameter
            init_noise: Noise added to initialization for symmetry breaking
            device: Device for computation
        """
        super().__init__()
        if device is None:
            device = A.device
        
        A = A.to(device)
        assert A.shape == (C, M, N), f"Expected A shape {(C, M, N)}, got {tuple(A.shape)}"
        
        self.C, self.M, self.N = C, M, N
        self.num_tail_steps = num_tail_steps
        self.K = K
        
        # Compute initialization matrices using Lipschitz constant
        AT = A.transpose(-2, -1)  # [C, N, M]
        ATA = torch.einsum("cnm,cmk->cnk", AT, A)  # [C, N, N]
        I = torch.eye(N, dtype=A.dtype, device=device).unsqueeze(0).repeat(C, 1, 1)  # [C, N, N]
        L = torch.norm(ATA, dim=(1, 2), p=2).max() + 1e-5  # Lipschitz constant
        
        W1_init = (1 / L) * AT  # [C, N, M]
        W2_init = I - (1 / L) * ATA  # [C, N, N]
        theta_init = lambda_init / L
        
        self.register_buffer("L_const", torch.tensor(float(L), device=device))
        
        self.lista_networks = nn.ModuleList([
            InternalLISTA(M=M, N=N, K=K, A=A, C=C,
                        W1_init=W1_init, W2_init=W2_init,
                        theta_init=theta_init, init_noise=init_noise,
                        tied=tied, device=device)
            for _ in range(num_tail_steps)
        ])
    
    def _update_support(self, Z: torch.Tensor, current_support_size: int) -> torch.Tensor:
        """
        Update support set by selecting top-k largest magnitude elements per channel.
        
        Args:
            Z: Current signal estimate [B, C, N]
            current_support_size: Size of support to select
        
        Returns:
            support_mask: Binary mask [B, C, N]
                         1 for tail indices (apply threshold)
                         0 for support indices (no threshold)
        """
        B, C, N = Z.shape
        
        # Get absolute values
        Z_abs = torch.abs(Z)
        
        # Get indices of top-k largest elements for each sample and channel
        # topk returns (values, indices)
        _, top_indices = torch.topk(Z_abs, k=current_support_size, dim=-1)  # [B, C, k]
        
        # Create support mask: start with all ones (all tail)
        support_mask = torch.ones_like(Z)  # [B, C, N]
        
        # Set support indices to 0 (no threshold applied)
        # scatter_(dim, index, value) - sets value at positions specified by index
        support_mask.scatter_(2, top_indices, 0.0)
        
        return support_mask
    
    def forward(self, Y: torch.Tensor, all_outputs: bool = False):
        """
        Forward pass through Tail-LISTA network.
        
        Args:
            Y: Observation vector [B, C, M] or [B, M] (will be reshaped)
            all_outputs: If True, return estimates after each tail step
        
        Returns:
            Z: Final estimate [B, C, N]
            OR
            (Z, Zs): Final estimate and list of num_tail_steps estimates if all_outputs=True
        """
        dev = self.L_const.device
        Y = Y.to(dev)
        
        # Handle input shape: if [B, M], reshape to [B, 1, M]
        if Y.dim() == 2:
            Y = Y.unsqueeze(1)  # [B, 1, M]
            if Y.shape[1] != self.C:
                Y = Y.repeat(1, self.C, 1)  # [B, C, M] if needed
        
        assert Y.shape[1] == self.C and Y.shape[2] == self.M, \
            f"Expected Y shape [B, {self.C}, {self.M}], got {tuple(Y.shape)}"
        
        B = Y.shape[0]
        
        # Initialize Z to zeros
        Z = torch.zeros(B, self.C, self.N, device=Y.device, dtype=Y.dtype)
        
        # Initialize support mask (start with all elements in tail)
        support_mask = torch.ones(B, self.C, self.N, device=Y.device)
        
        # Store estimates for loss computation if needed
        Zs = [] if all_outputs else None
        
        # Current support size (increases by 1 each tail step)
        current_support_size = 0
        
        # num_tail_steps tail update iterations
        for m in range(self.num_tail_steps):
            # Run internal LISTA network with current support
            Z = self.lista_networks[m](Y, Z, support_mask)
            
            # Store estimate for loss computation
            if all_outputs:
                Zs.append(Z)
            
            # Update support: select top-(|T|+1) largest magnitude elements
            current_support_size = min(current_support_size + 1, self.N)
            support_mask = self._update_support(Z, current_support_size)
        
        if all_outputs:
            return Z, Zs  # Final estimate and list of num_tail_steps tensors
        else:
            return Z  # Final estimate only
