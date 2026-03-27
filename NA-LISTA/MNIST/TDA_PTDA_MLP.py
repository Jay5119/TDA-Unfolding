import torch
import torch.nn as nn


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # sign(x) * relu(|x| - theta)
    return torch.relu(x - theta) - torch.relu(-x - theta)

class PTDAThresholdNet(nn.Module):
    """
    MLP-based threshold predictor with layer-wise (global) thresholds.
    
    Input:
      - y_vec: [B, M] or [B, 1, M] or [B, 1, P, P]
      - sigma: [B] or [B, 1]
    
    Output:
      - theta: [B, K]  (one threshold per layer, shared across all N)
    """
    
    def __init__(
        self,
        M: int,
        N: int,  # Not used for output, kept for compatibility
        K: int,
        fc_hidden: int = 64,
        lambda_init: float = 0.1,
        L_const: float = 1.0,
    ):
        super().__init__()
        self.N = N  # Stored but not used in output dimension
        self.K = K
        self.L_const = float(L_const)
        
        input_dim = M + 1  # 551 for M=550
        output_dim = K  # Changed from K*N to just K
        
        # Progressive dimensionality reduction
        # input_dim -> int(1.5 * input_dim) -> int(input_dim * 0.75) -> int(input_dim * 0.25) -> fc_hidden -> output_dim 
        self.fc1 = nn.Linear(input_dim, int(1.5 * input_dim), bias=True)
        self.fc2 = nn.Linear(int(1.5 * input_dim), int(input_dim * 0.75), bias=True)
        self.fc3 = nn.Linear(int(input_dim * 0.75), int(input_dim * 0.25), bias=True)
        self.fc4 = nn.Linear(int(input_dim * 0.25), fc_hidden, bias=True)
        self.fc_out = nn.Linear(fc_hidden, output_dim, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        
        # Initialize the output layer with init_gain
        self._initialize_output_layer(lambda_init, L_const)
    
    def _initialize_output_layer(self, lambda_init: float, L_const: float):
        """Initialize final layer to output values near init_gain."""
        init_gain = float(lambda_init) / L_const
        
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_out.bias, init_gain)
        
    def forward(self, y_vec: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        B = y_vec.shape[0]
        dev = y_vec.device
        
        # Flatten y_vec to [B, M]
        if y_vec.dim() == 4:
            if y_vec.shape[1] != 1:
                raise ValueError(f"Expected single channel, got {y_vec.shape[1]}")
            y_flat = y_vec.view(B, -1)
        elif y_vec.dim() == 3:
            if y_vec.shape[1] != 1:
                raise ValueError(f"Expected single channel, got {y_vec.shape[1]}")
            y_flat = y_vec.view(B, -1)
        elif y_vec.dim() == 2:
            y_flat = y_vec
        else:
            raise ValueError(f"Unsupported y_vec shape: {tuple(y_vec.shape)}")
        
        # Reshape sigma to [B, 1]
        sigma = sigma.to(dev)
        if sigma.dim() == 1:
            sigma = sigma.view(B, 1)
        elif sigma.dim() == 2 and sigma.shape[1] == 1:
            pass
        elif sigma.dim() == 3 and sigma.shape[1:] == (1, 1):
            sigma = sigma.view(B, 1)
        else:
            raise ValueError(f"Unsupported sigma shape: {tuple(sigma.shape)}")
        
        # Forward pass
        x = torch.cat([y_flat, sigma], dim=1)  # [B, 551]
        
        x = self.relu(self.fc1(x))    # [B, 512]
        x = self.relu(self.fc2(x))    # [B, 256]
        x = self.relu(self.fc3(x))    # [B, 128]
        x = self.relu(self.fc4(x))    # [B, fc_hidden]
        x = self.fc_out(x)             # [B, K]
        
        # Apply softplus for positive thresholds
        theta = self.softplus(x)       # [B, K]
        
        return theta


class MultiChannelLISTALayer_PTDA(nn.Module):
    """
    One untied multi-channel LISTA layer with PTDA-provided thresholds.

    Shapes:
      - W1: [C, N, M]
      - W2: [C, N, N]
      - Y:  [B, C, M]
      - Zk: [B, C, N]
      - theta_k: [B, C, N]
    """

    def __init__(
        self,
        M: int,
        N: int,
        A: torch.Tensor,         # [C, M, N]
        C: int = 3,
        init_noise: float = 1e-5,
        L_const: float = 1.0,
        device: torch.device | None = None
    ):
        super().__init__()
        if device is None:
            device = A.device
        A = A.to(device)

        assert A.shape == (C, M, N), f"Expected A shape {(C, M, N)}, got {tuple(A.shape)}"

        AT = A.transpose(-2, -1)                      # [C, N, M]
        ATA = torch.einsum("cnm,cmk->cnk", AT, A)     # [C, N, N]
        I = torch.eye(N, dtype=A.dtype, device=device).unsqueeze(0).repeat(C, 1, 1)

        W1 = (1.0 / float(L_const)) * AT
        W2 = I - (1.0 / float(L_const)) * ATA

        self.W1 = nn.Parameter(W1.clone())
        self.W2 = nn.Parameter(W2.clone())

        if init_noise > 0:
            with torch.no_grad():
                self.W1.add_(init_noise * torch.randn_like(self.W1))
                self.W2.add_(init_noise * torch.randn_like(self.W2))

        self.register_buffer("L_const", torch.tensor(float(L_const), device=device))

    def forward(self, Y: torch.Tensor, Zk: torch.Tensor, theta_k: torch.Tensor) -> torch.Tensor:
        W1_Y = torch.einsum("cnm,bcm->bcn", self.W1, Y)
        W2_Zk = torch.einsum("cnk,bck->bcn", self.W2, Zk)
        return soft_threshold(W1_Y + W2_Zk, theta_k)


class NA_LISTA_PTDA(nn.Module):
    """
    K-layer multi-channel LISTA with PTDA thresholds.

    Forward inputs:
      - Y:     [B, C, M]  (already vectorized patch per channel)
      - sigma: [B] or [B,1]

    Outputs:
      - if all_output=True:  (Z, Zs) where Zs is a list of length K (each [B,C,N])
      - else:                Z only
    """

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        A: torch.Tensor,                 # [C, M, N]
        C: int = 1,
        lambda_init: float = 0.1,
        init_noise: float = 1e-5,
        tied: bool = True,
        device: torch.device | None = None,
        thr_fc_hidden: int = 64,
    ):
        super().__init__()
        if device is None:
            device = A.device
        A = A.to(device)

        assert A.shape == (C, M, N), f"Expected A shape {(C, M, N)}, got {tuple(A.shape)}"
        self.C, self.M, self.N, self.K = C, M, N, K
        self.tied = tied

        AT = A.transpose(-2, -1)
        ATA = torch.einsum("cnm,cmk->cnk", AT, A)

        # Spectral-norm-based Lipschitz constant (per-channel), then max over channels.
        L = torch.linalg.matrix_norm(ATA, ord=2).max().item() + 1e-5
        self.register_buffer("L_const", torch.tensor(float(L), device=device))

        if tied:
            self.shared_layer = MultiChannelLISTALayer_PTDA(
                M=M, N=N, A=A, C=C,
                init_noise=init_noise, L_const=L,
                device=device
            )
        else:
            self.layers = nn.ModuleList([
                MultiChannelLISTALayer_PTDA(
                    M=M, N=N, A=A, C=C,
                    init_noise=init_noise, L_const=L,
                    device=device
                )
                for _ in range(K)
            ])

        # One threshold net per channel (registered properly via ModuleList). [web:1]
        self.thr_nets = nn.ModuleList([
            PTDAThresholdNet(
                M=M,
                N=N,
                K=K,
                fc_hidden=thr_fc_hidden,
                lambda_init=lambda_init,
                L_const=L,
            )
            for _ in range(C)
        ])

    def forward(self, Y: torch.Tensor, sigma: torch.Tensor, all_outputs: bool = False):
        dev = self.L_const.device
        Y = Y.to(dev)
        sigma = sigma.to(dev)

        B = Y.shape[0]
        assert Y.shape == (B, self.C, self.M), f"Expected Y [B,{self.C},{self.M}], got {tuple(Y.shape)}"

        # theta_all: [B, C, K, N]
        theta_per_c = []
        for c in range(self.C):
            y_c = Y[:, c, :]                 # [B, M] already vectorized
            theta_c = self.thr_nets[c](y_c, sigma)  # [B, K, N]
            theta_per_c.append(theta_c)
        theta_all = torch.stack(theta_per_c, dim=1)
        theta_all = theta_all.unsqueeze(-1).expand(B, self.C, self.K, self.N)

        Z = torch.zeros(B, self.C, self.N, device=dev, dtype=Y.dtype)
        Zs = [] if all_outputs else None

        if self.tied:
            for k in range(self.K):
                theta_k = theta_all[:, :, k, :]      # [B, C, N]
                Z = self.shared_layer(Y, Z, theta_k)
                if all_outputs:
                    Zs.append(Z)
        else:
            for k, layer in enumerate(self.layers):
                theta_k = theta_all[:, :, k, :]      # [B, C, N]
                Z = layer(Y, Z, theta_k)
                if all_outputs:
                    Zs.append(Z)

        if all_outputs:
            return Z, Zs
        return Z
