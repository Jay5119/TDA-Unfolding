import torch
import torch.nn as nn


def soft_threshold(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Soft-thresholding: sign(x) * relu(|x| - theta)."""
    theta = theta.unsqueeze(0)  # [1, C, N] for broadcasting over batch
    return torch.relu(x - theta) - torch.relu(-x - theta)


class MultiChannelLISTALayer(nn.Module):
    """
    One LISTA layer (per-channel) with parameters initialized from sensing matrices A.

    Shapes:
      A:  [C, M, N]  (used only for initialization here)
      W1: [C, N, M]
      W2: [C, N, N]
      θ:  [C, N]
      Y:  [B, C, M]
      Zk: [B, C, N]
    """
    def __init__(self, M: int, N: int, A: torch.Tensor, C: int = 3,
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
        
        self.W1 = nn.Parameter(W1_init.clone())
        self.W2 = nn.Parameter(W2_init.clone())
        self.theta = nn.Parameter(torch.full((C, N), float(theta_init), dtype=A.dtype, device=device))
        if init_noise > 0:
            with torch.no_grad():
                self.W1.add_(init_noise * torch.randn_like(self.W1))
                self.W2.add_(init_noise * torch.randn_like(self.W2))

    def forward(self, Y: torch.Tensor, Zk: torch.Tensor) -> torch.Tensor:
        W1_Y = torch.einsum("cnm,bcm->bcn", self.W1, Y)
        W2_Zk = torch.einsum("cnk,bck->bcn", self.W2, Zk)
        return soft_threshold(W1_Y + W2_Zk, self.theta)


class NA_LISTA_JT(nn.Module):
    """
    K-layer untied multi-channel LISTA.

    Shapes:
      A (init):   [C, M, N]
      Y (input):  [B, C, M]
      Z (output): [B, C, N]
      X_est:      [B, C, M]
    """
    def __init__(self, M: int, N: int, K: int, A: torch.Tensor, C: int = 1,
                 lambda_init: float = 0.1, init_noise: float = 1e-5,
                 tied: bool = True, device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = A.device

        A = A.to(device)
        assert A.shape == (C, M, N), f"Expected A shape {(C, M, N)}, got {tuple(A.shape)}"

        self.C, self.M, self.N, self.K = C, M, N, K
        self.tied = tied
        AT = A.transpose(-2, -1)                      # [C, N, M]
        ATA = torch.einsum("cnm,cmk->cnk", AT, A)     # [C, N, N]
        I = torch.eye(N, dtype=A.dtype, device=device).unsqueeze(0).repeat(C, 1, 1)  # [C, N, N]
        L = torch.norm(ATA, dim=(1, 2), p=2).max() + 1e-5  # Lipschitz constant
        W1_init = (1 / L) * AT                      # [C, N, M]
        W2_init = I - (1 / L) * ATA
        theta_init = lambda_init / L
        self.register_buffer("L_const", torch.tensor(float(L), device=device))

        if tied:
            self.shared_layer = MultiChannelLISTALayer(M=M, N=N, A=A, C=C,
                                                       W1_init=W1_init, W2_init=W2_init,
                                                       theta_init=theta_init, init_noise=init_noise,
                                                       device=device)
        else:
            self.layers = nn.ModuleList([
                MultiChannelLISTALayer(M=M, N=N, A=A, C=C,
                                       W1_init=W1_init, W2_init=W2_init,
                                       theta_init=theta_init, init_noise=init_noise,
                                       device=device)
                for _ in range(K)
            ])

    def forward(self, Y: torch.Tensor, all_outputs: bool = False):
        dev = self.L_const.device
        Y = Y.to(dev)

        B = Y.shape[0]
        Z = torch.zeros(B, self.C, self.N, device=Y.device, dtype=Y.dtype)
        Zs = [] if all_outputs else None
        if self.tied:
            for _ in range(self.K):
                Z = self.shared_layer(Y, Z)
                if all_outputs:
                    Zs.append(Z)
        else:
            for layer in self.layers:
                Z = layer(Y, Z)
                if all_outputs:
                    Zs.append(Z)

        if all_outputs:
            return Z, Zs
        return Z