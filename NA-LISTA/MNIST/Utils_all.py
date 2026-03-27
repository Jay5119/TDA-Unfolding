import copy
import torch
import numpy as np
from torch.utils.data import Dataset

try:
    from IPython import get_ipython
except Exception:  # pragma: no cover - safe fallback outside IPython
    def get_ipython():
        return None

## Create Train, Val and Test dataloader with validation of 10% from test.
class DictCS(Dataset):
    def __init__(self, Y, E, X, X_d, Label, Sigma=None):
        self.Y = Y
        self.X = X
        self.E = E
        self.X_d = X_d
        self.Label = Label
        self.Sigma = Sigma
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, i):
        out = {
            "y": self.Y[i], 
            "x": self.X[i], 
            "e": self.E[i],
            "x_d": self.X_d[i],
            "label": self.Label[i],
            }
        if self.Sigma is not None:
            out["sigma"] = self.Sigma[i]
        return out


def _remove_thop_buffers(model: torch.nn.Module) -> None:
    """Remove buffers/attrs that thop may add so state_dict stays clean."""
    for m in model.modules():
        if hasattr(m, "_buffers"):
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)
        if hasattr(m, "total_ops"):
            try:
                delattr(m, "total_ops")
            except Exception:
                pass
        if hasattr(m, "total_params"):
            try:
                delattr(m, "total_params")
            except Exception:
                pass


def try_profile_flops(model: torch.nn.Module, inputs, device: torch.device):
    """
    Profile FLOPs with fvcore on a deepcopy of `model`.
    Returns (flops, params) where params is computed via PyTorch (reliable).
    If fvcore is unavailable or profiling fails, flops is None.
    """
    params = int(sum(p.numel() for p in model.parameters()))

    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None, params

    try:
        m = copy.deepcopy(model).to(device)
        m.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(m, inputs).total()
        return float(flops), params
    except Exception:
        return None, params


def _fmt_flops(flops: float) -> str:
    if flops is None or not np.isfinite(flops):
        return "N/A"
    if flops >= 1e9:
        return f"{flops/1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops/1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


## Reshape DDIM data E and X_DDIM to (N, 1, 28, 28) before dataset creation
def cols784_to_mnist_tensor(mat_np: np.ndarray) -> torch.Tensor:
    """
    Convert np array shaped (784, N) or (N, 784) into torch tensor (N, 1, 28, 28).
    """
    a = np.asarray(mat_np)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    if a.shape[0] == 28 * 28:
        a = a.T  # (N, 784)
    elif a.shape[1] == 28 * 28:
        pass     # already (N, 784)
    else:
        raise ValueError(f"Expected one dim to be 784, got shape {a.shape}")
    return torch.from_numpy(a.astype(np.float32, copy=False)).view(-1, 1, 28, 28)

# ---- Show reconstructed MNIST image of one sample per SNR (pulled from test_loaders) ----
def _to_img01_from_vec(x_vec: torch.Tensor) -> np.ndarray:
    """
    x_vec: (1, 1, 784) or (1, 784) or (784,)
    returns (28, 28) in [0,1]
    """
    # Multiply by 255
    x = x_vec.float().view(-1)  # (784,)
    x = x * 255.0
    return x.view(28, 28).numpy()

def _to_img01_from_ddim(x_d: torch.Tensor) -> np.ndarray:
    """
    x_d: (1, 1, 28, 28) in [-1,1] -> (28,28) in [0,1]
    """
    # Add 1 and divide by 2 to scale to [0,1], then multiply by 255
    x = x_d.float().squeeze(0).squeeze(0)  # (28,28)
    x = ((x + 1.0) / 2.0) * 255.0
    return x.numpy()

class NotebookTee:
    """Write notebook cell output to both the current cell and a log file."""
    def __init__(self, logfile_path, target_stream):
        self._target = target_stream
        self._log = open(logfile_path, "w", buffering=1)
    def _sync_parent(self):
        ip = get_ipython()
        if hasattr(self._target, "set_parent") and ip and getattr(ip, "parent_header", None):
            self._target.set_parent(ip.parent_header)
    def write(self, message):
        self._sync_parent()
        self._target.write(message)
        self._target.flush()
        self._log.write(message)
        self._log.flush()
    def flush(self):
        self._sync_parent()
        self._target.flush()
        self._log.flush()