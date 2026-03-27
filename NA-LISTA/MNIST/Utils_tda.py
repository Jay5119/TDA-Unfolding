# Utils.py
import math
import torch
import torch.nn.functional as F
import numpy as np


def to_DB(MSE):
    """Convert MSE to dB scale."""
    return 10 * np.log10(MSE)

# def add_noise_fixed_snr(y, y_ddim, snr_db):
#     y = np.asarray(y, dtype=np.float64)
#     z = np.random.randn(*y.shape)
#     signal_power = np.sum(y**2, axis=0, keepdims=True)
#     noise_power = np.sum(z**2, axis=0, keepdims=True)
#     scale = np.sqrt(signal_power / (noise_power * 10**(snr_db / 10.0)))
#     n = scale * z
#     y_noisy = y + n
#     y_noisy_ddim = y_ddim + n
#     sigma = np.std(n, axis=0)
#     snr_out = 10 * np.log10(np.sum(y**2, axis=0) / np.sum(n**2, axis=0))
#     return y_noisy, y_noisy_ddim, sigma, snr_out

def add_noise_fixed_snr(y, y_ddim, snr_db):
    y = np.asarray(y, dtype=np.float64)
    y_ddim = np.asarray(y_ddim, dtype=np.float64)
    z = np.random.randn(*y_ddim.shape) 
    m = y.shape[0]
    # Noise for y
    z_y = z[:m, :]
    signal_power_y = np.sum(y**2, axis=0, keepdims=True)
    noise_power_y = np.sum(z_y**2, axis=0, keepdims=True)
    scale_y = np.sqrt(signal_power_y / (noise_power_y * 10**(snr_db / 10.0)))
    n_y = scale_y * z_y
    y_noisy = y + n_y
    
    # Noise for y_ddim
    z_ddim = z
    signal_power_ddim = np.sum(y_ddim**2, axis=0, keepdims=True)
    noise_power_ddim = np.sum(z_ddim**2, axis=0, keepdims=True)
    scale_ddim = np.sqrt(signal_power_ddim / (noise_power_ddim * 10**(snr_db / 10.0)))
    n_ddim = scale_ddim * z_ddim
    y_noisy_ddim = y_ddim + n_ddim
    
    sigma = np.std(n_y, axis=0)  # y's sigma
    snr_out = 10 * np.log10(np.sum(y**2, axis=0) / np.sum(n_y**2, axis=0))
    
    return y_noisy, y_noisy_ddim, sigma, snr_out

def flatten_image(x: torch.Tensor) -> torch.Tensor:
    """Reshape images to [B, C, N]."""
    return x.view(x.shape[0], x.shape[1], -1)


def compute_psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    """Scalar PSNR from scalar MSE."""
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    PSNR per sample. Expects pred/target shaped [B, C, N] or [B, C, H, W].
    """
    if pred.dim() == 4:
        mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    else:
        mse = torch.mean((pred - target) ** 2, dim=(1, 2))
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))
    return psnr


def to_device_batch(batch: dict, device: torch.device) -> dict:
    """Move all tensor values of a batch dict to the given device."""
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def model_forward(model, model_name: str | None, y: torch.Tensor, sigma: torch.Tensor | None = None):
    """
    Dispatch forward depending on model type/name.
    - JT: model(y)
    - PTDA: model(y, sigma)  (requires sigma)
    - DDTDA: model(y)
    - Tail-LISTA: model(y)
    Falls back to model(y) or model(y, sigma) when ambiguous.
    """
    name = (model_name or model.__class__.__name__).lower()
    if "ptda" in name:
        if sigma is None:
            raise ValueError("PTDA models require sigma in the batch.")
        return model(y, sigma)
    if "ddtda" in name:
        return model(y)
    if "jt" in name:
        return model(y)
    if "tail" in name:
        return model(y)
    # Fallback: prefer (y, sigma) if sigma given
    return model(y, sigma) if sigma is not None else model(y)


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.unsqueeze(0)  # [1, W]
    window_2d = window_1d.T @ window_1d  # [W, W]
    return window_2d


def batch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    SSIM per sample for shapes [B, C, H, W]. Assumes grayscale channels per sample.
    """
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")

    pad = window_size // 2
    B, C, H, W = x.shape
    window = _gaussian_window(window_size, sigma, x.device, x.dtype)
    window = window.expand(C, 1, window_size, window_size)

    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    # average over channel and spatial dims -> [B]
    return ssim_map.mean(dim=(1, 2, 3))


# ============================================================================
# Time Formatting Utilities for Research Paper Reporting
# ============================================================================

def format_time(seconds: float, precision: str = "auto") -> str:
    """
    Format elapsed time in a human-readable way suitable for research papers.
    
    Args:
        seconds: Elapsed time in seconds.
        precision: "auto", "microseconds", "milliseconds", "seconds", "minutes", or "hours".
                  "auto" selects the most appropriate unit.
    
    Returns:
        Formatted time string (e.g., "2.34s", "1.56m", "0.45h").
    
    Examples:
        >>> format_time(0.0008)   # -> "800.00µs"
        >>> format_time(0.08)     # -> "80.00ms"
        >>> format_time(5.6)      # -> "5.60s"
        >>> format_time(125.4)    # -> "2.09m"
        >>> format_time(3665.2)   # -> "1.02h"
    """
    if precision == "auto":
        if seconds < 1e-3:
            precision = "microseconds"
        elif seconds < 1:
            precision = "milliseconds"
        elif seconds < 60:
            precision = "seconds"
        elif seconds < 3600:
            precision = "minutes"
        else:
            precision = "hours"
    
    if precision == "microseconds":
        return f"{seconds * 1e6:.2f}µs"
    elif precision == "milliseconds":
        return f"{seconds * 1e3:.2f}ms"
    elif precision == "seconds":
        return f"{seconds:.2f}s"
    elif precision == "minutes":
        return f"{seconds / 60:.2f}m"
    elif precision == "hours":
        return f"{seconds / 3600:.2f}h"
    else:
        raise ValueError(f"Unknown precision: {precision}")


def format_time_detailed(seconds: float) -> str:
    """
    Format elapsed time with all components (hours, minutes, seconds).
    
    Args:
        seconds: Elapsed time in seconds.
    
    Returns:
        Formatted string (e.g., "1h 24m 35s").
    
    Examples:
        >>> format_time_detailed(5076.5)  # -> "1h 24m 36s"
        >>> format_time_detailed(125)     # -> "2m 5s"
    """
    hours = int(seconds // 3600)
    remaining = seconds % 3600
    minutes = int(remaining // 60)
    secs = int(remaining % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def get_time_unit(seconds: float) -> tuple[str, float]:
    """
    Determine the best unit and convert time accordingly.
    
    Args:
        seconds: Elapsed time in seconds.
    
    Returns:
        Tuple of (unit_string, converted_value).
        e.g., ("minutes", 2.09) for 125.4 seconds.
    
    Examples:
        >>> get_time_unit(0.08)
        ("milliseconds", 80.0)
        >>> get_time_unit(5.6)
        ("seconds", 5.6)
        >>> get_time_unit(125)
        ("minutes", 2.08)
    """
    if seconds < 1e-3:
        return "microseconds", seconds * 1e6
    elif seconds < 1:
        return "milliseconds", seconds * 1e3
    elif seconds < 60:
        return "seconds", seconds
    elif seconds < 3600:
        return "minutes", seconds / 60
    else:
        return "hours", seconds / 3600