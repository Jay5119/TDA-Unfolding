import time
import torch
import numpy as np
from tqdm import tqdm

from Utils_tda import (
    flatten_image,
    batch_psnr,
    batch_ssim,
    to_device_batch,
    model_forward,
    format_time,
)

from DDIM_utils_cs import (
    to_unit_interval,
    compute_psnr as compute_psnr_np,
    compute_ssim as compute_ssim_np,
    compute_nmse as compute_nmse_np,
)


@torch.no_grad()
def evaluate_model(
    model,
    ocr_model,
    test_loader,
    device: torch.device | None = None,
    model_name: str | None = None,
    data_range: float = 1.0,
    is_prnt: bool = False,
):
    """
    Evaluate JT/PTDA/DDTDA/Tail-LISTA models on the test set.

    Args:
        model: Trained model to evaluate.
        ocr_model: OCR model for digit classification.
        test_loader: DataLoader yielding dicts with x (images), y (measurements), label, and optionally sigma.
        device: torch.device; defaults to CUDA if available.
        model_name: Optional name for the model (used in print statements).
        data_range: The data range for PSNR/SSIM calculations (e.g., 1.0 for [0,1] data).
        is_prnt: Whether to print detailed timing info.
    Returns:
        dict with:
            average mse, psnr, ssim,
            average forward time per batch and per sample,
            last batch estimated and true images,
            calibrated and raw accuracy for digit labels,
            estimated and true labels,
            inference time per batch and per sample.
        Keys: 'mse_x', 'psnr_x', 'ssim_x', 'avg_forward_time_seconds_per_batch',
              'avg_forward_time_seconds_per_sample', 'x_estimated', 'x_true',
              'calibrated_accuracy_label', 'raw_accuracy_label', 'ocr_accuracy_ceiling',
              'labels_estimated', 'labels_true', 'inference_time_per_batch',
              'inference_time_per_sample'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    ocr_model.to(device)
    ocr_model.eval()

    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    n_batches = 0
    total_samples = 0
    forward_time_sum = 0.0  # seconds across all batches
    raw_correct = 0
    calibrated_correct = 0
    calibrated_total = 0
    ocr_ceiling_correct = 0
    total_label_count = 0

    pbar = tqdm(test_loader, desc="Test")
    for batch in pbar:
        batch = to_device_batch(batch, device)
        x_true = flatten_image(batch["x"])          # [B, C, N]
        y = batch["y"]
        label = batch["label"]
        sigma = batch.get("sigma", None)                # [B] or [B,1], optional

        forward_start = time.time()
        pred = model_forward(model, model_name, y, sigma)
        forward_time_sum += time.time() - forward_start
        if isinstance(pred, (tuple, list)):
            pred = pred[0]

        mse_batch = torch.mean((pred - x_true) ** 2, dim=(1, 2))  # [B]
        psnr_batch = batch_psnr(pred, x_true, data_range=data_range)  # [B]

        # reshape to images for SSIM: assume square (e.g., 28x28) per channel
        B, C, N = pred.shape
        side = int(N**0.5)
        pred_img = pred.view(B, C, side, side)
        x_true_img = x_true.view(B, C, side, side)
        ssim_batch = batch_ssim(pred_img, x_true_img, data_range=data_range)  # [B]

        mse_val = mse_batch.mean().item()
        psnr_val = psnr_batch.mean().item()
        ssim_val = ssim_batch.mean().item()

        mse_sum += mse_val
        psnr_sum += psnr_val
        ssim_sum += ssim_val
        n_batches += 1
        total_samples += x_true.shape[0]
        # Predict digit labels using OCR model
        # The input to this model is a batch of MNIST images and in model first it does x = x.view(x.size(0), -1)
        with torch.no_grad():
            # Scale images to [0,1] for OCR
            max_pred_ocr = pred_img.amax(dim=(1, 2, 3), keepdim=True)
            min_pred_ocr = pred_img.amin(dim=(1, 2, 3), keepdim=True)
            pred_img_ocr = (pred_img - min_pred_ocr) / (max_pred_ocr - min_pred_ocr)
            pred_logits = ocr_model(pred_img_ocr)
            pred_labels = pred_logits.argmax(dim=1)
            max_gt_ocr = x_true_img.amax(dim=(1, 2, 3), keepdim=True)
            min_gt_ocr = x_true_img.amin(dim=(1, 2, 3), keepdim=True)
            x_true_img_ocr = (x_true_img - min_gt_ocr) / (max_gt_ocr - min_gt_ocr)
            gt_logits = ocr_model(x_true_img_ocr)
            gt_labels = gt_logits.argmax(dim=1)
        # mask = OCR is correct on ground truth
        valid_mask = (gt_labels == label)

        if valid_mask.any():
            calibrated_correct += (pred_labels[valid_mask] == label[valid_mask]).sum().item()
            calibrated_total += valid_mask.sum().item()
        raw_correct += (pred_labels == label).sum().item()
        ocr_ceiling_correct += (gt_labels == label).sum().item()
        total_label_count += label.numel()


    if n_batches > 0:
        avg_forward_time_per_batch = forward_time_sum / n_batches
        avg_forward_time_per_sample = forward_time_sum / max(total_samples, 1)
    else:
        avg_forward_time_per_batch = 0.0
        avg_forward_time_per_sample = 0.0
    Model_name_UC = model_name.upper() if model_name is not None else "MODEL"

    raw_accuracy = raw_correct / total_label_count if total_label_count > 0 else float("nan")
    calibrated_accuracy = (
        calibrated_correct / calibrated_total if calibrated_total > 0 else float("nan")
    )
    ocr_accuracy_ceiling = (
        ocr_ceiling_correct / total_label_count if total_label_count > 0 else float("nan")
    )

    inference_time = format_time(avg_forward_time_per_sample)
    if is_prnt:
        print(
            f"Model: {Model_name_UC} \n"
            f"      Average inference time per batch: {format_time(avg_forward_time_per_batch)}\n"
            f"      Average inference time per sample: {format_time(avg_forward_time_per_sample)}"
        )

    return {
        "mse_x": mse_sum / max(n_batches, 1),
        "psnr_x": psnr_sum / max(n_batches, 1),
        "ssim_x": ssim_sum / max(n_batches, 1),
        "avg_forward_time_seconds_per_batch": avg_forward_time_per_batch,
        "avg_forward_time_seconds_per_sample": avg_forward_time_per_sample,
        "x_estimated": pred.cpu(),
        "x_true": x_true.cpu(),
        "calibrated_accuracy_label": calibrated_accuracy,
        "raw_accuracy_label": raw_accuracy,
        "ocr_accuracy_ceiling": ocr_accuracy_ceiling,
        "labels_estimated": pred_labels.cpu(),
        "labels_true": gt_labels.cpu(),
        "inference_time_per_batch": avg_forward_time_per_batch,
        "inference_time_per_sample": avg_forward_time_per_sample,
    }

@torch.no_grad()
def evaluate_ddim_model(
    model,
    gaussian_diffusion,
    ocr_model,
    test_loader,
    device: torch.device | None = None,
    data_range: float = 1.0,
    is_prnt: bool = False,
    test_timesteps: int = 50,
    ddim_discr_method: str = 'quad',
    ddim_eta: float = 0.0,
    clip_denoised: bool = True,
):
    """
    Evaluate CS-based DDIM model on the test set.

    Args:
        model: Trained CS-based DDIM model.
        gaussian_diffusion: GaussianDiffusion instance.
        ocr_model: OCR model for digit classification.
        test_loader: DataLoader yielding dicts with x_d (images), e (epsilon/measurements), label.
        device: torch.device; defaults to CUDA if available.
        data_range: Kept for API compatibility; metrics here follow DDIM_eval_cs.py (computed on [0,1]).
        is_prnt: Whether to print detailed timing info.
        test_timesteps: Number of DDIM timesteps for sampling.
        ddim_discr_method: DDIM discretization method.
        ddim_eta: DDIM eta parameter.
        clip_denoised: Whether to clip denoised samples.
    Returns:
        dict with average mse, psnr, ssim, nmse, and accuracy metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    ocr_model.to(device)
    ocr_model.eval()

    mse_list = []
    psnr_list = []
    ssim_list = []
    nmse_list = []
    n_batches = 0
    total_samples = 0
    forward_time_sum = 0.0
    raw_correct = 0
    calibrated_correct = 0
    calibrated_total = 0
    ocr_ceiling_correct = 0
    total_label_count = 0

    pbar = tqdm(test_loader, desc="DDIM CS Test")
    for batch in pbar:
        batch = to_device_batch(batch, device)
        test_img_gt = batch["x_d"].to(device)      # [B, 1, 28, 28] in [-1, 1]
        test_eps_batch = batch["e"].to(device)      # [B, 1, 28, 28] measurements
        test_lbl_batch = batch["label"].to(device)  # [B]

        batch_sz = test_img_gt.shape[0]

        forward_start = time.time()
        # CS-based DDIM sampling with conditional measurements
        recon = gaussian_diffusion.ddim_sample_conditional(
            model,
            test_eps_batch,
            28,
            batch_size=batch_sz,
            channels=1,
            ddim_timesteps=test_timesteps,
            ddim_discr_method=ddim_discr_method,
            ddim_eta=ddim_eta,
            clip_denoised=clip_denoised,
        )
        forward_time_sum += time.time() - forward_start

        gt = test_img_gt.float().to(device)

        # Metrics follow DDIM_eval_cs.py: convert [-1,1] -> [0,1] then compute
        recon_np = to_unit_interval(recon.detach().cpu().numpy())
        gt_np = to_unit_interval(gt.detach().cpu().numpy())
        for b in range(batch_sz):
            mse_list.append(float(np.mean((recon_np[b, 0] - gt_np[b, 0]) ** 2)))
            psnr_list.append(float(compute_psnr_np(recon_np[b, 0], gt_np[b, 0])))
            ssim_list.append(float(compute_ssim_np(recon_np[b, 0], gt_np[b, 0])))
            nmse_list.append(float(compute_nmse_np(recon_np[b, 0], gt_np[b, 0])))

        n_batches += 1
        total_samples += batch_sz

        # OCR evaluation
        with torch.no_grad():
            # Scale to [0, 1] for OCR
            recon_for_ocr = (recon.clamp(-1.0, 1.0) + 1.0) / 2.0
            gt_for_ocr = (gt.clamp(-1.0, 1.0) + 1.0) / 2.0
            pred_logits = ocr_model(recon_for_ocr)
            pred_labels = pred_logits.argmax(dim=1)
            gt_logits = ocr_model(gt_for_ocr)
            gt_labels = gt_logits.argmax(dim=1)

        # Calibrated accuracy: only count samples where OCR is correct on GT
        valid_mask = (gt_labels == test_lbl_batch)
        if valid_mask.any():
            calibrated_correct += (pred_labels[valid_mask] == test_lbl_batch[valid_mask]).sum().item()
            calibrated_total += valid_mask.sum().item()
        raw_correct += (pred_labels == test_lbl_batch).sum().item()
        ocr_ceiling_correct += (gt_labels == test_lbl_batch).sum().item()
        total_label_count += test_lbl_batch.numel()

    if n_batches > 0:
        avg_forward_time_per_batch = forward_time_sum / n_batches
        avg_forward_time_per_sample = forward_time_sum / max(total_samples, 1)
    else:
        avg_forward_time_per_batch = 0.0
        avg_forward_time_per_sample = 0.0

    raw_accuracy = raw_correct / total_label_count if total_label_count > 0 else float('nan')
    calibrated_accuracy = calibrated_correct / calibrated_total if calibrated_total > 0 else float('nan')
    ocr_accuracy_ceiling = ocr_ceiling_correct / total_label_count if total_label_count > 0 else float('nan')

    if is_prnt:
        print(
            f"DDIM CS Model\n"
            f"      Average inference time per batch: {format_time(avg_forward_time_per_batch)}\n"
            f"      Average inference time per sample: {format_time(avg_forward_time_per_sample)}"
        )

    return {
        "mse_x": float(np.mean(mse_list)) if mse_list else float('nan'),
        "psnr_x": float(np.mean(psnr_list)) if psnr_list else float('nan'),
        "ssim_x": float(np.mean(ssim_list)) if ssim_list else float('nan'),
        "nmse_x": float(np.mean(nmse_list)) if nmse_list else float('nan'),
        "avg_forward_time_seconds_per_batch": avg_forward_time_per_batch,
        "avg_forward_time_seconds_per_sample": avg_forward_time_per_sample,
        "x_estimated": recon.cpu(),
        "x_true": gt.cpu(),
        "calibrated_accuracy_label": calibrated_accuracy,
        "raw_accuracy_label": raw_accuracy,
        "ocr_accuracy_ceiling": ocr_accuracy_ceiling,
        "labels_estimated": pred_labels.cpu(),
        "labels_true": gt_labels.cpu(),
        "inference_time_per_batch": avg_forward_time_per_batch,
        "inference_time_per_sample": avg_forward_time_per_sample,
    }
