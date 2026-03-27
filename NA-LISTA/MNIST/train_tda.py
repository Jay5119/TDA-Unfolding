# train.py
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import copy
from OCR import OCR_MNIST


from Utils_tda import (
    flatten_image,
    batch_psnr,
    to_device_batch,
    model_forward,
    format_time,
)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    lr_step : int = 5,
    lr_decay: float = 0.6,
    device: torch.device | None = None,
    model_name: str | None = None,
    data_range: float = 1.0,
    show_progress: bool = True,
    leave_progress:  bool = False,
    print_per_epoch: int = 5,
):
    """
    Generic trainer for JT, PTDA, and DDTDA LISTA variants.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training, yielding dicts with keys "x", "y", "sigma" (optional), "label".
        val_loader: DataLoader for validation, same format as train_loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        lr_step: Step size for learning rate scheduler.
        lr_decay: Gamma for learning rate scheduler.
        device: torch.device; defaults to CUDA if available.
        model_name: Optional name of the model for logging purposes.
        data_range: Maximum pixel value for PSNR calculation (e.g., 1.0 for normalized images).
        show_progress: Whether to show tqdm progress bars.
        leave_progress: Whether to leave tqdm bars after completion.
        print_per_epoch: Print status every this many epochs.
    Returns:
        best_model as trained model, history dict with training/validation loss, PSNR, OS NMSE, and calibrated OCR accuracy per epoch.
        history Keys:
            "train_loss": List of average training losses per epoch.
            "val_loss": List of average validation losses per epoch.
            "train_psnr": List of average training PSNR values per epoch.
            "val_psnr": List of average validation PSNR values per epoch.
            "val_os_nmse": List of average validation OS NMSE values per epoch.
            "val_cal_acc": List of validation calibrated OCR accuracy per epoch (only on samples where OCR is correct on GT).
    """
    Model_name_Upp= model_name.upper() if model_name else "UNNAMED MODEL"
    print(f"\n{'='*60}")
    print(
        f"Starting training for {Model_name_Upp if Model_name_Upp else 'Unnamed Model'}"
        f"\nDevice: {device}"
        f"\nEpochs: {epochs}"
        f"\nLearning Rate: {lr}"
        f"\nLR Step: {lr_step}"
        f"\nLR Decay: {lr_decay}"
        f"\nBatch Size: {train_loader.batch_size}"
        f"\nTraining Samples: {len(train_loader.dataset)}"
        f"\nValidation Samples: {len(val_loader.dataset)}"
    )
    print(f"{'='*60}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "/home/jp/PHD/TDA MNIST/Trained_Models/mnist_ocr_state_dict.pt"
    ocr_model = OCR_MNIST().to(device)
    ocr_model.load_state_dict(torch.load(weights_path))
    ocr_model.eval()

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": [], "val_os_nmse": [], "val_cal_acc": []}

    best_val= float("inf")
    best_model_dir = None
    best_model = copy.deepcopy(model)

    train_start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} [Train]",
                    disable=not show_progress, leave=leave_progress)

        train_loss_sum = 0.0
        train_psnr_sum = 0.0
        train_sample_count = 0
        for step, batch in enumerate(pbar, start=1):
            batch = to_device_batch(batch, device)
            x_true = flatten_image(batch["x"])          # [B, C, N]
            y = batch["y"]                        # [B, C, M]
            sigma = batch.get("sigma", None)                # [B] or [B,1], optional
            batch_size = x_true.size(0)

            pred = model_forward(model, model_name, y, sigma)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]  # take main output if model returns (Z, Zs)

            loss = criterion(pred, x_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                psnr_batch = batch_psnr(pred, x_true, data_range=data_range)  # [B]

            train_loss_sum += loss.item() * batch_size
            train_psnr_sum += psnr_batch.sum().item()
            train_sample_count += batch_size
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}", 
                    psnr=f"{psnr_batch.mean().item():.2f}dB"
                )

        history["train_loss"].append(train_loss_sum / max(train_sample_count, 1))
        history["train_psnr"].append(train_psnr_sum / max(train_sample_count, 1))

        # Validation
        model.eval()
        pbar_val = tqdm(val_loader, desc=f"Ep {epoch+1}/{epochs} [Val]", leave=leave_progress)
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_os_nmse_sum = 0.0
        val_correct_valid = 0
        val_total_valid = 0
        val_sample_count = 0

        with torch.no_grad():
            for batch in pbar_val:
                batch = to_device_batch(batch, device)
                x_true = flatten_image(batch["x"])  # [B, C, N]
                y = batch["y"]
                sigma = batch.get("sigma", None)
                batch_size = x_true.size(0)
                labels = batch["label"]

                pred = model_forward(model, model_name, y, sigma)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]

                loss = criterion(pred, x_true)  # scalar
                psnr_batch = batch_psnr(pred, x_true, data_range=data_range)  # [B]

                # Accumulate loss and PSNR weighted by batch size
                val_loss_sum += loss.item() * batch_size
                val_psnr_sum += psnr_batch.sum().item()
                val_sample_count += batch_size

                # Compute OS NMSE per sample directly on flattened inputs [B, C, N]
                support = (x_true != 0)
                os_mask = ~support
                num = (pred**2 * os_mask).sum(dim=(1, 2))  # [B]
                denom = (x_true**2 * support).sum(dim=(1, 2))  # [B]
                os_nmse_per_sample = num / (denom + 1e-12)  # [B]
                val_os_nmse_sum += os_nmse_per_sample.sum().item()

                # Compute calibrated OCR accuracy: accuracy on predicted images
                # where OCR is correct on ground truth
                n_pix = batch["x"].size(-1)
                side = int(n_pix ** 0.5)
                if side * side != n_pix:
                    raise ValueError(f"Expected square images, got {n_pix} pixels")

                x_true_4d = batch["x"].view(batch_size, -1, side, side)  # [B, C, H, W]
                pred_4d = pred.view(batch_size, -1, side, side)  # [B, C, H, W]

                pred_logits = ocr_model(pred_4d)
                pred_labels = pred_logits.argmax(dim=1)  # [B]

                gt_logits = ocr_model(x_true_4d)
                gt_labels = gt_logits.argmax(dim=1)  # [B]

                valid_mask = (gt_labels == labels)  # [B]
                val_correct_valid += (pred_labels[valid_mask] == labels[valid_mask]).sum().item()
                val_total_valid += valid_mask.sum().item()

                pbar_val.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr_batch.mean().item():.2f}dB")

        history["val_loss"].append(val_loss_sum / max(val_sample_count, 1))
        history["val_psnr"].append(val_psnr_sum / max(val_sample_count, 1))
        history["val_os_nmse"].append(val_os_nmse_sum / max(val_sample_count, 1))
        history["val_cal_acc"].append(val_correct_valid / max(val_total_valid, 1) if val_total_valid > 0 else float("nan"))
        ## Get epochs digit length for formatting
        epoch_digits = len(str(epochs))
        if epoch == 0 or (epoch + 1) % print_per_epoch == 0 or (epoch + 1) == epochs:
            tqdm.write(
                f"Epoch {epoch+1:0{epoch_digits}d}/{epochs:0{epoch_digits}d} | "
                f"train_loss: {history['train_loss'][-1]:.6f} train_psnr: {history['train_psnr'][-1]:.4f}dB | "
                f"val_loss: {history['val_loss'][-1]:.6f} val_psnr: {history['val_psnr'][-1]:.4f}dB | "
                f"val_cal_acc: {history['val_cal_acc'][-1]*100:.2f} % | "
                f"val_os_nmse: {history['val_os_nmse'][-1]:.6f}"
            )
        scheduler.step()
        # Save best model
        if history['val_loss'][-1] < best_val:
            best_val = history['val_loss'][-1]
            best_model_dir = model.state_dict()

    best_model.load_state_dict(best_model_dir)
    ## Print Final Training Time Here
    total_train_time = time.time() - train_start_time
    print(f"\nTotal training time: {format_time(total_train_time)}")
    print(f"{'='*60}\n")
    
    return best_model, history

## DDIM Train (CS-based)
def train_ddim_model(
    model,
    gaussian_diffusion,
    train_loader,
    epochs: int = 15,
    epoch_print: int = 5,
    lr: float = 1e-4,
    lr_step: int = 5,
    lr_decay: float = 0.6,
    timesteps: int = 500,
    device: torch.device | None = None,
    show_progress: bool = True,
    leave_progress: bool = False,
    ckpt_save_epochs: int = 5,
    ckpt_dir: str | None = None,
    save_name: str = "ddim_cs_model",
):
    """
    Trainer for CS-based DDIM model (no validation).
    Args:
        model: DDIM model instance.
        gaussian_diffusion: GaussianDiffusion instance for q_sample.
        train_loader: DataLoader yielding dicts with keys x_d (clean image), e (measurement).
        epochs: Number of epochs.
        epoch_print: Print status every this many epochs.
        lr: Learning rate.
        lr_step: Step size for learning rate scheduler.
        lr_decay: Gamma for learning rate scheduler.
        timesteps: Number of diffusion steps (T).
        device: torch.device; defaults to CUDA if available.
        show_progress: Whether to show tqdm progress bars.
        leave_progress: Whether to leave tqdm bars after completion.
        ckpt_save_epochs: Save checkpoint every this many epochs.
        ckpt_dir: Directory to save checkpoints; if None, no checkpoints are saved.
        save_name: Base name for checkpoint files.
    Returns:
        model as trained model, history dict with training loss and steps.
        history Keys:
            "train_loss": List of training losses per step.
            "steps": List of cumulative step counts corresponding to each loss entry.
    """
    import torch.nn.functional as F
    import os

    Model_name_Upp = "DDIM CS MODEL"
    print(f"\n{'='*60}")
    print(
        f"Starting training for {Model_name_Upp}"
        f"\nDevice: {device}"
        f"\nEpochs: {epochs}"
        f"\nLearning Rate: {lr}"
        f"\nLR Step: {lr_step}"
        f"\nLR Decay: {lr_decay}"
        f"\nBatch Size: {train_loader.batch_size}"
        f"\nTraining Samples: {len(train_loader.dataset)}"
    )
    print(f"{'='*60}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    history = {"train_loss": [], "steps": []}
    total_steps = epochs * len(train_loader)
    step_count = 0
    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]",
                    disable=not show_progress, leave=leave_progress)

        for step, batch in enumerate(pbar, start=1):
            batch = to_device_batch(batch, device)
            images = batch["x_d"].to(device)
            eps_target = batch["e"].to(device)

            optimizer.zero_grad()
            batch_size = images.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            x_noisy = gaussian_diffusion.q_sample(images, t, noise=noise)
            x_cond = torch.cat([x_noisy, eps_target], dim=1)
            predicted_noise = model(x_cond, t)
            loss = F.mse_loss(noise, predicted_noise, reduction='mean')

            step_count += 1
            history["train_loss"].append(float(loss.item()))
            history["steps"].append(step_count)

            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    step=f"{step_count}/{total_steps}"
                )

        scheduler.step()
        epoch_d = len(str(epochs))
        step_d = len(str(total_steps))
        if epoch == 0 or (epoch + 1) % epoch_print == 0 or (epoch+1) == ckpt_save_epochs or (epoch + 1) == epochs:
            tqdm.write(
                f"Epoch: {epoch+1:>{epoch_d}}/{epochs:d}  Step: {step_count:>{step_d}}/{total_steps:d}  "
                f"Loss {loss.item():.10f}  Lr {optimizer.param_groups[0]['lr']:.10f}"
            )
        # Save checkpoint
        if ckpt_dir is not None:
            if (epoch + 1) % ckpt_save_epochs == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{save_name}_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint at epoch{epoch+1}: {ckpt_path}")

    total_train_time = time.time() - train_start_time
    print(f"\nTotal training time: {format_time(total_train_time)}")
    print(f"{'='*60}\n")

    return model, history

    
    
