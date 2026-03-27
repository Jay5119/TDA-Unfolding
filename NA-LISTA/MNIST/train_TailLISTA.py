# train_tailLISTA.py
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
    compute_psnr_from_mse,
    to_device_batch,
    format_time,
)

def compute_tailLISTA_loss(all_Zs, x_true, criterion):
    """
    Compute Tail-LISTA loss: average MSE across all intermediate estimates.
    
    As per paper equation (19):
    L = (1/NM) * sum_{j=1}^{N} sum_{k=1}^{M} f(x*_j, x^k_j)
    
    Args:
        all_Zs: List of M intermediate estimates, each [B, C, N]
        x_true: Ground truth [B, C, N]
        criterion: Loss function (MSE)
    
    Returns:
        loss: Scalar tensor, averaged across all tail steps
    """
    num_tail_steps = len(all_Zs)
    total_loss = 0.0
    
    # Sum loss over all tail steps
    for Z_k in all_Zs:
        total_loss += criterion(Z_k, x_true)
    
    # Average over tail steps (division by M)
    # Note: criterion already averages over batch (N), so this gives 1/(NM) * sum
    avg_loss = total_loss / num_tail_steps
    
    return avg_loss


def train_tailLISTA(
    model,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 4e-4,
    lr_step: int = 50,
    lr_decay: float = 0.5,
    device: torch.device | None = None,
    model_name: str = "Tail-LISTA",
    data_range: float = 1.0,
    show_progress: bool = True,
    leave_progress: bool = False,
    print_per_epoch: int = 5,
):
    """
    Train a Tail-LISTA model with comprehensive logging and best model checkpointing.

    Args:
        model: Tail-LISTA model instance to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Total number of training epochs.
        lr: Initial learning rate for the optimizer.
        lr_step: Number of epochs after which to decay the learning rate.
        lr_decay: Multiplicative factor for learning rate decay.
        device: torch.device to use for training (e.g., "cuda" or "cpu"). If None, auto-detects.
        model_name: String name of the model (for logging purposes).
        data_range: Maximum possible pixel value (used for PSNR calculation).
        show_progress: Whether to display tqdm progress bars during training/validation.
        leave_progress: Whether to leave tqdm progress bars after completion.
        print_per_epoch: Frequency (in epochs) to print training/validation metrics.
    Returns:
        best_model as trained model, history dict with training/validation metrics.
        history Keys:
            "train_loss": List of average training losses per epoch.
            "val_loss": List of average validation losses per epoch.
            "train_psnr": List of average training PSNR values per epoch.
            "val_psnr": List of average validation PSNR values per epoch.
            "val_os_nmse": List of average validation Off-Support NMSE values per epoch.
            "val_cal_acc": List of average validation calibrated OCR accuracy per epoch.
    """
    Model_name_Upp = model_name.upper()
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
        f"\nValidation Samples: {len(val_loader.dataset)}"
    )
    print(f"{'='*60}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load OCR model for calibrated accuracy
    weights_path = "/home/jp/PHD/TDA MNIST/Trained_Models/mnist_ocr_state_dict.pt"
    ocr_model = OCR_MNIST().to(device)
    ocr_model.load_state_dict(torch.load(weights_path))
    ocr_model.eval()

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)

    history = {
        "train_loss": [], 
        "val_loss": [], 
        "train_psnr": [], 
        "val_psnr": [], 
        "val_os_nmse": [], 
        "val_cal_acc": []
    }

    best_val = float("inf")
    best_model_dir = None
    best_model = copy.deepcopy(model)

    train_start_time = time.time()

    for epoch in range(epochs):
        # ========== TRAINING ==========
        model.train()
        pbar = tqdm(
            train_loader, 
            desc=f"Ep {epoch+1}/{epochs} [Train]",
            disable=not show_progress, 
            leave=leave_progress
        )

        train_loss_sum = 0.0
        train_psnr_sum = 0.0
        train_sample_count = 0
        
        for step, batch in enumerate(pbar, start=1):
            batch = to_device_batch(batch, device)
            x_true = flatten_image(batch["x"])  # [B, C, N]
            y = batch["y"]  # [B, C, M]
            batch_size = x_true.size(0)

            # Forward pass: get final output and all intermediate estimates
            Z_final, all_Zs = model(y, all_outputs=True)

            # Compute Tail-LISTA loss: average MSE over all tail steps
            loss = compute_tailLISTA_loss(all_Zs, x_true, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute metrics on final output for monitoring
            with torch.no_grad():
                psnr_batch = batch_psnr(Z_final, x_true, data_range=data_range)  # [B]

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

        # ========== VALIDATION ==========
        model.eval()
        pbar_val = tqdm(
            val_loader, 
            desc=f"Ep {epoch+1}/{epochs} [Val]", 
            disable=not show_progress,
            leave=leave_progress
        )
        
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
                y = batch["y"]  # [B, C, M]
                batch_size = x_true.size(0)
                labels = batch["label"]

                # Forward pass
                Z_final, all_Zs = model(y, all_outputs=True)

                # Compute Tail-LISTA loss on all intermediate outputs
                loss = compute_tailLISTA_loss(all_Zs, x_true, criterion)

                # Compute metrics on final output
                psnr_batch = batch_psnr(Z_final, x_true, data_range=data_range)  # [B]

                # Accumulate loss and PSNR weighted by batch size
                val_loss_sum += loss.item() * batch_size
                val_psnr_sum += psnr_batch.sum().item()
                val_sample_count += batch_size

                # Compute OS NMSE (Off-Support NMSE) per sample
                support = (x_true != 0)
                os_mask = ~support
                num = (Z_final**2 * os_mask).sum(dim=(1, 2))  # [B]
                denom = (x_true**2 * support).sum(dim=(1, 2))  # [B]
                os_nmse_per_sample = num / (denom + 1e-12)  # [B]
                val_os_nmse_sum += os_nmse_per_sample.sum().item()

                # Compute calibrated OCR accuracy
                n_pix = batch["x"].size(-1)
                side = int(n_pix ** 0.5)
                if side * side != n_pix:
                    raise ValueError(f"Expected square images, got {n_pix} pixels")

                x_true_4d = batch["x"].view(batch_size, -1, side, side)  # [B, C, H, W]
                pred_4d = Z_final.view(batch_size, -1, side, side)  # [B, C, H, W]

                pred_logits = ocr_model(pred_4d)
                pred_labels = pred_logits.argmax(dim=1)  # [B]

                gt_logits = ocr_model(x_true_4d)
                gt_labels = gt_logits.argmax(dim=1)  # [B]

                valid_mask = (gt_labels == labels)  # [B]
                val_correct_valid += (pred_labels[valid_mask] == labels[valid_mask]).sum().item()
                val_total_valid += valid_mask.sum().item()

                pbar_val.set_postfix(
                    loss=f"{loss.item():.6f}", 
                    psnr=f"{psnr_batch.mean().item():.2f}dB"
                )
                
        scheduler.step()
        history["val_loss"].append(val_loss_sum / max(val_sample_count, 1))
        history["val_psnr"].append(val_psnr_sum / max(val_sample_count, 1))
        history["val_os_nmse"].append(val_os_nmse_sum / max(val_sample_count, 1))
        history["val_cal_acc"].append(
            val_correct_valid / max(val_total_valid, 1) if val_total_valid > 0 else float("nan")
        )
        epoch_digits = len(str(epochs))

        if epoch == 0 or (epoch + 1) % print_per_epoch == 0 or (epoch + 1) == epochs:
            tqdm.write(
                f"Epoch {epoch+1:0{epoch_digits}d}/{epochs:0{epoch_digits}d} | "
                f"train_loss: {history['train_loss'][-1]:.6f} train_psnr: {history['train_psnr'][-1]:.4f}dB | "
                f"val_loss: {history['val_loss'][-1]:.6f} val_psnr: {history['val_psnr'][-1]:.4f}dB | "
                f"val_cal_acc: {history['val_cal_acc'][-1]*100:.2f}% | "
                f"val_os_nmse: {history['val_os_nmse'][-1]:.6f}"
            )
        
        # Save best model based on validation loss
        if history['val_loss'][-1] < best_val:
            best_val = history['val_loss'][-1]
            best_model_dir = model.state_dict()

    # Load best model weights
    best_model.load_state_dict(best_model_dir)
    total_train_time = time.time() - train_start_time
    print(f"Total training time: {format_time(total_train_time)}")
    print(f"{'='*60}\n")
    
    return best_model, history