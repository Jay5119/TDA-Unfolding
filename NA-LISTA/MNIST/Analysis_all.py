# %% [markdown]
# ### Analysis of TDA models based on NA-LISTA on MNIST Data

# %% [markdown]
# #### Import libraries + Functions
# 1. Import Libraries and Functions
# 2. Set seed
# 3. Check data from directory
# 
# Brief: initialize the run by pulling core dependencies, seeding all RNGs, and pointing the working directory at the project root for consistent relative paths.

# %%
import os
from pathlib import Path
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import sys
import os
from datetime import datetime
from IPython import get_ipython
import copy

import warnings
warnings.filterwarnings("ignore")
# Currect Directory as Root directory
ROOT_DIR = Path.cwd()
os.chdir(ROOT_DIR)
print("Project root:", ROOT_DIR)
print("Contents:", os.listdir(ROOT_DIR))

from train_tda import train_model, train_ddim_model
from test_tda import evaluate_model, evaluate_ddim_model
from Utils_tda import add_noise_fixed_snr, to_DB, format_time
from TDA_PTDA_MLP import NA_LISTA_PTDA
from TDA_JT import NA_LISTA_JT
from TDA_DDTDA_MLP import NA_LISTA_DDTDA
from OCR import OCR_MNIST
from TAIL_LISTA import TailLISTA
from train_TailLISTA import train_tailLISTA
from DDIM_mnist_cs import GaussianDiffusion, UnetModel
from Utils_all import DictCS, try_profile_flops, _fmt_flops, cols784_to_mnist_tensor, _to_img01_from_vec, _to_img01_from_ddim
from Utils_all import NotebookTee

def set_seed_tda(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# seed = 791
seed = 436
set_seed_tda(seed) 

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# Timestamped result directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results_All")
CURRENT_RESULTS_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(CURRENT_RESULTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(CURRENT_RESULTS_DIR, "run.log")


# Activate print logger (keeps notebook output visible)
_orig_stdout = sys.stdout
sys.stdout = NotebookTee(LOG_PATH, _orig_stdout)

print("=" * 100)
print("TDA MNIST RUN STARTED")
print(f"Timestamp      : {TIMESTAMP}")
print(f"Results folder : {RESULTS_DIR}")
print("=" * 80)
print(f"Using device: {device} | Seed: {seed}\n")

# %% [markdown]
# #### Load compressed-sensing MNIST data
# Quick sanity checks on the dataset structure before any filtering or augmentation.

# %%
is_gen_data = True  # Set to True to regenerate data
if is_gen_data:
    SNRs = [-10, -5, 0, 5, 10] ## In dB
    J = len(SNRs)
    m = 500
    DATA_DIR = ROOT_DIR / "Data_500"
    ## Load the contens if DATA_DIR
    if DATA_DIR.exists():
        print("Data directory contents:", os.listdir(DATA_DIR))
    else:
        print("Data directory does not exist.")
    ## Load 'MNIST_COMPRESSED_SENSING_DATASET.npy' dictiornary and print its keys
    mnist_data = np.load(DATA_DIR / "MNIST_COMPRESSED_SENSING_DATASET.npy", allow_pickle=True).item()
    print("MNIST data loaded. Keys:", mnist_data.keys())

    
    X_all_train = mnist_data['X_train']
    X_all_test = mnist_data['X_test']
    train_labels = mnist_data['train_labels']
    test_labels  = mnist_data['test_labels']
    n = X_all_train.shape[0]  # 784
    A_np = mnist_data['A']  # shape (m, n)
    X_all_train_ddim = X_all_train * 2.0 - 1.0  # For DDIM, we can use the same sparse images but range [-1, 1]
    Y_all_train = (A_np @ X_all_train).astype(np.float32, copy=False) # shape (m, num_train_samples)
    Y_all_train_d = (A_np @ X_all_train_ddim).astype(np.float32, copy=False) # shape (m, num_train_samples)
    extra_y_need = n - m
    # select random different extra_y_need idex from Y_all_train to pad with Y_all_train_ddim
    extra_indices_train = np.random.choice(Y_all_train.shape[0], size=extra_y_need, replace=False)
    Y_all_train_ddim = np.vstack((Y_all_train_d, Y_all_train_d[extra_indices_train, :]))

    X_all_test_ddim = X_all_test * 2.0 - 1.0  # For DDIM, we can use the same sparse images but range [-1, 1]
    Y_all_test = (A_np @ X_all_test).astype(np.float32, copy=False) # shape (m, num_test_samples)
    Y_all_test_d = (A_np @ X_all_test_ddim).astype(np.float32, copy=False) # shape (m, num_test_samples)
    extra_indices_test = np.random.choice(Y_all_test.shape[0], size=extra_y_need, replace=False)
    Y_all_test_ddim = np.vstack((Y_all_test_d, Y_all_test_d[extra_indices_test, :]))

    print(f" A TDA shape: {A_np.shape}")
    print(f" Train data shapes: \n     TDA - X: {X_all_train.shape}, Y: {Y_all_train.shape} \n     DDIM - X: {X_all_train_ddim.shape}, Y: {Y_all_train_ddim.shape} \n")

    ## Only keep good (from measurements that are not totally zero) 50000 samples for training
    num_train_samples = 50000
    non_zero_indices = np.where(np.linalg.norm(Y_all_train, axis=0) > 1e-6)[0]
    print(f"Number of non-zero measurement training samples: {len(non_zero_indices)}")
    selected_indices = non_zero_indices[:num_train_samples]
    X_train = X_all_train[:, selected_indices]
    X_train_ddim = X_all_train_ddim[:, selected_indices]
    Y_train = Y_all_train[:, selected_indices]
    Y_train_ddim = Y_all_train_ddim[:, selected_indices]
    Label_train = train_labels[selected_indices]
    print(f"Filtered training data shape: Sparse Images (Vecorized) : {X_train.shape}, Compressed measuments: {Y_train.shape}")

    ## Filterout the test samples that are all zero measurements
    non_zero_test_indices = np.where(np.linalg.norm(Y_all_test, axis=0) > 1e-6)[0]
    X_test = X_all_test[:, non_zero_test_indices]
    X_test_ddim = X_all_test_ddim[:, non_zero_test_indices]
    Y_test = Y_all_test[:, non_zero_test_indices]
    Y_test_ddim = Y_all_test_ddim[:, non_zero_test_indices]
    Label_test = test_labels[non_zero_test_indices]

    print(f"Filtered testing data shape: Sparse Images (Vecorized) : {X_test.shape}, Compressed measuments: {Y_test.shape}")
    ## Print the max, min, and some unique of the data of 1 random sample
    rnd_idx = np.random.randint(X_train.shape[1])
    print(f"Random training sample index: {rnd_idx}")
    print(f"TDA:       Max pixel value: {np.max(X_train[:, rnd_idx])}, Min pixel value: {np.min(X_train[:, rnd_idx])}")
    print(f"DDIM:      Max pixel value: {np.max(X_train_ddim[:, rnd_idx])}, Min pixel value: {np.min(X_train_ddim[:, rnd_idx])}")

    ## Create J random subsets of indexes of Y
    index_subsets = []
    num_samples = Y_train.shape[1]
    for j in range(J):
        indices = np.random.choice(num_samples, size=num_samples//J, replace=False)
        index_subsets.append(indices)
    print(f"Created {J} random subsets of training data indices, each with {num_samples//J} samples.")

    ## Add noise to each subset according to the specified SNRs
    Y_train_noisy_subsets = []
    snr_outs = []
    Sigma_noises = []
    E_train_noisy_subsets = []
    for j in range(J):
        Y_subset = Y_train[:, index_subsets[j]]
        Y_ddim_subset = Y_train_ddim[:, index_subsets[j]]
        Y_noisy, E_ddim, Sigma_noise, snr_out = add_noise_fixed_snr(Y_subset, Y_ddim_subset, SNRs[j])
        Y_train_noisy_subsets.append(Y_noisy)
        E_train_noisy_subsets.append(E_ddim)
        Sigma_noises.append(Sigma_noise)
        snr_outs.append(snr_out)
    print("Added noise to training data subsets according to specified SNRs.")
    print("Avg SNRs achieved for each subset:")
    for j in range(J):
        print(f"        Subset {j+1} (SNR={SNRs[j]} dB): Avg SNR = {np.mean(snr_outs[j]):.2f} dB")

    ## Properly concatenate the noisy subsets back to form the full noisy training set with X_train, Y_train_noisy, Sigma_train with proper ordering
    Y_train_noisy = np.zeros_like(Y_train)
    E_train_noisy = np.zeros_like(Y_train_ddim)
    Sigma_train = np.zeros(Y_train.shape[1])
    for j in range(J):
        indices = index_subsets[j]
        Y_train_noisy[:, indices] = Y_train_noisy_subsets[j]
        E_train_noisy[:, indices] = E_train_noisy_subsets[j]
        Sigma_train[indices] = Sigma_noises[j]
    ## Randomly mix the training data
    perm = np.random.permutation(Y_train.shape[1])
    Y_train_noisy = Y_train_noisy[:, perm]
    E_train_noisy = E_train_noisy[:, perm]
    X_train = X_train[:, perm]
    X_train_ddim = X_train_ddim[:, perm]
    Sigma_train = Sigma_train[perm]
    Label_train = Label_train[perm]
    print(f"Final noisy training data shape: \n Compressed measuments: {Y_train_noisy.shape}, \n Sparse Images (Vecorized) : {X_train.shape}, \n Noise Std Dev array shape: {Sigma_train.shape}")

    ## Create noisy test data with SNR from SNRs like train and save per domain noisy test data
    num_test_samples = Y_test.shape[1]
    X_test_gt = {}  # Dictionary with SNR as Key and Ground Truth test data as Value
    X_test_ddim_gt = {}  # Dictionary with SNR as Key and DDIM test data as Value
    Y_test_noisy = {} # Dictionary with SNR as Key and Noisy tets data as Value
    E_test_noisy = {} # Dictionary with SNR as Key and Noisy DDIM tets data as Value    
    Sigma_test = {}    # Dictionary with SNR as Key and Noise Std Dev array as Value
    Label_test_gt = {}  # Dictionary with SNR as Key and Labels as Value
    index_subsets_test = []
    for j in range(J):
        indices = np.random.choice(num_test_samples, size=num_test_samples//J, replace=False)
        index_subsets_test.append(indices)
        Y_subset = Y_test[:, indices]
        Y_ddim_subset = Y_test_ddim[:, indices]
        Y_noisy, E_ddim, Sigma_noise, snr_out = add_noise_fixed_snr(Y_subset, Y_ddim_subset, SNRs[j])
        Y_test_noisy[SNRs[j]] = Y_noisy
        E_test_noisy[SNRs[j]] = E_ddim
        Sigma_test[SNRs[j]] = Sigma_noise
        X_test_gt[SNRs[j]] = X_test[:, indices]
        X_test_ddim_gt[SNRs[j]] = X_test_ddim[:, indices]
        Label_test_gt[SNRs[j]] = Label_test[indices]
        print(f"Test Subset {j+1} (SNR={SNRs[j]} dB): Avg SNR = {np.mean(snr_out):.2f} dB| Number of test samples: {Y_noisy.shape[1]}")


if not is_gen_data:
        # Load pre-generated data from :/home/jp/PHD/TDA MNIST/Data_Final/TDA_MNIST_Data.npy
        DATA_FINAL_DIR = ROOT_DIR / "Data_Final"
        mnist_data_final = np.load(DATA_FINAL_DIR / "TDA_MNIST_Data.npy", allow_pickle=True).item()
        print(f"MNIST final data loaded: \n Keys: {mnist_data_final.keys()}")
        X_train = mnist_data_final['X_train']
        Y_train_noisy = mnist_data_final['Y_train_noisy']
        E_train_noisy = mnist_data_final['E_train_noisy']
        X_train_ddim = mnist_data_final['X_train_ddim']
        Sigma_train = mnist_data_final['Sigma_train']
        Label_train = mnist_data_final['Label_train']

        X_test_gt = mnist_data_final['X_test_gt']
        Y_test_noisy = mnist_data_final['Y_test_noisy']
        Sigma_test = mnist_data_final['Sigma_test']
        Label_test_gt = mnist_data_final['Label_test_gt']
        E_test_noisy = mnist_data_final['E_test_noisy']
        X_test_ddim_gt = mnist_data_final['X_test_ddim']

        SNRs = mnist_data_final['SNRs']
        A_np = mnist_data_final['A']
        m = A_np.shape[0]
        n = A_np.shape[1]
        J = len(SNRs)
        print(f"Loaded data shapes: \n Train X: {X_train.shape} | Y_noisy: {Y_train_noisy.shape} | Sigma: {Sigma_train.shape} \n Test X_gt (SNR={SNRs[0]}): {X_test_gt[SNRs[0]].shape} | Y_noisy (SNR={SNRs[0]}): {Y_test_noisy[SNRs[0]].shape} | Sigma (SNR={SNRs[0]}): {Sigma_test[SNRs[0]].shape} \n A shape: {A_np.shape} ")


# %% [markdown]
# #### Create the Datasets and DataLoaders
# Maintain consistent tensor shapes for compressed measurements, sparse images, and noise standard deviations.

# %%
batch_size = 100

# --- Train dataset (needed; your code later uses train_dataset but it wasn't created) ---
train_dataset = DictCS(
    Y=torch.from_numpy(Y_train_noisy.T.astype(np.float32)).unsqueeze(1),     # (N, 1, m)
    X=torch.from_numpy(X_train.T.astype(np.float32)).unsqueeze(1),           # (N, 1, 784) vector form (for TDA)
    E=cols784_to_mnist_tensor(E_train_noisy),                                # (N, 1, 28, 28)
    X_d=cols784_to_mnist_tensor(X_train_ddim),                               # (N, 1, 28, 28)
    Label=torch.from_numpy(Label_train.astype(np.int64)),                    # (N,)
    Sigma=torch.from_numpy(Sigma_train.astype(np.float32)),                  # (N,)
)           

# --- Per-SNR test datasets/loaders ---
test_datasets = {}
test_loaders = {}
for snr_db in SNRs:
    test_datasets[snr_db] = DictCS(
        Y=torch.from_numpy(Y_test_noisy[snr_db].T.astype(np.float32)).unsqueeze(1),   # (N, 1, m)
        X=torch.from_numpy(X_test_gt[snr_db].T.astype(np.float32)).unsqueeze(1),     # (N, 1, 784)
        E=cols784_to_mnist_tensor(E_test_noisy[snr_db]),                              # (N, 1, 28, 28)
        X_d=cols784_to_mnist_tensor(X_test_ddim_gt[snr_db]),                             # (N, 1, 28, 28)
        Label=torch.from_numpy(Label_test_gt[snr_db].astype(np.int64)),               # (N,)
        Sigma=torch.from_numpy(Sigma_test[snr_db].astype(np.float32)),                # (N,)
    )
    test_loaders[snr_db] = DataLoader(test_datasets[snr_db], batch_size=batch_size, shuffle=False)

# --- Overall test dataset for average performance ---
Y_test_all = np.concatenate([Y_test_noisy[snr_db] for snr_db in SNRs], axis=1)
X_test_all = np.concatenate([X_test_gt[snr_db] for snr_db in SNRs], axis=1)
E_test_all = np.concatenate([E_test_noisy[snr_db] for snr_db in SNRs], axis=1)
X_d_test_all = np.concatenate([X_test_ddim_gt[snr_db] for snr_db in SNRs], axis=1)
Label_test_all = np.concatenate([Label_test_gt[snr_db] for snr_db in SNRs], axis=0)
Sigma_test_all = np.concatenate([Sigma_test[snr_db] for snr_db in SNRs], axis=0)

test_dataset = DictCS(
    Y=torch.from_numpy(Y_test_all.T.astype(np.float32)).unsqueeze(1),
    X=torch.from_numpy(X_test_all.T.astype(np.float32)).unsqueeze(1),
    E=cols784_to_mnist_tensor(E_test_all),
    X_d=cols784_to_mnist_tensor(X_d_test_all),
    Label=torch.from_numpy(Label_test_all.astype(np.int64)),
    Sigma=torch.from_numpy(Sigma_test_all.astype(np.float32)),
)

# Split test dataset into validation and test sets (90% test, 10% val)
val_size = int(0.1 * Y_test_all.shape[1])
test_size = Y_test_all.shape[1] - val_size
val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Created DataLoaders: Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
print(f"Shapes:\n Train Y: {train_loader.dataset.Y.shape}, X: {train_loader.dataset.X.shape}, Sigma: {train_loader.dataset.Sigma.shape}")


# %% [markdown]
# #### Create the models PTDA, JT, DDTDA , Tail-LISTA (Tight), DDIM
# Set model hyperparameters (layers, tying) once and reuse across the three architectures.

# %%
K = 15                  # Number of layers
tied = True
## A as a torch tensor
A = torch.tensor(A_np, dtype=torch.float32).unsqueeze(0)
# print("Using sensing matrix A with shape:", A.shape)
print(f"Models: \n   Number of layers = {K} \n   Tied weights = {tied}\n")

# %%
# ---- FLOPs report (uses one test sample) ----
batch_flop = next(iter(test_loader))  # DictCS batch: y, x, e, x_d, label, sigma

y_flop = batch_flop["y"][:1].to(device)                         # (1,1,m)
sigma_flop = batch_flop["sigma"][:1].to(device).view(1, 1)      # (1,1)
label_flop = batch_flop["label"][:1].to(device)                 # (1,)
eps_flop = batch_flop["e"][:1].to(device)                       # (1,1,28,28)


# %%
## Create PTDA Model

lambda_init = 0.01
thr_fc_hidden = 64
model_ptda = NA_LISTA_PTDA(m, n, K, A, C = 1,
                            lambda_init=lambda_init,
                            tied = tied, thr_fc_hidden=thr_fc_hidden).to(device)
## Print Model Summary & Number of parameters
print(f"PTDA Model Summary:")
# print(model_ptda)
PTDA_flop_param = try_profile_flops(model_ptda, inputs=(y_flop, sigma_flop), device=device)
print(f"  # of parameters: {PTDA_flop_param[1]} \n FLOPs per sample: {_fmt_flops(PTDA_flop_param[0])} \n")

# %%
## Create the DDTDA Model
lambda_init = 1.0
thr_fc_hidden = 64
model_ddtda = NA_LISTA_DDTDA(m, n, K, A, C =1,
                            lambda_init=lambda_init,
                            tied = tied, thr_fc_hidden=thr_fc_hidden).to(device)
## Print Model Summary
print(f"DDTDA Model Summary:")
# print(model_ddtda)
DDTDA_flop_param = try_profile_flops(model_ddtda, inputs=(y_flop,), device=device)
print(f"  # of parameters: {DDTDA_flop_param[1]} \n FLOPs per sample: {_fmt_flops(DDTDA_flop_param[0])} \n")

## Create Tail-LISTA (Tight) Model
num_tail_steps = 12
lambda_init = 0.01
model_tail_t = TailLISTA(M=m, N=n, num_tail_steps=num_tail_steps, K=8, A=A, C=1,
                       lambda_init=lambda_init,
                       tied=True, device=device).to(device)
# print Model Summary
print(f"Tail-LISTA (Tight) Model Summary:")
# print(model_tail_t)
TailLISTA_flop_param = try_profile_flops(model_tail_t, inputs=(y_flop,), device=device)
print(f"  # of parameters: {TailLISTA_flop_param[1]} \n FLOPs per sample: {_fmt_flops(TailLISTA_flop_param[0])} \n")

## Create DDIM Model
timesteps_train = 500
timesteps_test = 50
Unet_chnls = 128
Unet_ch_mults = (1, 2, 4)
model_ddim = UnetModel(
    in_channels=2,
    model_channels=Unet_chnls,
    out_channels=1,
    num_res_blocks=3,
    channel_mult=Unet_ch_mults,
    attention_resolutions=[7],
).to(device)
gaussian_diffusion = GaussianDiffusion(timesteps=timesteps_train)
# Print Model Summary
print(f"DDIM-Unet Model Summary:")
# print(model_ddim)
t_flop = torch.zeros(1, dtype=torch.long, device=device)
ddim_in_flop = torch.randn(1, 2, 28, 28, device=device)
DDIM_flop_param = try_profile_flops(model_ddim, inputs=(ddim_in_flop, t_flop), device=device)
print(f"  # of parameters: {DDIM_flop_param[1]} \n FLOPs per sample: {_fmt_flops(DDIM_flop_param[0])} \n")

# %%
## Create JT Model
lambda_init = 1.0
model_jt = NA_LISTA_JT(m, n, K, A, C = 1,
                        lambda_init=lambda_init,
                        tied = tied).to(device)
## Print Model Summary
print(f"JT Model Summary:")
# print(model_jt)
JT_flop_param = try_profile_flops(model_jt, inputs=(y_flop,), device=device)
print(f"  # of parameters: {JT_flop_param[1]} \n FLOPs per sample: {_fmt_flops(JT_flop_param[0])} \n")

# %%
# Load the OCR model for MNIST digit recognition
weights_path = "/home/jp/PHD/TDA MNIST/Trained_Models/mnist_ocr_state_dict.pt"
ocr_model = OCR_MNIST().to(device)
ocr_model.load_state_dict(torch.load(weights_path))
# print Model Summary
print("OCR (MLP) Model Summary:")
# print(ocr_model)
print(f"Number of parameters in OCR model: {sum(p.numel() for p in ocr_model.parameters())} \n")

# %% [markdown]
# #### Train the models
# Training loops reuse the shared dataloaders; learning rates and schedules match the script defaults.
#%% 
# Flags for training of each model
is_train_ptda = True
is_train_ddtda = True
is_train_tail_t = True
is_train_jt = True
is_train_ddim = True

# %%
if (is_train_ptda or is_train_jt or is_train_ddtda or is_train_tail_t or is_train_ddim):
    # epoch_all_tda = 250
    # epoch_print_tda = 10
    # lr_all_tda = 1e-4
    # lr_step_all_tda = 25
    # lr_decay_all_tda = 0.7

    epochs_ddim = 500
    epoch_print_ddim = 20
    lr_ddim = 1e-4
    lr_gamma_ddim = 0.7
    lr_step_ddim = 100
    ckpt_save_epochs_ddim = 50
    ckpt_dir_ddim = os.path.join(CURRENT_RESULTS_DIR, "ddim_checkpoints")
    TRAINED_MODELS_DIR = ROOT_DIR / f"Trained_Models/K_{K}_m{m}_n{n}_{TIMESTAMP}"

if is_train_ptda:
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    ## Train PTDA model
    print("PTDA ---->")
    PTDA_Trained, history_ptda = train_model(model_ptda, train_loader, val_loader,
                                            epochs=150,
                                            lr=1e-4,
                                            lr_step=10,
                                            lr_decay=0.6,
                                            device=device,
                                            model_name="ptda",
                                            print_per_epoch=5)

    ptda_test_metrics = evaluate_model(PTDA_Trained, ocr_model, test_loader, device=device, model_name = "ptda", is_prnt=True)
    ptda_model_path = TRAINED_MODELS_DIR / "ptda_model.pt"
    torch.save(PTDA_Trained.state_dict(), ptda_model_path)
    print(f"TEST Results (PTDA Model) \n MSE: {to_DB(ptda_test_metrics['mse_x'])} dB \n PSNR: {ptda_test_metrics['psnr_x']} dB \n SSIM: {ptda_test_metrics['ssim_x']} \n Calibrated Acc.: {ptda_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {ptda_test_metrics['raw_accuracy_label']*100} % \n")

if is_train_ddtda:
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    ## Train DDTDA model
    print("DDTDA ---->")
    DDTDA_Trained, history_ddtda = train_model(model_ddtda, train_loader, val_loader,
                                            epochs=150,
                                            lr=1e-4,
                                            lr_step=10,
                                            lr_decay=0.6,
                                            device=device,
                                            model_name="ddtda",
                                            print_per_epoch=5)

    ddtda_test_metrics = evaluate_model(DDTDA_Trained, ocr_model, test_loader, device=device, model_name = "ddtda", is_prnt=True)
    ddtda_model_path = TRAINED_MODELS_DIR / "ddtda_model.pt"
    torch.save(DDTDA_Trained.state_dict(), ddtda_model_path)
    print(f"TEST Results (DDTDA Model) \n MSE: {to_DB(ddtda_test_metrics['mse_x'])} dB \n PSNR: {ddtda_test_metrics['psnr_x']} dB \n SSIM: {ddtda_test_metrics['ssim_x']} \n Calibrated Acc.: {ddtda_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {ddtda_test_metrics['raw_accuracy_label']*100} % \n")

if is_train_tail_t:
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    ## Train Tail-LISTA (Tight) model
    print("TAILLISTA (Tight) ---->")
    TAIL_T_Trained_t, history_tail_t = train_tailLISTA(
        model_tail_t,
        train_loader,
        val_loader,
        epochs=250,
        lr=1e-4,
        lr_step=10,
        lr_decay=0.6,
        device=device,
        model_name="taillista",
        print_per_epoch=10)

    tail_test_metrics_t = evaluate_model(TAIL_T_Trained_t, ocr_model, test_loader, device=device, model_name = "taillista", is_prnt=True)
    tail_model_path = TRAINED_MODELS_DIR / "taillista_model.pt"
    torch.save(TAIL_T_Trained_t.state_dict(), tail_model_path)
    print(f"TEST Results (Tail-LISTA Tight Model) \n MSE: {to_DB(tail_test_metrics_t['mse_x'])} dB \n PSNR : {tail_test_metrics_t['psnr_x']} dB \n SSIM: {tail_test_metrics_t['ssim_x']} \n Calibrated Acc.: {tail_test_metrics_t['calibrated_accuracy_label']*100} % \n Raw Acc.: {tail_test_metrics_t['raw_accuracy_label']*100} % \n")

if is_train_jt:
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    ## Train JT model
    print("JT ---->")
    JT_Trained, history_jt = train_model(model_jt, train_loader, val_loader,
                                            epochs=100,
                                            lr=1e-4,
                                            lr_step=10,
                                            lr_decay=0.8,
                                            device=device,
                                            model_name="jt",
                                            print_per_epoch=10)

    jt_test_metrics = evaluate_model(JT_Trained, ocr_model, test_loader, device=device, model_name = "jt", is_prnt=True)
    jt_model_path = TRAINED_MODELS_DIR / "jt_model.pt"
    torch.save(JT_Trained.state_dict(), jt_model_path)
    print(f"TEST Results (JT Model) \n MSE: {to_DB(jt_test_metrics['mse_x'])} dB \n PSNR: {jt_test_metrics['psnr_x']} dB \n SSIM: {jt_test_metrics['ssim_x']} \n Calibrated Acc.: {jt_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {jt_test_metrics['raw_accuracy_label']*100} % \n")

if is_train_ddim:
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    ## Train DDIM model
    print("DDIM CS ---->")
    DDIM_trained, history_ddim = train_ddim_model(
        model_ddim,
        gaussian_diffusion,
        train_loader,
        epochs=epochs_ddim,
        epoch_print=epoch_print_ddim,
        lr=lr_ddim,
        lr_step=lr_step_ddim,
        lr_decay=lr_gamma_ddim,
        timesteps=timesteps_train,
        ckpt_save_epochs=ckpt_save_epochs_ddim,
        ckpt_dir=ckpt_dir_ddim,
        device=device)

    ddim_test_metrics = evaluate_ddim_model(
        DDIM_trained,
        gaussian_diffusion,
        ocr_model,
        test_loader,
        device=device,
        test_timesteps=timesteps_test,
        is_prnt=True)

    ddim_model_path = TRAINED_MODELS_DIR / "ddim_model_cs.pt"
    torch.save(DDIM_trained.state_dict(), ddim_model_path)
    print(
        f"TEST Results (DDIM CS Model) \n"
        f" MSE: {to_DB(ddim_test_metrics['mse_x'])} dB \n"
        f" PSNR: {ddim_test_metrics['psnr_x']} dB \n"
        f" SSIM: {ddim_test_metrics['ssim_x']} \n"
        f" NMSE: {to_DB(ddim_test_metrics['nmse_x'])} dB \n"
        f" Calibrated Acc.: {ddim_test_metrics['calibrated_accuracy_label']*100} % \n"
        f" Raw Acc.: {ddim_test_metrics['raw_accuracy_label']*100} % \n"
    )


# %%
# ## If not trained any one model then load the trained models from "Trained_Models" folder in root directory
if (not is_train_ptda or not is_train_jt or not is_train_ddtda or not is_train_tail_t or not is_train_ddim):
    print("Loading trained models...")
    TRAINED_MODELS_LOAD_DIR = ROOT_DIR / f"Trained_Models/Final_Models" # Loaded from a foler without timestamp and check file names
    print("Loading trained models from", TRAINED_MODELS_LOAD_DIR)
    ptda_model_path = TRAINED_MODELS_LOAD_DIR / "ptda_model.pt"
    jt_model_path = TRAINED_MODELS_LOAD_DIR / "jt_model.pt"
    ddtda_model_path = TRAINED_MODELS_LOAD_DIR / "ddtda_model.pt"
    tail_model_path = TRAINED_MODELS_LOAD_DIR / "taillista_model.pt"
    ddim_model_path = TRAINED_MODELS_LOAD_DIR / "ddim_model_cs.pt"

if not is_train_ptda:
    # Load PTDA model
    model_ptda.load_state_dict(torch.load(ptda_model_path, map_location=device))
    PTDA_Trained = model_ptda.to(device)
    print("Loaded PTDA model from", ptda_model_path)
    ptda_test_metrics = evaluate_model(PTDA_Trained, ocr_model, test_loader, device=device, model_name = "ptda", is_prnt=True)
    print(f"TEST Results (PTDA Model) \n MSE: {to_DB(ptda_test_metrics['mse_x'])} dB \n PSNR: {ptda_test_metrics['psnr_x']} dB \n SSIM: {ptda_test_metrics['ssim_x']} \n Calibrated Acc.: {ptda_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {ptda_test_metrics['raw_accuracy_label']*100} % \n")
    
if not is_train_jt:
    # Load JT model
    model_jt.load_state_dict(torch.load(jt_model_path, map_location=device))
    JT_Trained = model_jt.to(device)
    print("Loaded JT model from", jt_model_path)
    jt_test_metrics = evaluate_model(JT_Trained, ocr_model, test_loader, device=device, model_name = "jt", is_prnt=True)
    print(f"TEST Results (JT Model) \n MSE: {to_DB(jt_test_metrics['mse_x'])} dB \n PSNR: {jt_test_metrics['psnr_x']} dB \n SSIM: {jt_test_metrics['ssim_x']} \n Calibrated Acc.: {jt_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {jt_test_metrics['raw_accuracy_label']*100} % \n")

if not is_train_ddtda:
    # Load DDTDA model
    model_ddtda.load_state_dict(torch.load(ddtda_model_path, map_location=device))
    DDTDA_Trained = model_ddtda.to(device)
    print("Loaded DDTDA model from", ddtda_model_path)
    ddtda_test_metrics = evaluate_model(DDTDA_Trained, ocr_model, test_loader, device=device, model_name = "ddtda", is_prnt=True)
    print(f"TEST Results (DDTDA Model) \n MSE: {to_DB(ddtda_test_metrics['mse_x'])} dB \n PSNR: {ddtda_test_metrics['psnr_x']} dB \n SSIM: {ddtda_test_metrics['ssim_x']} \n Calibrated Acc.: {ddtda_test_metrics['calibrated_accuracy_label']*100} % \n Raw Acc.: {ddtda_test_metrics['raw_accuracy_label']*100} % \n")

if not is_train_tail_t:
    # Load Tail-LISTA model
    model_tail_t.load_state_dict(torch.load(tail_model_path, map_location=device))
    TAIL_T_Trained_t = model_tail_t.to(device)
    print("Loaded Tail-LISTA model from", tail_model_path)
    tail_test_metrics_t = evaluate_model(TAIL_T_Trained_t, ocr_model, test_loader, device=device, model_name = "taillista", is_prnt=True)
    print(f"TEST Results (Tail-LISTA Tight Model) \n MSE: {to_DB(tail_test_metrics_t['mse_x'])} dB \n PSNR : {tail_test_metrics_t['psnr_x']} dB \n SSIM: {tail_test_metrics_t['ssim_x']} \n Calibrated Acc.: {tail_test_metrics_t['calibrated_accuracy_label']*100} % \n Raw Acc.: {tail_test_metrics_t['raw_accuracy_label']*100} % \n")

if not is_train_ddim:
    # Load DDIM model
    model_ddim.load_state_dict(torch.load(ddim_model_path, map_location=device))
    DDIM_trained = model_ddim.to(device)
    print("Loaded DDIM model from", ddim_model_path)
    ddim_test_metrics = evaluate_ddim_model(
        DDIM_trained,
        gaussian_diffusion,
        ocr_model,
        test_loader,
        device=device,
        test_timesteps=timesteps_test,
        is_prnt=True,
    )
    print(
        f"TEST Results (DDIM CS Model) \n"
        f" MSE: {to_DB(ddim_test_metrics['mse_x'])} dB \n"
        f" PSNR: {ddim_test_metrics['psnr_x']} dB \n"
        f" SSIM: {ddim_test_metrics['ssim_x']} \n"
        f" NMSE: {to_DB(ddim_test_metrics['nmse_x'])} dB \n"
        f" Calibrated Acc.: {ddim_test_metrics['calibrated_accuracy_label']*100} % \n"
        f" Raw Acc.: {ddim_test_metrics['raw_accuracy_label']*100} % \n"
    )


# %% [markdown]
# #### Test the Trained models
# Evaluation uses the frozen OCR head to score reconstruction quality and downstream digit accuracy.

# %%
# --- Per-SNR test results for each model ---
print("\nSummary Per-SNR Test Results:")

# Print formatted table header
print("=" * 130)
print(f"{'SNR (dB)':<10} | {'Model':<12} | {'NMSE (dB)':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Cal. Acc. (%)':<13} | {'Raw Acc. (%)':<13}")
print("=" * 130)

for idx, snr_db in enumerate(SNRs):
    # Evaluate all models for this SNR
    jt_metrics_snr = evaluate_model(JT_Trained, ocr_model, test_loaders[snr_db], device=device, model_name="jt", is_prnt=False)
    ptda_metrics_snr = evaluate_model(PTDA_Trained, ocr_model, test_loaders[snr_db], device=device, model_name="ptda", is_prnt=False)
    ddtda_metrics_snr = evaluate_model(DDTDA_Trained, ocr_model, test_loaders[snr_db], device=device, model_name="ddtda", is_prnt=False)
    tail_metrics_snr = evaluate_model(TAIL_T_Trained_t, ocr_model, test_loaders[snr_db], device=device, model_name="taillista", is_prnt=False)
    ddim_metrics_snr = evaluate_ddim_model(DDIM_trained, gaussian_diffusion, ocr_model, test_loaders[snr_db], device=device, test_timesteps=timesteps_test, is_prnt=False)
    
    # Format SNR with proper sign
    snr_str = f"{snr_db:+d}" if snr_db != 0 else f"{snr_db}"
    
    # Print rows for each model
    print(f"{snr_str:<10} | {'JT':<12} | {to_DB(jt_metrics_snr['mse_x']):>10.2f} | {jt_metrics_snr['psnr_x']:>10.2f} | {jt_metrics_snr['ssim_x']:>8.4f} | {jt_metrics_snr['calibrated_accuracy_label']*100:>13.2f} | {jt_metrics_snr['raw_accuracy_label']*100:>13.2f}")
    print(f"{'':10} | {'P-TDA':<12} | {to_DB(ptda_metrics_snr['mse_x']):>10.2f} | {ptda_metrics_snr['psnr_x']:>10.2f} | {ptda_metrics_snr['ssim_x']:>8.4f} | {ptda_metrics_snr['calibrated_accuracy_label']*100:>13.2f} | {ptda_metrics_snr['raw_accuracy_label']*100:>13.2f}")
    print(f"{'':10} | {'DD-TDA':<12} | {to_DB(ddtda_metrics_snr['mse_x']):>10.2f} | {ddtda_metrics_snr['psnr_x']:>10.2f} | {ddtda_metrics_snr['ssim_x']:>8.4f} | {ddtda_metrics_snr['calibrated_accuracy_label']*100:>13.2f} | {ddtda_metrics_snr['raw_accuracy_label']*100:>13.2f}")
    print(f"{'':10} | {'Tail-LISTA':<12} | {to_DB(tail_metrics_snr['mse_x']):>10.2f} | {tail_metrics_snr['psnr_x']:>10.2f} | {tail_metrics_snr['ssim_x']:>8.4f} | {tail_metrics_snr['calibrated_accuracy_label']*100:>13.2f} | {tail_metrics_snr['raw_accuracy_label']*100:>13.2f}")
    print(f"{'':10} | {'DDIM':<12} | {to_DB(ddim_metrics_snr['mse_x']):>10.2f} | {ddim_metrics_snr['psnr_x']:>10.2f} | {ddim_metrics_snr['ssim_x']:>8.4f} | {ddim_metrics_snr['calibrated_accuracy_label']*100:>13.2f} | {ddim_metrics_snr['raw_accuracy_label']*100:>13.2f}")
    print("-" * 130)
print("=" * 130)
OCR_acc_ceiling = jt_metrics_snr['ocr_accuracy_ceiling'] * 100  # in %
print(f"OCR Accuracy Ceiling: {OCR_acc_ceiling:.2f} %\n")
print("=" * 130)


#%%
## Select a random 5 test sample to visualize and save per SNR
for I_id in range(5):
    CURRENT_RESULTS_DIR_FIGS = os.path.join(CURRENT_RESULTS_DIR, f"Reconstructed_Samples_Set_{I_id+1}")
    os.makedirs(CURRENT_RESULTS_DIR_FIGS, exist_ok=True)
    print(f"Generating reconstructed sample set {I_id+1} visualizations...")
    samples_to_plot = []
    for snr_db in SNRs:
        loader = test_loaders[snr_db]
        batch = next(iter(loader))  # DictCS -> collates into dict of tensors

        # pick one sample index from this batch
        bsz = batch["x"].shape[0]
        k = np.random.randint(0, bsz)

        y_noisy = batch["y"][k:k+1].to(device)                     # (1,1,m)
        x_gt_vec = batch["x"][k:k+1]                               # (1,1,784) on CPU in range 0 to 1
        sigma_1 = batch["sigma"][k:k+1].to(device).view(1, 1)      # (1,1) like your old code
        x_d_img = batch["x_d"][k:k+1]                              # (1,1,28,28) on CPU in range -1 to 1
        e_ddim = batch["e"][k:k+1].to(device)                                # (1,1,28,28) on CPU in range -1 to 1
        label_1 = batch["label"][k:k+1].item()               # scalar
        with torch.no_grad():
            x_rec_ptda = PTDA_Trained(y_noisy, sigma_1).detach().cpu()   # expected (1,1,784)
            x_rec_ddtda = DDTDA_Trained(y_noisy).detach().cpu()          # expected (1,1,784) - DDTDA only takes y
            x_rec_tail = TAIL_T_Trained_t(y_noisy).detach().cpu()  # expected (1,1,784)
            x_rec_jt = JT_Trained(y_noisy).detach().cpu()               # expected (1,1,784)
            x_rec_ddim = gaussian_diffusion.ddim_sample_conditional(
                DDIM_trained,
                e_ddim,
                image_size=28,
                batch_size=1,
                channels=1,
                ddim_timesteps=timesteps_test,
                clip_denoised=True,
            )
            x_rec_ddim = torch.as_tensor(x_rec_ddim).float().detach().cpu()
        samples_to_plot.append({
            "snr": snr_db,
            "gt": _to_img01_from_vec(x_gt_vec),
            "jt": _to_img01_from_vec(x_rec_jt),
            "ptda": _to_img01_from_vec(x_rec_ptda),
            "ddtda": _to_img01_from_vec(x_rec_ddtda),
            "tail_t": _to_img01_from_vec(x_rec_tail),
            "ddim": _to_img01_from_ddim(x_rec_ddim),
        })

    # Plot: rows = SNRs, cols = models
    col_names = ["GT", "JT", "PTDA", "DDTDA", "TailLISTA(T)", "DDIM"]
    rows_names = [f"{s}dB" for s in SNRs]
    nrows = len(samples_to_plot)
    ncols = len(col_names)


    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 5.3 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    for r, sample in enumerate(samples_to_plot):
        imgs = [
            sample["gt"],
            sample["jt"],
            sample["ptda"],
            sample["ddtda"],
            sample["tail_t"],
            sample["ddim"],
        ]
        for c in range(ncols):
            ax = axes[r, c]
            if r == 0 and c == 0:
                ax.set_title(f"SNR: {rows_names[r]} | Model:{col_names[c]}", fontsize=28)
            elif r == 0:
                ax.set_title(f"Model: {col_names[c]}", fontsize=28)
            elif c == 0:
                ax.set_title(f"SNR: {rows_names[r]}", fontsize=28)
            ax.imshow(imgs[c], cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_RESULTS_DIR_FIGS, f"Reconstructed_samples_from_test_loaders.pdf"))
    plt.close()

    # Save individual images with names as "snr_({postive/negetive}){|snr|}_{model name like gt, ptda, ddtda, tail, jt}.pdf"
    for row_idx, sample in enumerate(samples_to_plot):
        snr_str = f"snr_({'pos' if sample['snr'] >= 0 else 'neg'}){abs(sample['snr'])}"
        # Save GT
        gt_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_gt.pdf")
        plt.imsave(gt_path, sample['gt'], cmap='gray')
        # Save JT
        jt_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_jt.pdf")
        plt.imsave(jt_path, sample['jt'], cmap='gray')
        # Save PTDA
        ptda_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_ptda.pdf")
        plt.imsave(ptda_path, sample['ptda'], cmap='gray')
        # Save DDTDA
        ddtda_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_ddtda.pdf")
        plt.imsave(ddtda_path, sample['ddtda'], cmap='gray')
        # Save Tail-LISTA (Tight)
        tail_t_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_tail_t.pdf")
        plt.imsave(tail_t_path, sample['tail_t'], cmap='gray')
        # Save DDIM
        ddim_path = os.path.join(CURRENT_RESULTS_DIR_FIGS, f"{snr_str}_ddim.pdf")
        plt.imsave(ddim_path, sample['ddim'], cmap='gray')
    print(f"Individual reconstructed images saved to {CURRENT_RESULTS_DIR_FIGS}")

# Select 5 select random different test samples visualize and save for only -10 dB SNR (worst case) in a separate folder both compared single image with all separate images as above

CURRENT_RESULTS_DIR_FIGS_WORST = os.path.join(CURRENT_RESULTS_DIR, f"Reconstructed_Samples_Worst_SNR_Set")
os.makedirs(CURRENT_RESULTS_DIR_FIGS_WORST, exist_ok=True)
print(f"Generating reconstructed sample set for worst SNR visualizations...")
samples_to_plot_worst = []
worst_snr_db = min(SNRs)  # -10 dB
loader_worst = test_loaders[worst_snr_db]
rand_idx = np.random.choice(len(loader_worst.dataset), size=5, replace=False).tolist()
sampler = SubsetRandomSampler(rand_idx)
loader_worst_rand = DataLoader(
    loader_worst.dataset,
    batch_size=1,          # so each batch is exactly one sample
    sampler=sampler,
    shuffle=False
)
for batch in loader_worst_rand:
    y_noisy = batch["y"].to(device)                    # (1,1,m)
    x_gt_vec = batch["x"]                              # (1,1,784) on CPU in range 0 to 1
    sigma_1 = batch["sigma"].to(device).view(1, 1)      # (1,1) like your old code
    x_d_img = batch["x_d"]                              # (1,1,28,28) on CPU in range -1 to 1
    e_ddim = batch["e"].to(device)                                # (1,1,28,28) on CPU in range -1 to 1
    label_1 = batch["label"].item()               # scalar
    with torch.no_grad():
        x_rec_ptda = PTDA_Trained(y_noisy, sigma_1).detach().cpu()   # expected (1,1,784)
        x_rec_ddtda = DDTDA_Trained(y_noisy).detach().cpu()          # expected (1,1,784) - DDTDA only takes y
        x_rec_tail = TAIL_T_Trained_t(y_noisy).detach().cpu()  # expected (1,1,784)
        x_rec_jt = JT_Trained(y_noisy).detach().cpu()               # expected (1,1,784)
        x_rec_ddim = gaussian_diffusion.ddim_sample_conditional(
            DDIM_trained,
            e_ddim,
            image_size=28,
            batch_size=1,
            channels=1,
            ddim_timesteps=timesteps_test,
            clip_denoised=True,
        )
        x_rec_ddim = torch.as_tensor(x_rec_ddim).float().detach().cpu()
    samples_to_plot_worst.append({
        "snr": worst_snr_db,
        "gt": _to_img01_from_vec(x_gt_vec),
        "jt": _to_img01_from_vec(x_rec_jt),
        "ptda": _to_img01_from_vec(x_rec_ptda),
        "ddtda": _to_img01_from_vec(x_rec_ddtda),
        "tail_t": _to_img01_from_vec(x_rec_tail),
        "ddim": _to_img01_from_ddim(x_rec_ddim),
    })
# Plot: rows = 5 samples, cols = models
col_names = ["GT", "JT", "PTDA", "DDTDA", "TailLISTA(T)", "DDIM"]
nrows = len(samples_to_plot_worst)
ncols = len(col_names)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 5.3 * nrows))
if nrows == 1:
    axes = np.expand_dims(axes, 0)
for r, sample in enumerate(samples_to_plot_worst):
    imgs = [
        sample["gt"],
        sample["jt"],
        sample["ptda"],
        sample["ddtda"],
        sample["tail_t"],
        sample["ddim"],
    ]
    for c in range(ncols):
        ax = axes[r, c]
        if r == 0 and c == 0:
            ax.set_title(f"SNR: {sample['snr']}dB | Model:{col_names[c]}", fontsize=28)
        elif r == 0:
            ax.set_title(f"Model: {col_names[c]}", fontsize=28)
        elif c == 0:
            ax.set_title(f"SNR: {sample['snr']}dB", fontsize=28)
        ax.imshow(imgs[c], cmap="gray")
        ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"Reconstructed_samples_worst_snr_from_test_loader.pdf"))
plt.close()
# Save individual images with names as "snr_({postive/negetive}){|snr|}_{model name like gt, ptda, ddtda, tail, jt}.pdf"
for I_id, sample in enumerate(samples_to_plot_worst):
    snr_str = f"snr_({'pos' if sample['snr'] >= 0 else 'neg'}){abs(sample['snr'])}_sample_{I_id+1}"
    # Save GT
    gt_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_gt.pdf")
    plt.imsave(gt_path, sample['gt'], cmap='gray')
    # Save JT
    jt_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_jt.pdf")
    plt.imsave(jt_path, sample['jt'], cmap='gray')
    # Save PTDA
    ptda_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_ptda.pdf")
    plt.imsave(ptda_path, sample['ptda'], cmap='gray')
    # Save DDTDA
    ddtda_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_ddtda.pdf")
    plt.imsave(ddtda_path, sample['ddtda'], cmap='gray')
    # Save Tail-LISTA (Tight)
    tail_t_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_tail_t.pdf")
    plt.imsave(tail_t_path, sample['tail_t'], cmap='gray')
    # Save DDIM
    ddim_path = os.path.join(CURRENT_RESULTS_DIR_FIGS_WORST, f"{snr_str}_ddim.pdf")
    plt.imsave(ddim_path, sample['ddim'], cmap='gray')
print(f"Individual reconstructed images for worst SNR saved to {CURRENT_RESULTS_DIR_FIGS_WORST}")



if is_gen_data:
    ## Save the training and testing data used in this analysis if generated
    Data_all = {}
    Data_all['X_train'] = X_train
    Data_all['Y_train_noisy'] = Y_train_noisy
    Data_all['E_train_noisy'] = E_train_noisy
    Data_all['X_train_ddim'] = X_train_ddim
    Data_all['Sigma_train'] = Sigma_train
    Data_all['Label_train'] = Label_train

    Data_all['X_test_gt'] = X_test_gt
    Data_all['Y_test_noisy'] = Y_test_noisy
    Data_all['Sigma_test'] = Sigma_test
    Data_all['Label_test_gt'] = Label_test_gt
    Data_all['E_test_noisy'] = E_test_noisy
    Data_all['X_test_ddim'] = X_test_ddim_gt

    Data_all['X_test_all'] = X_test_all
    Data_all['Y_test_all'] = Y_test_all
    Data_all['Sigma_test_all'] = Sigma_test_all
    Data_all['Label_test_all'] = Label_test_all
    Data_all['E_test_all'] = E_test_all
    Data_all['X_d_test_all'] = X_d_test_all

    Data_all['SNRs'] = SNRs
    Data_all['A'] = A_np

    ### Save in CURRENT_RESULTS_DIR
    data_save_path = os.path.join(CURRENT_RESULTS_DIR, "TDA_MNIST_Data.npy")
    np.save(data_save_path, Data_all)
    print(f"Training and testing data saved to {data_save_path}")
