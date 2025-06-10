from datetime import datetime
from contextlib import redirect_stdout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.model_selection import train_test_split
import argparse
from contextlib import redirect_stdout
import os
#%%
def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

class LISTA(nn.Module):
    def __init__(self, m, n, numIter, device, A, L):
        super().__init__()
        self.numIter = numIter
        self.device = device
        self.n = n
        self.m = m
        self.A = A
        self.L = L
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n, bias=False)
        self.thr = nn.Parameter(torch.ones(1, 1) * 0.1 / L, requires_grad = True)  # Threshold parameter
        self.diag = nn.Parameter(torch.ones(m))  # Trainable diagonal elements
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        A = self.A
        L = self.L
        # Initialize S and B matrices
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/L) * np.matmul(A.T, A)).float().to(self.device)
        B = torch.from_numpy((1/L) * A.T).float().to(self.device)
        # Assign weights correctly without re-wrapping
        with torch.no_grad():  # Disable gradient tracking during initialization
            self._S.weight.copy_(S)
            self._W.weight.copy_(B)
            self.thr.copy_(torch.ones(1, 1).to(self.device) * 0.1 / L)

    def forward(self, y):
        d = torch.zeros(y.shape[0], self.n, device=self.device)
        outputs = []
        y_corrected = torch.mul(y, self.diag)
        for _ in range(self.numIter):
            d = soft_thr(self._W(y_corrected) + self._S(d), self.thr)
            outputs.append(d)

        return outputs


# defining custom dataset
class dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :] # since the input to the datset is tensor


def LISTA_train(X, Y, X_val, Y_val, numEpochs, numLayers, device, learning_rate, batch_size, A):
    m = int(Y.shape[0])
    n = int(X.shape[0])
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    Y_t = Y_t.float().to(device)
    Y_val_t = torch.from_numpy(Y_val.T)
    Y_val_t = Y_val_t.float().to(device)
    # we need to use ISTA to get X
    X_t = torch.from_numpy(X.T)
    X_t = X_t.float().to(device)
    X_val_t = torch.from_numpy(X_val.T)
    X_val_t = X_val_t.float().to(device)
    dataset_train = dataset(X_t, Y_t)
    dataset_valid = dataset(X_val_t, Y_val_t)
    dataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
    dataLoader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle = False)
    T = np.matmul(A.T, A)
    eg, _ = np.linalg.eig(T)
    eg = np.abs(eg)
    L = np.max(eg)*1.001
    net = LISTA(m, n, numLayers, device = device, A = A, L = L)
    net = net.float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, betas = (0.9, 0.999))
    train_loss_list = []
    valid_loss_list = []
    best_model = net; best_loss = 1e6
    # ------- Training phase --------------
    for epoch in range(numEpochs):
        if epoch == round(numEpochs*0.5):
            for param_group in optimizer.param_groups:
              param_group['lr'] *= 0.5  # Reduce learning rate
        elif epoch == round(numEpochs*0.8):
            for param_group in optimizer.param_groups:
              param_group['lr'] *= 0.5  # Reduce learning rate further
        else:
            pass
        T_tot_loss = 0
        net.train()
        for iter, data in enumerate(dataLoader_train):
            X_GT_batch, Y_batch = data
            X_batch_hat = net(Y_batch.float())  # get the outputs
            loss = criterion(X_batch_hat[numLayers-1].float(), X_GT_batch.float())
            T_tot_loss += loss.detach().cpu().data
            optimizer.zero_grad()   #clear the gradients
            loss.backward()     # compute the gradiettns
            optimizer.step()    # Update the weights
            net.zero_grad()
        train_loss_list.append(T_tot_loss.detach().data/len(dataLoader_train))
        # Validation stage
        with torch.no_grad():
            V_tot_loss = 0
            for iter, data in enumerate(dataLoader_valid):
                X_GT_batch, Y_batch = data
                X_batch_hat = net(Y_batch.float())  # get the outputs
                loss = criterion(X_batch_hat[numLayers-1].float(), X_GT_batch.float())
                V_tot_loss += loss.detach().cpu().data
            valid_loss_list.append(V_tot_loss/len(dataLoader_valid))
            if best_loss > V_tot_loss:
                best_model = net
                best_loss = V_tot_loss
        if epoch % 5 == 0 :
            print('Epoch: {:03d} :   Training Loss: {:<20.15f}  |  Validation Loss: {:<20.15f}'.format( epoch, T_tot_loss/len(dataLoader_train), V_tot_loss/len(dataLoader_valid)), flush=True)
    return best_model

def LISTA_test(net, Y, D, device):
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_lista = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_lista = X_lista.view(-1)
        X_final = X_lista[-1].cpu().numpy()
        X_final = X_final.T
    return X_final
##################################################################
#datageneration
# Dataset class
class datagen():
    def __init__(self, n, m, k):
        self.n = n
        self.m = m
        self.k = k  #this is sparsity
    def generate_sparse_signal(self, p):
        x_lst = np.zeros((self.n, p))
        for i in range(x_lst.shape[1]):
            indices=[]
            available_indices = list(range(self.n))
            for _ in range(self.k):
                available_indices = [idx for idx in available_indices if all(abs(idx - chosen_idx) >= 10 for chosen_idx in indices)]
                if not available_indices:
                    break
                chosen_index = np.random.choice(available_indices)
                indices.append(chosen_index)
                available_indices = [idx for idx in available_indices if abs(idx - chosen_index) >= 10]
            x = np.random.uniform(0.2, 1.0, self.k)
            x_lst[indices, i] = x
        return x_lst
    
    # Generate measurement matrix A
    def generate_measurement_matrix(self):
        mes_mat = np.random.randn(self.m, self.n)
        norms = np.linalg.norm(mes_mat, axis=0)
        mes_mat = mes_mat/norms
        return mes_mat
    # Generate measurements Y
    def generate_measurement(self, A, x_lst):
        return np.matmul(A, x_lst)
    
    # Add noise to measurements Y
    def add_noise(self, y_lst, sigma):
        for i in range(y_lst.shape[1]):
            noise = np.random.randn(y_lst.shape[0])*sigma
            y_lst[:,i] += noise
        return y_lst
    
    def data_gen(self, A, X, sigma_lst):
        Y = []
        # X = self.generate_sparse_signal(p)
        for sigma in sigma_lst:
            y = self.generate_measurement(A, X)
            y_noisy = self.add_noise(y, sigma)
            Y.append(y_noisy)
        return X, Y

    def random_sinusoidal_diagonal_matrix(self, m, frequency, Amp):
        # Generate m random points between 0 and 1
        random_t = np.linspace(0, 1, m)
        # Apply the sinusoidal function and ensure positive values
        diag_values = Amp*(np.sin(2 * np.pi * frequency * random_t) + 4)
        # Create the diagonal matrix
        return np.diag(diag_values)
    
    def random_uniform_matrix(self, m, low, high):
        samples = np.random.uniform(low, high, m)
        # Create the diagonal matrix
        print(samples)
        return np.diag(samples)


def evaluate_lista_per_sample(net, Y_test, X_test_true, A, device, g=0.3):
    # Run LISTA network inference
    X_est = LISTA_test(net, Y_test, A, device)
    # Convert to tensors
    X_est_tensor = torch.tensor(X_est)
    X_true_tensor = torch.tensor(X_test_true)
    # Compute per-sample squared error
    error = X_est_tensor - X_true_tensor              
    numerator = torch.sum(error ** 2, dim=0)         
    denominator = torch.sum(X_true_tensor ** 2, dim=0)  
    # Avoid divide-by-zero
    eps = 1e-8
    nmse = numerator / (denominator + eps)             
    nmse_avg = torch.mean(nmse)                      
    # Convert to decibels
    nmse_db = 10 * torch.log10(nmse_avg)
    # Calculate hit rate
    hr_list = np.array([
        np.sum((a != 0) & (np.abs(a - b) <= g * a)) / np.sum(a != 0) * 100
        for a, b in zip(X_test_true.T, X_est.T)
    ])
    hr = np.mean(hr_list)
    
    return nmse_db, hr

def data(PHI, mix_D_number, A, X, stddev_noise, n, m, N, sparsity):
    datagenerator = datagen(n, m, sparsity)
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []
    for d in range(mix_D_number):
        phi = np.diag(PHI[:, d])
        x, y = datagenerator.data_gen(np.matmul(phi, A), X, [stddev_noise])
        x = x.T
        y = y[0].T
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=42)
        x_train, x_val, x_test = x_train.T, x_val.T, x_test.T
        y_train, y_val, y_test = y_train.T, y_val.T, y_test.T
        X_train.append(x_train)
        Y_train.append(y_train)
        X_val.append(x_val)
        Y_val.append(y_val)
        X_test.append(x_test)
        Y_test.append(y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def data_mix(seed, X, Y):
    l = X[0].shape[1]
    sel = l // len(X)
    print("Samples per domain for mix dataset: ", sel, flush=True)
    x_mixed_list = []
    y_mixed_list = []
    for x, y in zip(X, Y):
        idx = seed.choice(l, sel, replace=False)  # random indices
        x_selected = x[:, idx]
        y_selected = y[:, idx]
        x_mixed_list.append(x_selected)
        y_mixed_list.append(y_selected)
    # concatenate all selected samples
    x_mixed = np.concatenate(x_mixed_list, axis=1)
    y_mixed = np.concatenate(y_mixed_list, axis=1)
    # shuffle along the sample axis
    indices = np.arange(x_mixed.shape[1])
    np.random.shuffle(indices)
    X_mixed = x_mixed[:, indices]
    Y_mixed = y_mixed[:, indices]

    return X_mixed, Y_mixed


#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%%
def run_experiment(learning_rate, numEpochs, numLayers, batch_size):
    m = 30
    n = 100
    sparsity = 3
    J = 50
    N = 10000
    seed = 80
    np.random.seed(seed)
    torch.manual_seed(42)
    
    ## For Structured Gain
    # t = np.linspace(1, 10, m, endpoint=False)  # Time vector
    # S = np.zeros((m,J))
    # for i in range(J):
    #     freq = np.random.uniform(low=0.5, high=1, size=(1,))
    #     phase_shift =  np.random.rand(1)
    #     sine_vector = 0.6 + 0.5 * np.sin(freq * t + phase_shift)
    #     S[:, i] = sine_vector
    
    # # Matrix multiplication
    # PHI = S          # Result shape (30, 25)
    # # np.save(os.path.join(save_path, "Struct_Gain_Gener_50.npy"), PHI)
    datagenerator = datagen(n,m,sparsity)
    X = datagenerator.generate_sparse_signal(numTrain)
    np.save(os.path.join(save_path, "X_Generalization_big.npy"), X) # Save the big .npy file
    A = np.load("A.npy")
    X = np.load("X_Generalization_big.npy")
    PHI = np.load("Struct_Gain_Gener_50.npy")
    print("X:", X.shape, flush=True)
    print("A:", A.shape, flush=True)
    print("PHI:", PHI.shape, flush=True)

    mix_D_number = J-5
    stddev_noise = 0.005
    N_PT = 5 * N
    PHI_rest = PHI[:, mix_D_number:]
    rest_D_number = PHI_rest.shape[1]
    X_train_gen, Y_train_gen, X_val_gen, Y_val_gen, X_test_gen, Y_test_gen = data(PHI_rest, rest_D_number, A, X[:,0:N_PT], 
                                                                                stddev_noise, n, m, N, sparsity)

    print("X_train_gen :", np.shape(X_train_gen), flush=True)
    print("Y_train_gen :", np.shape(Y_train_gen), flush=True)
    print("X_val_gen :", np.shape(X_val_gen), flush=True)
    print("Y_val_gen :", np.shape(Y_val_gen), flush=True)
    print("X_test_gen :", np.shape(X_test_gen), flush=True)
    print("Y_test_gen :", np.shape(Y_test_gen), flush=True)
    print("-"*50)
    N_PT = np.shape(X_train_gen)[2] + np.shape(X_val_gen)[2] + np.shape(X_test_gen)[2]
    print("For Each PT model dataset size is: ", N_PT, flush=True)

    for i in range(rest_D_number):
        start = time.time()
        net_PT = LISTA_train(X_train_gen[i], Y_train_gen[i], X_val_gen[i], Y_val_gen[i], numEpochs, numLayers, device, learning_rate, batch_size, A = A)
        print(f'Time taken for training of {i+ mix_D_number+ 1}th PT model :  {(time.time() - start)/60} mnt\n', flush=True)
        mse, hr = evaluate_lista_per_sample(net_PT, Y_test_gen[i], X_test_gen[i], A, device)
        domain_index = i + mix_D_number + 1  # Calculate domain index
        print(f"Domain {domain_index:<2} : MSE: {mse:>10.1f}  |  HR: {hr:>12.1f} \n", flush=True)

        # Save the model
        model_path = os.path.join(script_dir, f"PT_{i}.pth")
        torch.save(net_PT.state_dict(), model_path)
        print(f"Saved model for domain {domain_index} at {model_path}", flush=True)

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with given hyperparameters.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--numEpochs", type=int, default=500)
    parser.add_argument("--numLayers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    # Get the directory of the current Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create full path for the output file
    output_path = os.path.join(script_dir, "PT.txt")
    # Redirect all prints to PTDA.txt
    with open(output_path, "a", buffering=1) as f:
        with redirect_stdout(f):
            # Print date and time at the start of the file
            now = datetime.now()
            print("=" * 50)
            print("Experiment run started at:", now.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 50)
            # Print arguments used
            print("Arguments used:")
            print(f"  Learning rate: {args.learning_rate}")
            print(f"  Num epochs:    {args.numEpochs}")
            print(f"  Num layers:    {args.numLayers}")
            print(f"  Batch size:    {args.batch_size}")
            print("=" * 50)
            run_experiment(args.learning_rate, args.numEpochs, args.numLayers, args.batch_size)
            now = datetime.now()
            print("=" * 50)
            print("Experiment run ended at:", now.strftime("%Y-%m-%d %H:%M:%S"))
            print("=" * 50)

        ## Run:
        ##  python3 PT.py --learning_rate 0.0005 --numEpochs 250 --numLayers 10 --batch_size 64
