# -*- coding: utf-8 -*-
"""Wiener_normal_gain_calibration_PT1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-ixr1SNorJ7qSqXthKHLbsXzeWyxK00W
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

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
        """Randomly initializes weights for W, S, and thr """
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

    Train_size = Y.shape[1]

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


    # dataset_train = dataset(X_t, Y_t)
    dataset_train = dataset(X_t, Y_t)
    dataset_valid = dataset(X_val_t, Y_val_t)
    print('DataSet size is: ', dataset_train.__len__())
    dataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
    dataLoader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle = False)

    T = np.matmul(A.T, A)
    eg, _ = np.linalg.eig(T)
    eg = np.abs(eg)
    L = np.max(eg)*1.001

    # Numpy Random State if rng (passed through arguments)
    net = LISTA(m, n, numLayers, device = device, A = A, L = L)
    net = net.float().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, betas = (0.9, 0.999))

    train_loss_list = []
    valid_loss_list = []

    best_model = net; best_loss = 1e6
    lr = learning_rate
    # ------- Training phase --------------
    print('Training >>>>>>>>>>>>>>')
    for epoch in range(numEpochs):

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
            net.eval()
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

        if epoch % 1 == 0 :
            print('Epoch: {:03d} :   Training Loss: {:<20.15f}  |  Validation Loss: {:<20.15f}'.format( epoch, T_tot_loss/len(dataLoader_train), V_tot_loss/len(dataLoader_valid)))

    print('Training Completed')

    plt.plot(train_loss_list)
    plt.show()
    plt.plot(valid_loss_list)
    plt.show()

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

#datageneration
# Dataset class
class datagen():
    def __init__(self, n, m, k, dist):
        self.n = n
        self.m = m
        self.k = k
        self.dist = dist

    def generate_sparse_signal(self, p):
        x_lst = np.zeros((self.n, p))
        for i in range(x_lst.shape[1]):
            indices=[]
            available_indices = list(range(self.n))

            for mf in range(self.k):
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

#%%
m = 30 #N_y
n = 100 #N_x
sparsity = 3 #K

seed = 80
np.random.seed(seed)
torch.manual_seed(42)

dist = 10
numTrain = int(43000*3)
loss_function = nn.MSELoss()

datagenerator = datagen(n, m, sparsity, dist)
#use for random A initialization
A = datagenerator.generate_measurement_matrix() #dictionary matrix
X = datagenerator.generate_sparse_signal(numTrain) #sparse x common vector generation

A_bf = np.load("A.npy")
X_bf = np.load("X.npy")
print(np.array_equal(A_bf, A), np.array_equal(X_bf, X))

PHI = np.load("Random_gains_close_to_zero.npy")
c1 = PHI[:, 0]
c2 = PHI[:, 1]
c3 = PHI[:, 2]

phi1 = np.diag(c1)
phi2 = np.diag(c2)
phi3 = np.diag(c3)


def data(phi1, stddev_noise):
    X1, Y1 = datagenerator.data_gen(np.matmul(phi1, A), X, [stddev_noise])
    print(isinstance(X1, np.ndarray))
    SNR = 10*np.log10(np.mean(np.matmul(A, X1)**2, axis = 0)/stddev_noise**2)
    print("SNR:", np.mean(SNR), "SNR std:", np.std(SNR))

    X1 = X1.T
    Y1 = Y1[0].T

    X1_train_val, X1_test, Y1_train_val, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)
    X1_train, X1_val, Y1_train, Y1_val = train_test_split(X1_train_val, Y1_train_val, test_size=0.3, random_state=42)

    X1_train, X1_val, X1_test = X1_train.T, X1_val.T, X1_test.T
    Y1_train, Y1_val, Y1_test = Y1_train.T, Y1_val.T, Y1_test.T

    return X1_train, Y1_train, X1_val, Y1_val, X1_test, Y1_test


#computing X and Y dataset for LISTA#

X1_train, Y1_train, X1_val, Y1_val, X1_test, Y1_test = data(phi1, 0.005)

X2_train, Y2_train, X2_val, Y2_val, X2_test, Y2_test = data(phi2, 0.005)

X3_train, Y3_train, X3_val, Y3_val, X3_test, Y3_test = data(phi3, 0.005)


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print("device: ",device)

def data_mix(seed, x1, y1, x2, y2, x3, y3):
    l = x1.shape[1]
    print("l : ", l)
    sel = l//1

    # sigma1, sigma2, sigma3 = np.array(sigma1), np.array(sigma2), np.array(sigma3)

    idx1 = seed.choice(l, sel, replace=False)
    idx2 = seed.choice(l, sel, replace=False)
    idx3 = seed.choice(l, sel, replace=False)


    x_mixed = np.concatenate((x1[:,idx1], x2[:,idx2], x3[:,idx3]), axis=1)
    y_mixed = np.concatenate((y1[:,idx1], y2[:,idx2], y3[:,idx3]), axis=1)
    # sigma_mixed = np.concatenate((sigma1[:, idx1], sigma2[:, idx2], sigma3[:, idx3]), axis = 1)

    indices = np.arange(x_mixed.shape[1])
    print("indices", indices)
    np.random.shuffle(indices)
    X_mixed = x_mixed[:, indices]
    Y_mixed = y_mixed[:, indices]
    # Sigma_mixed = sigma_mixed[:, indices]


    return X_mixed, Y_mixed


seed_mixed = 10
# alpha_mixed = torch.tensor([[0.1132, 0.0616, 0.1100, 0.1029, 0.1028, 0.1041, 0.1061, 0.1080, 0.1095,0.1108, 0.1119, 0.1128, 0.1136, 0.1143, 0.1149]]).T
np_seed_mixed = np.random.RandomState(seed_mixed)
X_train_mixed, Y_train_mixed = data_mix(np_seed_mixed, X1_train, Y1_train, X2_train, Y2_train, X3_train, Y3_train)
X_val_mixed, Y_val_mixed = data_mix(np_seed_mixed, X1_val, Y1_val, X2_val, Y2_val, X3_val, Y3_val)

#%%

x = c1
y = c2
dot_product = np.dot(x, y)
# L2 norms
norm_x = np.linalg.norm(x, ord=2)
norm_y = np.linalg.norm(y, ord=2)
# Result
result = dot_product / (norm_x * norm_y)

print("corelation between c1 and c2:", result)

print(np.linalg.norm(x - y)/(norm_x*norm_y))

x = c2
y = c3
dot_product = np.dot(x, y)
# L2 norms
norm_x = np.linalg.norm(x, ord=2)
norm_y = np.linalg.norm(y, ord=2)
# Result
result = dot_product / (norm_x * norm_y)

print("corelation between c2 and c3:", result)

print(np.linalg.norm(x - y)/(norm_x*norm_y))

x = c3
y = c1
dot_product = np.dot(x, y)
# L2 norms
norm_x = np.linalg.norm(x, ord=2)
norm_y = np.linalg.norm(y, ord=2)
# Result
result = dot_product / (norm_x * norm_y)

print("corelation between c3 and c1:", result)

print(np.linalg.norm(x - y)/(norm_x*norm_y))

learning_rate = 5e-5
numEpochs = 100
numLayers = 10
batch_size = 32

start = time.time()

net_mixed = LISTA_train(X1_train, Y1_train, X1_val, Y1_val, numEpochs, numLayers, device, learning_rate, batch_size, A = A)

print(f'time taken is {time.time() - start}')

for name, param in net_mixed.named_parameters():
    if param.requires_grad:
        print(f"Trainable Parameter: {name}")
    else:
        print(f"Non-Trainable Parameter: {name}")

X1_test = np.load("random/X1_test.npy")
X2_test = np.load("random/X2_test.npy")
X3_test = np.load("random/X3_test.npy")

Y1_test = np.load("random/Y1_test.npy")
Y2_test = np.load("random/Y2_test.npy")
Y3_test = np.load("random/Y3_test.npy")

# torch.save(net_mixed, "/content/drive/My Drive/Weiner/random_close_to_zero/PT1_epoch_100_lr_5e_5.pth")

#testing mse and HR metric function
def test_mse_hr1_hr2(net_mixed_1, Y1_test, A, device, X1_test):
  g = 0.3
  X_est_5 = LISTA_test(net_mixed_1, Y1_test, A, device)

  gt = X1_test
  pred = X_est_5

#*************************************
  error = pred - gt
  numerator = np.sum(error ** 2, axis=0)      # per-sample squared error
  denominator = np.sum(gt ** 2, axis=0)       # per-sample signal power
  # To avoid division by zero
  eps = 1e-8
  nmse = numerator / (denominator + eps)      # shape: [batch_size]
  print(nmse.shape)
  nmse_avg = np.mean(nmse)
  # Convert to decibels
  nmse_db = 10 * np.log10(nmse_avg)
#*************************************
  mse_5 = nmse_db

  hr_list = np.array([np.sum((a != 0) & (np.abs(a - b) <= g*a)) / np.sum(a != 0) * 100 for a, b in zip(X1_test.T, X_est_5.T)])
  hr_5 = np.mean(hr_list)


  print(mse_5, hr_5)

test_mse_hr1_hr2(net_mixed, Y1_test, A, device, X1_test)

test_mse_hr1_hr2(net_mixed, Y2_test, A, device, X2_test)

test_mse_hr1_hr2(net_mixed, Y3_test, A, device, X3_test)