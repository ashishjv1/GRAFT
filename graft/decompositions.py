import numpy as np
from tqdm import tqdm
import torch

def index_sel(vh, r):
    '''
    Performs a Maximum Vol Index selection on batch-wise Vt elements from index_decomposition
    A - is the input matrix (dataset)
    r - is the desired rank
    device - cpu or gpu
    '''
    # Move input tensor to GPU
    device = "cuda" if torch.cuda.is_available() and isinstance(vh, torch.Tensor) else "cpu"

    if isinstance(vh, torch.Tensor):
        V = torch.transpose(vh, 0, 1).to(device)
        V = V[:, :r]
        icol = []
        for i in range(0, r):
            col_i = torch.where(torch.abs(V) == (torch.max(torch.abs(V[:, i]))))[0]
            V[:, i+1:] = V[:, i+1:] - (V[:, 0:i+1] @ (torch.pinverse(V[col_i, 0:i+1]) @ V[col_i, i+1:]))
            icol.append(col_i.cpu().numpy())
    else:  # Assume numpy array
        V = np.transpose(vh)
        V = V[:, :r]
        icol = []
        for i in range(0, r):
            col_i = np.where(np.abs(V) == (np.max(np.abs(V[:, i]))))[0]
            V[:, i+1:] = V[:, i+1:] - (V[:, 0:i+1] @ (np.linalg.pinv(V[col_i, 0:i+1]) @ V[col_i, i+1:]))
            icol.append(col_i)
    return icol



def feature_sel(trainloader, batch_size, device, decomp_type="numpy"):
    '''
    Performs a SVD Decomposition of each batch using either torch or numpy.
    trainloader - the input trainloader on which training will be performed
    batch_size - batch_size used for training(Note: training batch size and decomposition batch_size should be same)
    device - "cuda" can only be used with torch (if the device is "cuda" and decomp_type is numpy "cpu" will be used by default.)
    decomp_type - perform decomposition using "torch" or "numpy")
    '''
    V_list = []
    for _, (trainsamples, _) in enumerate(tqdm(trainloader)):
        
        if decomp_type == "torch":
            _, _, Vt = torch.linalg.svd(torch.reshape(trainsamples.to(device),(-1, trainsamples.shape[0])),full_matrices=False)
        else:
            _, _, Vt = np.linalg.svd(np.reshape(trainsamples.cpu().numpy(),(-1,trainsamples.shape[0])),full_matrices=False)
        
        V_list.append(Vt)
        
    return V_list
        