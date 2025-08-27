
import torch
import numpy as np


def calnorm(idxgrads, fgrads):

    ss_grad = torch.transpose(idxgrads.clone().detach().cpu(), 0, 1)
    b_ = fgrads.sum(dim = 0).detach().cpu().numpy()


    pinverse = np.linalg.pinv(ss_grad)
    x = pinverse @ b_
     
    x = torch.FloatTensor(x)
    norm_residual = torch.norm(ss_grad @ x - torch.FloatTensor(b_))
    return norm_residual
