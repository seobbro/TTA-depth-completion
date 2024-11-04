'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import numpy as np
import torch

eps = 1e-9
def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / (tgt+eps)) - (1.0 / (src+eps))) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / (tgt+eps)) - (1.0 / (src+eps))))

def abs_rel_err(src, tgt):
    '''
    Absolute relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : absolute relative error
    '''

    return np.mean(np.abs(src - tgt) / tgt)

def sq_rel_err(src, tgt):
    '''
    Squared relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : squared relative error
    '''

    return np.mean(((src - tgt) ** 2) / (tgt * tgt))



def torch_root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return torch.sqrt(torch.mean((tgt - src) ** 2))

def torch_mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return torch.mean(torch.abs(tgt - src))

def torch_inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return torch.sqrt(torch.mean(((1.0 / (tgt+eps)) - (1.0 / (src+eps))) ** 2))

def torch_inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return torch.mean(torch.abs((1.0 / (tgt+eps)) - (1.0 / (src+eps))))

def torch_abs_rel_err(src, tgt):
    '''
    Absolute relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : absolute relative error
    '''

    return torch.mean(torch.abs(src - tgt) / tgt)

def torch_sq_rel_err(src, tgt):
    '''
    Squared relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : squared relative error
    '''

    return torch.mean(((src - tgt) ** 2) / (tgt * tgt))





def torch_root_mean_sq_err_each(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.sqrt(torch.mean((tgt - src) ** 2, dim=1, keepdim=True))

def torch_mean_abs_err_each(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.mean(torch.abs(tgt - src), dim=1, keepdim=True)

def torch_inv_root_mean_sq_err_each(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.sqrt(torch.mean(((1.0 / (tgt+eps)) - (1.0 / (src+eps))) ** 2, dim=1, keepdim=True))

def torch_inv_mean_abs_err_each(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.mean(torch.abs((1.0 / (tgt+eps)) - (1.0 / (src+eps))), dim=1, keepdim=True)

def torch_abs_rel_err_each(src, tgt):
    '''
    Absolute relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : absolute relative error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.mean(torch.abs(src - tgt) / tgt, dim=1, keepdim=True)

def torch_sq_rel_err_each(src, tgt):
    '''
    Squared relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : squared relative error
    '''
    batch_size = src.size(0)
    src = src.view(batch_size, -1)
    tgt = tgt.view(batch_size, -1)
    return torch.mean(((src - tgt) ** 2) / (tgt * tgt), dim=1, keepdim=True)
