import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def sample_uniform_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)
