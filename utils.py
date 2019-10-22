import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def sample_uniform_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def init_normal_weights(module, mu, std):
    for m in module.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(mu, std)
            m.bias.data.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_normal_weights(sub_mod, mu, std)
