import torch
import torch.nn as nn
import torch.nn.init as init


# REF: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
def loss_bce(x_hat, x):
    loss = x_hat.clamp(min=0) - x_hat * x + torch.log(1 + torch.exp(-torch.abs(x_hat)))
    return torch.mean(loss)


def sample_uniform_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def sample_gauss_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).normal_(0, 1)


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


def init_xavier_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_xavier_weights(sub_mod)


def init_wgan_weights(m):
    # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def wgan_generation_example(G, noise_dim, n_samples, img_shape, use_cuda):

    z_real = sample_uniform_noise(n_samples, noise_dim)
    z_real = z_real.cuda() if use_cuda else z_real

    x_hat = G(z_real).cpu().view(n_samples, img_shape[0], img_shape[1], img_shape[2])

    # due to tanh output layer in the generator
    x_hat = x_hat * 0.5 + 0.5

    return x_hat


def vaegan_generation_example(vaegan, noise_dim, n_samples, img_shape, use_cuda):

    x_hat = vaegan.generate(n_samples)
    x_hat = x_hat.cpu().view(n_samples, img_shape[0], img_shape[1], img_shape[2])

    # due to tanh output layer in the generator
    x_hat = x_hat * 0.5 + 0.5

    return x_hat
