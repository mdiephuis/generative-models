import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from scipy.stats import norm


def pca_project(x, num_elem=2):

    if isinstance(x, torch.Tensor) and len(x.size()) == 3:
        batch_proj = []
        for batch_ind in range(x.size(0)):
            tensor_proj = pca_project(x[batch_ind].squeeze(0), num_elem)
            batch_proj.append(tensor_proj)
        return torch.cat(batch_proj)

    xm = x - torch.mean(x, 1, keepdim=True)
    xx = torch.matmul(xm, torch.transpose(xm, 0, -1))
    u, s, _ = torch.svd(xx)
    x_proj = torch.matmul(u[:, 0:num_elem], torch.diag(s[0:num_elem]))
    return x_proj


# REF: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
def loss_bce(x_hat, x):
    loss = x_hat.clamp(min=0) - x_hat * x + torch.log(1 + torch.exp(-torch.abs(x_hat)))
    return torch.mean(loss)


def sample_uniform_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).uniform_(-1, 1)


def sample_gauss_noise(batch_size, dim):
    return torch.Tensor(batch_size, dim).normal_(0, 1)


def one_hot(labels, n_class):

    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)

    mask = torch.DoubleTensor(labels.size(0), n_class).fill_(0)

    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)


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


def latentcluster2d_example(E, model_type, data_loader, use_pca, use_cuda):
    E.eval()
    img_shape = data_loader.img_shape[1:]

    data = []
    labels = []
    for _, (x, y) in enumerate(data_loader.test_loader):
        x = x.cuda() if use_cuda else x
        if model_type != 'conv':
            x = x.view(-1, img_shape[0] * img_shape[1])

        z = E(x)
        data.append(z.detach().cpu())
        y = y.detach().cpu().numpy()
        labels.extend(y.flatten())

    centroids = torch.cat(data)
    centroids = centroids.reshape(-1, z.size(1))

    if centroids.size(1) > 2 and use_pca:
        centroids = pca_project(centroids, 2)
    elif centroids.size(1) > 2:
        centroids = centroids[:, :2]

    return centroids.numpy(), labels


def aae_generation_example(G, model_type, latent_size, n_samples, img_shape, use_cuda):

    z_real = sample_gauss_noise(n_samples, latent_size).view(-1, latent_size, 1, 1)
    z_real = z_real.cuda() if use_cuda else z_real

    if model_type != 'conv':
        z_real = z_real.view(-1, latent_size)

    x_hat = G(z_real).cpu().view(n_samples, 1, img_shape[0], img_shape[1])

    return x_hat


def aae_reconstruct(E, G, model_type, test_loader, n_samples, img_shape, use_cuda):
    E.eval()
    G.eval()

    x, _ = next(iter(test_loader))
    x = x.cuda() if use_cuda else x

    if model_type != 'conv':
        x = x.view(-1, img_shape[0] * img_shape[1])

    z_val = E(x)

    x_hat = G(z_val)

    x = x[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    x_hat = x_hat[:n_samples].cpu().view(10 * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(10 * img_shape[0], 2 * img_shape[1])
    return comparison


def aae_manifold_generation_example(G, model_type, img_shape, use_cuda):
    # This nifty little grid trick is from:
    # https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py

    z_range = 1
    nx, ny = 15, 15
    # gives (15, 15, 2) or (_, _) z pair, per image we wish to generate
    zgrid = np.rollaxis(np.mgrid[z_range:-z_range:ny * 1j, -z_range:z_range:nx * 1j], 0, 3)
    zgrid = zgrid.reshape([-1, 2])

    DELTA = 1E-10
    zgrid_normal = np.array([norm.ppf(np.clip(z_i, DELTA, 0.9999999)) for z_i in zgrid])

    zgrid_normal = torch.from_numpy(zgrid_normal).type(torch.FloatTensor)
    zgrid_normal = zgrid_normal.cuda() if use_cuda else zgrid_normal

    # Generator forward
    manifold = G(zgrid_normal)
    manifold = manifold.cpu().view(nx * img_shape[0], ny * img_shape[1])
    return manifold
