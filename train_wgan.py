import numpy as np
import argparse
import torch
from torch.optim import RMSprop
from tensorboardX import SummaryWriter
import torchvision.utils as tvu

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='WGAN')


# Task parametersm and model name
parser.add_argument('--uid', type=str, default='WGAN',
                    help='Staging identifier (default: WGAN)')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='MNIST',
                    help='Name of dataset (default: MNIST')

parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')

# Optimizer
parser.add_argument('--niter', type=int, default=5000, metavar='N',
                    help='number of training iterations (default: 5000)')

# Noise dimension Generator
parser.add_argument('--noise-dim', type=int, default=10, metavar='N',
                    help='Noise dimension (default: 10)')

# Clipping value
parser.add_argument('--clip', type=int, default=0.01, metavar='N',
                    help='Gradient clipping value (default: 0.01)')

parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()


# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()


# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Data set transforms
transforms = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(G, D, G_optim, D_optim, loader, curr_iter, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    # losses
    G_batch_loss = 0
    D_batch_loss = 0

    # 1) While not converged, taken from:
    # https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
    if curr_iter < 25 or curr_iter % 500 == 0:
        D_iter = 100
    else:
        D_iter = 5

    for _ in range(D_iter):

        x, _ = next(iter(data_loader))
        x = x.cuda() if args.cuda else x
        x = x.view(x.size(0), img_shape[0], img_shape[1], img_shape[2])

        batch_size = x.size(0)

        # Optimize over Discriminator weights, freeze Generator
        for p in G.parameters():
            p.requires_grad = False

        # Sample z from p(z)
        z = sample_gauss_noise(batch_size, args.noise_dim).type(dtype)

        # Generator forward
        x_hat = G(z)

        # Discriminator forward real data
        y_hat_fake = D(x_hat.view(x.size(0), img_shape[0], img_shape[1], img_shape[2]))
        y_hat_real = D(x)

        D_loss = - (torch.mean(y_hat_real) - torch.mean(y_hat_fake))

        D_batch_loss += D_loss.item() / batch_size

        if is_train:
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            for p in D.parameters():
                p.data.clamp_(-args.clip, args.clip)

    # 2) Optimize over Generator weights
    for p in G.parameters():
        p.requires_grad = True

    # Generator forward, Discriminator forward
    z = sample_gauss_noise(batch_size, args.noise_dim).type(dtype)
    x_hat = G(z)
    y_hat_fake = D(x_hat.view(x.size(0), img_shape[0], img_shape[1], img_shape[2]))

    # Generator loss backward
    G_loss = - torch.mean(y_hat_fake)
    G_batch_loss += G_loss.item() / batch_size

    if is_train:
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

    return G_batch_loss, D_batch_loss / 5


def execute_graph(G, D, G_optim, D_optim, loader, curr_iter, use_tb):

    # Training loss
    G_t_loss, D_t_loss = train_validate(G, D, G_optim, D_optim, loader, curr_iter, is_train=True)

    # Validation loss
    G_v_loss, D_v_loss = train_validate(G, D, G_optim, D_optim, loader, curr_iter, is_train=False)

    if curr_iter % 25 == 0:

        print('=> curr_iter: {} Average Train G loss: {:.4f}, D loss: {:.4f}'.format(curr_iter, G_t_loss, D_t_loss))
        print('=> curr_iter: {} Average Valid G loss: {:.4f}, D loss: {:.4f}'.format(curr_iter, G_v_loss, D_v_loss))

        # to do, only ever 100 iters

        if use_tb:
            logger.add_scalar(log_dir + '/G-train-loss', G_t_loss, curr_iter)
            logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, curr_iter)

            logger.add_scalar(log_dir + '/G-valid-loss', G_v_loss, curr_iter)
            logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, curr_iter)

        # Generate examples
            img_shape = loader.img_shape
            sample = wgan_generation_example(G, args.noise_dim, 10, img_shape, args.cuda)
            sample = sample.detach()
            sample = tvu.make_grid(sample, normalize=True, scale_each=True)
            logger.add_image('generation example', sample, curr_iter)

    return G_v_loss, D_v_loss


D = DCGAN_Discriminator(1).type(dtype)
G = DCGAN_Generator(args.noise_dim).type(dtype)

G.apply(init_wgan_weights)
D.apply(init_wgan_weights)

# TODO
G_optim = RMSprop(G.parameters(), lr=5e-5)
D_optim = RMSprop(D.parameters(), lr=5e-5)


# Utils
num_iter = args.niter
best_loss = np.inf

# Main training loop
for i in range(1, num_iter + 1):
    _, _ = execute_graph(G, D, G_optim, D_optim, loader, i, use_tb)


# TensorboardX logger
logger.close()
