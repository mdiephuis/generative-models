import numpy as np
import argparse
import torch
from torch.optim import Adam
from tensorboardX import SummaryWriter
import torchvision.utils as tvu

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='DCGAN')


# Task parametersm and model name
parser.add_argument('--uid', type=str, default='DCGAN',
                    help='Staging identifier (default: DCGAN)')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='MNIST',
                    help='Name of dataset (default: MNIST')

parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')

parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of training epochs (default: 15)')

# Noise dimension Generator
parser.add_argument('--noise-dim', type=int, default=96, metavar='N',
                    help='Noise dimension (default: 96)')


parser.add_argument('--encoder-size', type=int, default=128, metavar='N',
                    help='VAE encoder size (default: 128')


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
transforms = None

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(G, D, G_optim, D_optim, loader, epoch, is_train):

    img_shape = loader.img_shape

    data_loader = loader.train_loader if is_train else loader.test_loader

    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    # losses
    G_batch_loss = 0
    D_batch_loss = 0

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x
        x = x.view(batch_size, img_shape[0], img_shape[1], img_shape[2])

        # Real data, discriminator forward
        y_real = D(2 * (x - 0.5))

        # Sample z from uniform
        z = sample_uniform_noise(batch_size, args.noise_dim).type(dtype)

        # Generator forward
        x_fake = G(z).detach()
        y_fake = D(x_fake.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        # Discriminator loss
        y_ones = torch.ones(batch_size, )
        y_zeros = torch.zeros(batch_size, )

        y_ones = y_ones.cuda() if args.cuda else y_ones
        y_zeros = y_zeros.cuda() if args.cuda else y_zeros

        d_loss = loss_bce(y_real, y_ones) + loss_bce(y_fake, y_zeros)
        D_batch_loss += d_loss.item() / batch_size

        if is_train:
            D_optim.zero_grad()
            d_loss.backward()
            D_optim.step()

        # Generator forward again
        z = sample_uniform_noise(batch_size, args.noise_dim).type(dtype)

        # Generator forward
        x_fake = G(z)
        y_fake = D(x_fake.view(batch_size, img_shape[0], img_shape[1], img_shape[2]))

        # Generator loss
        y_ones = torch.ones(batch_size, )
        y_ones = y_ones.cuda() if args.cuda else y_ones

        g_loss = loss_bce(y_fake, y_ones)
        G_batch_loss += g_loss.item() / batch_size

        if is_train:
            G_optim.zero_grad()
            g_loss.backward()
            G_optim.step()

    return G_batch_loss / (batch_idx + 1), D_batch_loss / (batch_idx + 1)


def execute_graph(G, D, G_optim, D_optim, loader, epoch, use_tb):

    # Training loss
    G_t_loss, D_t_loss = train_validate(G, D, G_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    G_v_loss, D_v_loss = train_validate(G, D, G_optim, D_optim, loader, epoch, is_train=False)

    print('=> epoch: {} Average Train G loss: {:.4f}, D loss: {:.4f}'.format(epoch, G_t_loss, D_t_loss))
    print('=> epoch: {} Average Valid G loss: {:.4f}, D loss: {:.4f}'.format(epoch, G_v_loss, D_v_loss))

    if use_tb:
        logger.add_scalar(log_dir + '/G-train-loss', G_t_loss, epoch)
        logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, epoch)

        logger.add_scalar(log_dir + '/G-valid-loss', G_v_loss, epoch)
        logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, epoch)

    # Generate examples
        img_shape = loader.img_shape
        sample = wgan_generation_example(G, args.noise_dim, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)

    return G_v_loss, D_v_loss


# MNIST/DCGAN Model definitions
D = DCGAN_Discriminator(1).type(dtype)
G = DCGAN_Generator(args.noise_dim).type(dtype)

G.apply(init_xavier_weights)
D.apply(init_xavier_weights)

learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.999
G_optim = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
D_optim = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))


# Main training loop
for epoch in range(1, args.epochs):
    _, _ = execute_graph(G, D, G_optim, D_optim, loader, epoch, use_tb)


# TensorboardX logger
logger.close()
