import numpy as np
import argparse
import torch


from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='WGAN')


# Task parametersm and model name
parser.add_argument('--uid', type=str, default='WGAN',
                    help='Staging identifier (default: WGAN)')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='FashionMNIST',
                    help='Name of dataset (default: FashionMNIST')

parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')

# Optimizer
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')

parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: logs)')

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')


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
transforms = None
# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(G, D, G_optim, D_optim, loader, epoch, is_train):

    data_loader = loader.train_loader if is_train else loader.test_loader

    G.train() if is_train else G.eval()
    D.train() if is_train else D.eval()

    # losses

    for batch_idx, (x, _) in enumerate(data_loader):

        x = x.cuda() if args.cuda else x
        batch_size = x.size(0)

        if train:

        # forward passes




        return


def execute_graph(G, D, G_optim, D_optim, loader, epoch, use_tb):

    # Training loss
    G_t_loss, D_t_loss = train_validate(G, D, G_optim, D_optim, loader, epoch, is_train=True)

    # Validation loss
    G_v_loss, D_v_loss = train_validate(G, D, G_optim, D_optim, loader, epoch, is_train=False)

    print('=> Epoch: {} Average Train G loss: {:.4f}, D loss: {:.4f}'.format(epoch, G_t_loss, D_t_loss))
    print('=> Epoch: {} Average Valid G loss: {:.4f}, D loss: {:.4f}'.format(epoch, G_v_loss, D_v_loss))

    if use_tb:
        logger.add_scalar(log_dir + '/G-train-loss', G_t_loss, epoch)
        logger.add_scalar(log_dir + '/D-train-loss', D_t_loss, epoch)

        logger.add_scalar(log_dir + '/G-valid-loss', G_v_loss, epoch)
        logger.add_scalar(log_dir + '/D-valid-loss', D_v_loss, epoch)

    # Generate examples


# Model definitions
D = DCGAN_Discriminator(1).type(dtype)
G = DCGAN_Generator(1024).type(dtype)

# init model weights (TODO)
init_normal_weights(E, 0, 0.02)
init_normal_weights(G, 0, 0.02)
init_normal_weights(D, 0, 0.02)

# TODO
G_optim = Adam(G.parameters(), lr=1e-3)
D_optim = Adam(D.parameters(), lr=1e-3)


# Utils
num_epochs = args.epochs
best_loss = np.inf

# Main training loop
for epoch in range(1, num_epochs + 1):
    _, _, _ = execute_graph(G, D, G_optim, D_optim, loader, epoch, use_tb)

