import numpy as np
import argparse
import torch
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from tensorboardX import SummaryWriter
import torchvision.utils as tvu

from models import *
from utils import *
from data import *


parser = argparse.ArgumentParser(description='VAEGAN')


# Task parametersm and model name
parser.add_argument('--uid', type=str, default='VAEGAN',
                    help='Staging identifier (default: VAEGAN)')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='MNIST',
                    help='Name of dataset (default: MNIST')

parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input training batch-size')

# Optimizer
parser.add_argument('--epochs', type=int, default=12, metavar='N',
                    help='Number of epochs (default: 12)')

# Noise dimension Generator
parser.add_argument('--latent-size', type=int, default=128, metavar='N',
                    help='Latent size (default: 128)')

parser.add_argument("--lambda_mse", default=1e-6,
                    action="store", type=float, help='MSE weight (default: 1e6')

parser.add_argument("--decay-mse", default=1,
                    action="store", type=float, help='MSE decay (default: 1')

parser.add_argument("--margin", default=0.35,
                    action="store", type=float, help='Margin (default: 0.35')

parser.add_argument("--equilibrium", default=0.68,
                    action="store", type=float, help='Equilibrium (default: 0.68')

parser.add_argument("--lr", default=3e-4,
                    action="store", type=float, help='Learning rate (default: 3e-4')


parser.add_argument("--decay-lr", default=0.75,
                    action="store", type=float, help='Learning rate decay (default: 0.75')


parser.add_argument("--decay-margin", default=1,
                    action="store", type=float, help='Decay margin (default: 1')

parser.add_argument("--decay-equilibrium", default=1,
                    action="store", type=float, help='Decay equilibrium (default: 1')


parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: logs)')

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

args = parser.parse_args()


# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# Enable CUDA, set tensor type and device
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Data set transforms
transforms = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

# Get train and test loaders for dataset
loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
train_loader = loader.train_loader
test_loader = loader.test_loader


def train_validate(vaegan, Enc_optim, Dec_optim, Disc_optim, margin, equilibrium, lambda_mse, loader, epoch, train):

    model.train() if train else model.eval()
    data_loader = loader.train_loader if train else loader.test_loader

    fn_loss_mse = nn.MSELoss(reduction='sum')

    batch_encoder_loss = 0
    batch_decoder_loss = 0
    batch_discriminator_loss = 0

    for batch_idx, x, _ in enumerate(data_loader):
        batch_size = x.size(0)

        x = to_cuda(x) if args.cuda else x

        # base forward pass, no training
        mu, log_var, x_hat, x_draw_hat, x_features, x_hat_features, y_x, y_x_hat, y_draw_hat = vaegan(x)

        # negative loglikelihood loss
        recon_loss = fn_loss_mse(x.view(batch_size, -1), x_hat.view(batch_size, -1))

        # kl div against standard normal
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # MSE loss between intermediate layers
        feature_loss = fn_loss_mse(x_features.view(batch_size, -1), x_hat_features.view(batch_size, -1))

        # bce over the labels for the discriminator/gan
        bce_disc_y_x = torch.sum(-torch.log(y_x + 1e-3))
        bce_disc_y_x_hat = torch.sum(-torch.log(1 - y_x_hat + 1e-3))
        bce_disc_y_draw_hat = torch.sum(-torch.log(1 - y_draw_hat + 1e-3))
        bce_disc_total = torch.sum(bce_disc_y_x + bce_disc_y_x_hat + bce_disc_y_draw_hat)

        # Aggregate losses
        encoder_loss = kld_loss + feature_loss
        decoder_loss = lambda_mse * feature_loss - (1 - lambda_mse) * (bce_disc_total)
        discriminator_loss = bce_disc_total

        # Reporting
        batch_encoder_loss += torch.mean(encoder_loss).item() / batch_size
        batch_decoder_loss += torch.mean(decoder_loss).item() / batch_size
        batch_discriminator_loss += torch.mean(discriminator_loss).item() / batch_size

        # Encoder back
        if train:
            # Encoder is always trained
            Enc_optim.zero_grad()
            encoder_loss.backward()
            Enc_optim.step()

            # REFERENCE:
            # Selectively train decoder and discriminator
            if torch.mean(-torch.log(y_x + 1e-3)).item() < equilibrium - margin or \
                    torch.mean(-torch.log(1 - y_draw_hat + 1e-3)).item() < equilibrium - margin:
                train_disc = False
            else:
                train_disc = True

            if torch.mean(-torch.log(y_x + 1e-3)).item() > equilibrium + margin or \
                    torch.mean(-torch.log(1 - y_draw_hat + 1e-3)).item() > equilibrium + margin:
                train_dec = False
            else:
                train_dec = True

            if train_disc is False and train_dec is False:
                train_disc = True
                train_dec = True

            if train_dec:
                Dec_optim.zero_grad()
                decoder_loss.backward()
                Dec_optim.step()

            if train_disc:
                Disc_optim.zero_grad()
                discriminator_loss.backward()
                Disc_optim.step()

    # all done
    batch_encoder_loss /= (batch_idx + 1)
    batch_decoder_loss /= (batch_idx + 1)
    batch_discriminator_loss /= (batch_idx + 1)

    return batch_encoder_loss, batch_decoder_loss, batch_discriminator_loss


def execute_graph(vaegan, Enc_optim, Dec_optim, Disc_optim, enc_schedular,
                  dec_schedular, disc_schedular, margin, equilibrium, lambda_mse, loader, epoch):

    t_loss_enc, t_loss_dec, t_loss_disc = train_validate(vaegan, Enc_optim, Dec_optim, Disc_optim, margin, equilibrium, lambda_mse, loader, epoch, True)

    v_loss_enc, v_loss_dec, v_loss_disc = train_validate(vaegan, Enc_optim, Dec_optim, Disc_optim, margin, equilibrium, lambda_mse, loader, epoch, False)

    # Step the schedulers
    lr_encoder.step()
    lr_decoder.step()
    lr_discriminator.step()

    # use_tb
    if use_tb:
        logger.add_scalar(log_dir + '/Enc-train-loss', t_loss_enc, epoch)
        logger.add_scalar(log_dir + '/Dec-train-loss', t_loss_dec, epoch)
        logger.add_scalar(log_dir + '/Disc-train-loss', t_loss_disc, epoch)

        logger.add_scalar(log_dir + '/Enc-valid-loss', v_loss_enc, epoch)
        logger.add_scalar(log_dir + '/Dec-valid-loss', v_loss_dec, epoch)
        logger.add_scalar(log_dir + '/Disc-valid-loss', v_loss_disc, epoch)

        # TODO: image examples

    return


# Model definitions
reconstruction_level = 3
in_channels = 1

vaegan = VAEGAN(in_channels, args.latent_size, reconstruction_level)


# Init
vaegan.apply(init_xavier_weights)

# Optimizers
Enc_optim = RMSprop(params=vaegan.encoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8,
                    weight_decay=0, momentum=0, centered=False)
Dec_optim = RMSprop(params=vaegan.decoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8,
                    weight_decay=0, momentum=0, centered=False)
Disc_optim = RMSprop(params=vaegan.discriminator.parameters(), lr=args.lr, alpha=0.9, eps=1e-8,
                     weight_decay=0, momentum=0, centered=False)

# Scheduling
enc_schedular = ExponentialLR(Enc_optim, gamma=args.decay_lr)
dec_schedular = ExponentialLR(Dec_optim, gamma=args.decay_lr)
disc_schedular = ExponentialLR(Disc_optim, gamma=args.lr_decay_lr)

# Main epoch loop
margin = args.margin
equilibrium = args.equilibrium
lambda_mse = args.lambda_mse

for epoch in range(args.epochs):

    execute_graph(vaegan, Enc_optim, Dec_optim, Disc_optim, enc_schedular,
                  dec_schedular, disc_schedular, margin, equilibrium, lambda_mse, loader, epoch)
    # REFERENCE FROM here
    margin *= decay_margin
    equilibrium *= decay_equilibrium
    if margin > equilibrium:
        equilibrium = margin
    lambda_mse *= decay_mse
    if lambda_mse > 1:
        lambda_mse = 1
