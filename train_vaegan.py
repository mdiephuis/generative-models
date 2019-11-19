import argparse
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR

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
                    help='Input training batch-size (default: 64)')

# Optimizer
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='Number of epochs (default: 100)')

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
                    help='logging directory (default: runs)')

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

torch.cuda.set_device(0)

if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# ugly
if args.dataset_name == 'CelebA':

    reconstruction_level = 2
    in_channels = 3

    loader = CelebALoader(args.data_dir, args.batch_size, 0.2, True, True, args.cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader
else:
    reconstruction_level = 1
    in_channels = 1

    # Data set transforms
    transforms = [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

    # DATASET
    loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, transforms, None, args.cuda)
    train_loader = loader.train_loader
    test_loader = loader.test_loader


def train_validate(vaegan, Enc_optim, Dec_optim, Disc_optim, margin, equilibrium, lambda_mse, loader, epoch, train):

    vaegan.train() if train else vaegan.eval()
    data_loader = loader.train_loader if train else loader.test_loader

    batch_encoder_loss = 0
    batch_decoder_loss = 0
    batch_discriminator_loss = 0

    for batch_idx, (x, _) in enumerate(data_loader):

        batch_size = x.size(0)

        x = x.cuda() if args.cuda else x

        # base forward pass, no training
        mu, log_var, x_hat, x_draw_hat, x_features, x_hat_features, y_x, y_draw_hat = vaegan(x)

        # kl div against standard normal
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # MSE loss between intermediate layers
        feature_loss = torch.sum(0.5 * (x_features - x_hat_features) ** 2, 1)

        # bce over the labels for the discriminator/gan
        bce_disc_y_x = -torch.log(y_x + 1e-3)

        # bce_disc_y_x_hat = torch.sum(-torch.log(1 - y_x_hat + 1e-3))
        bce_disc_y_draw_hat = -torch.log(1 - y_draw_hat + 1e-3)

        # Aggregate losses
        encoder_loss = torch.sum(kld_loss) + torch.sum(feature_loss)
        discriminator_loss = torch.sum(bce_disc_y_x) + torch.sum(bce_disc_y_draw_hat)
        decoder_loss = torch.sum(lambda_mse * feature_loss) - (1.0 - lambda_mse) * discriminator_loss

        # Reporting
        batch_encoder_loss += torch.mean(encoder_loss).item() / batch_size
        batch_decoder_loss += torch.mean(decoder_loss).item() / batch_size
        batch_discriminator_loss += torch.mean(discriminator_loss).item() / batch_size

        # Encoder back
        if train:
            # Encoder is always trained
            vaegan.zero_grad()
            # Enc_optim.zero_grad()
            encoder_loss.backward(retain_graph=True)
            Enc_optim.step()

            vaegan.zero_grad()

            # Selectively train decoder and discriminator
            # REFERENCE: https://github.com/lucabergamini/VAEGAN-PYTORCH
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
                decoder_loss.backward(retain_graph=True)
                Dec_optim.step()
                vaegan.discriminator.zero_grad()

            if train_disc:
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

    print('=> Epoch: {} Train loss encoder: {:.4f} - decoder: {:.4f} - discriminator: {:.4f}'.format(
          epoch, t_loss_enc, t_loss_dec, t_loss_disc))
    print('=> Epoch: {} Validation loss: encoder: {:.4f} - decoder: {:.4f} - discriminator: {:.4f}'.format(
          epoch, v_loss_enc, v_loss_dec, v_loss_disc))

    # Step the schedulars
    enc_schedular.step()
    dec_schedular.step()
    disc_schedular.step()

    if use_tb:
        logger.add_scalar(log_dir + '/Encoder-train-loss', t_loss_enc, epoch)
        logger.add_scalar(log_dir + '/Decoder-train-loss', t_loss_dec, epoch)
        logger.add_scalar(log_dir + '/Discriminator-train-loss', t_loss_disc, epoch)

        # logger.add_scalar(log_dir + '/Encoder-valid-loss', v_loss_enc, epoch)
        # logger.add_scalar(log_dir + '/Decoder-valid-loss', v_loss_dec, epoch)
        # logger.add_scalar(log_dir + '/Discriminator-valid-loss', v_loss_disc, epoch)

        # Generate images
        img_shape = loader.img_shape
        sample = vaegan_generation_example(vaegan, args.latent_size, 10, img_shape, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=True, scale_each=True)
        logger.add_image('generation example', sample, epoch)

    return


# Model definitions
vaegan = VAEGAN(in_channels, args.latent_size, reconstruction_level).type(dtype)

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
disc_schedular = ExponentialLR(Disc_optim, gamma=args.decay_lr)

# Main epoch loop
margin = args.margin
equilibrium = args.equilibrium
lambda_mse = args.lambda_mse

for epoch in range(args.epochs):

    execute_graph(vaegan, Enc_optim, Dec_optim, Disc_optim, enc_schedular,
                  dec_schedular, disc_schedular, margin, equilibrium, lambda_mse, loader, epoch)

    # Decay
    # REFERENCE: https://github.com/lucabergamini/VAEGAN-PYTORCH
    margin *= args.decay_margin
    equilibrium *= args.decay_equilibrium

    if margin > equilibrium:
        equilibrium = margin
    lambda_mse *= args.decay_mse
    if lambda_mse > 1:
        lambda_mse = 1

# TensorboardX logger
logger.close()
