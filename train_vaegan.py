import numpy as np
import argparse
import torch
from torch.optim import RMSprop
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
                    help='input training batch-size')

# Optimizer
parser.add_argument('--epochs', type=int, default=12, metavar='N',
                    help='Number of epochs (default: 12)')

# Noise dimension Generator
parser.add_argument('--latent-size', type=int, default=10, metavar='N',
                    help='Latent size (default: 128)')


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

def train_validate():
	pass

def execute_graph():
	pass


# Model definitions



# Init



# Optimizer



# Main epoch loop