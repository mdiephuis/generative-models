import torch.nn as nn


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(DCGAN_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.network = nn.ModuleList([
            nn.Conv2d(self.in_channels, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            BatchFlatten(),
            nn.Linear(64 * 16, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 1)
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class DCGAN_Generator(nn.Module):
    def __init__(self, noise_dim):
        super(DCGAN_Generator, self).__init__()
        self.noise_dim = noise_dim

        self.network = nn.ModuleList([
            nn.Linear(self.noise_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(7 * 7 * 128),
            BatchReshape((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            BatchFlatten()
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class MNIST_Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(MNIST_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x


class MNIST_Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MNIST_Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.network = nn.ModuleList([
            BatchFlatten(),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        ])

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x
