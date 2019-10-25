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


class ConvBatchLeaky(nn.Conv2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class ConvTrBatchLeaky(nn.ConvTranspose2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvTrBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(1)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvTrBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


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


class VAEGAN_Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAEGAN_Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.basenet = nn.Sequential([
            ConvBatchLeaky(self.in_channels, 64, kernel_size=5, padding=2, stride=2),
            ConvBatchLeaky(64, 64 * 2, kernel_size=5, padding=2, stride=2),
            ConvBatchLeaky(64 * 2, 64 * 4, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(),
            nn.ReLU()
        ])
        self.mu_encoder = nn.Linear(1024, self.latent_dim)
        self.logvar_encoder = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        x = self.basenet(x)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar


class VAEGAN_Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, batch_size):
        super(VAEGAN_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size

        self.network = nn.Sequential([
            nn.Linear(self.latent_dim, 8 * 8 * self.in_channels),
            nn.BatchNorm(),
            nn.ReLU(),
            BatchReshape((self.batch_size, -1, 8, 8)),
            ConvTrBatchLeaky(self.out_channels, self.out_channels, kernel_size=5, padding=2, stride=2),
            ConvTrBatchLeaky(self.out_channels, self.out_channels // 2, kernel_size=5, padding=2, stride=2),
            ConvTrBatchLeaky(self.out_channels // 2, self.out_channels // 4, kernel_size=5, padding=2, stride=2),
            nn.ConvTranspose2d(self.out_channels // 4, self.in_channels, kernel_size=5, padding=2, stride=1),
            nn.Tanh()
        ])

    def forward(self, x):
        return self.network(x)


class VAEGAN_Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(VAEGAN_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.network = nn.Sequential([
            nn.Conv2d(self.in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            ConvBatchLeaky(32, 128, kernel_size=5, stride=1, padding=2),
            ConvBatchLeaky(128, 256, kernel_size=5, stride=1, padding=2),
            ConvBatchLeaky(256, 256, kernel_size=5, stride=1, padding=2),
            nn.Linear(8 * 8 * 256, 512),
            nn.BatchNorm(),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        return self.network(x)
