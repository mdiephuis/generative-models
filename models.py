import torch.nn as nn
import torch


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


class ConvBatchRelu(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ConvBatchRelu, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.ReLU()

    def forward(self, x):
        x = super(ConvBatchRelu, self).forward(x)
        return self.lr(self.bn(x))


class VEAGAN_ConvBlock(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(VEAGAN_ConvBlock, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.ReLU()

    def forward(self, x, raw_output=False):
        conv_out = super(VEAGAN_ConvBlock, self).forward(x)
        x = self.lr(self.bn(conv_out))
        if raw_output is True:
            return x, conv_out
        else:
            return conv_out


class ConvTrBatchLeaky(nn.ConvTranspose2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvTrBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(1)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvTrBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class ConvTrBatchRelu(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(ConvTrBatchRelu, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(1)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.ReLU()

    def forward(self, x):
        x = super(ConvTrBatchRelu, self).forward(x)
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

        self.basenet = nn.Sequential(
            ConvBatchRelu(self.in_channels, 64, 5, 2, 2),
            ConvBatchRelu(64, 64 * 2, 5, 2, 2),
            ConvBatchRelu(64 * 2, 64 * 4, 5, 2, 2),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * 256, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                nn.ReLU(True))
        self.mu_encoder = nn.Linear(1024, self.latent_dim)
        self.logvar_encoder = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        x = self.basenet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar


class VAEGAN_Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels):
        super(VAEGAN_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = 64 * 4

        self.input_network = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * 8 * self.size),
            nn.BatchNorm1d(8 * 8 * self.size),
            nn.ReLU(),
        )

        self.decode_network = nn.Sequential(
            ConvTrBatchRelu(self.size, self.size, 5, 2, 2, 1),
            ConvTrBatchRelu(self.size, self.size // 2, 5, 2, 2, 1),
            ConvTrBatchRelu(self.size // 2, self.size // 8, 5, 2, 2, 1),
            nn.ConvTranspose2d(self.size // 8, self.in_channels, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_network(x)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.decode_network(x)
        return x


class VAEGAN_Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(VAEGAN_Discriminator, self).__init__()
        self.in_channels = in_channels

        self.input_network = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 5, 1, 2),
            nn.ReLU()
        )

        self.feature_network = nn.ModuleList([
            VEAGAN_ConvBlock(32, 128, 5, 2, 2),
            VEAGAN_ConvBlock(128, 256, 5, 2, 2),
            VEAGAN_ConvBlock(256, 256, 5, 2, 2),
        ])

        self.output_network = nn.Sequential(
            nn.Linear(8 * 8 * 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, reconstruction_level=3, mode='reconstruction'):

        # Input encoding network
        x = self.input_network(x)

        # Return intermediate convolution layer features for passed reconstruction_level
        if mode == 'reconstruction':
            for level, veagan_layer in enumerate(self.feature_network):
                if level == reconstruction_level:
                    x, conv_out = veagan_layer(x, True)
                    # needs a view()
                    return conv_out.view(x.size(0), -1)
                else:
                    x = veagan_layer(x, False)

        # Traditional full forward
        else:
            # vaegan layers
            for veagan_layer in self.feature_network:
                x = veagan_layer(x, False)

            # output encoding
            x = x.view(x.size(0), -1)
            x = self.output_network(x)
            return x


class VAEGAN(nn.Module):
    def __init__(self, in_channels, latent_dim=128, reconstruction_level=3):
        super(VAEGAN, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.reconstruction_level = reconstruction_level

        self.encoder = VAEGAN_Encoder(self.in_channels, self.latent_dim)
        self.decoder = VAEGAN_Decoder(self.latent_dim, self.in_channels, self.in_channels)
        self.discriminator = VAEGAN_Discriminator(self.in_channels)

    def reparametrize(self, mu, log_var):
        variance = torch.exp(0.5 * log_var)
        eps = torch.randn_like(variance)
        z = eps.mul(variance).add_(mu)
        return z

    def forward(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)

        # discriminator intermediate feature layer output
        x_batch = torch.cat((x_hat, x), 0)

        dout_x_batch = self.discriminator(x_batch, self.reconstruction_level, 'reconstruction')

        h = dout_x_batch.size(0) // 2
        x_hat_features = dout_x_batch[:h]
        x_features = dout_x_batch[h:]

        # Draw an x from the z distribution
        x_draw = torch.randn(x.size(0), self.latent_dim)
        x_draw = x_draw.cuda() if x.is_cuda else x_draw

        # Decode
        x_draw_hat = self.decoder(x_draw)

        # Discriminator class estimation
        x_batch = torch.cat((x, x_draw_hat), 0)
        dout_x_batch = self.discriminator(x_batch, mode='classifier')
        h = dout_x_batch.size(0) // 2

        y_x = dout_x_batch[:h]
        y_draw_hat = dout_x_batch[h:]

        return mu, log_var, x_hat, x_draw_hat, x_features, x_hat_features, y_x, y_draw_hat

    def reconstruct(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat

    def generate(self, num_samples):
        x_draw = torch.randn(num_samples, self.latent_dim)
        if next(self.decoder.parameters()).is_cuda:
            x_draw = x_draw.cuda()

        x_draw_hat = self.decoder(x_draw)
        return x_draw_hat
