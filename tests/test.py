import torch.nn as nn
import torch

# Enable CUDA, set tensor type and device

torch.cuda.set_device(0)

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.output_network = nn.Sequential(
            nn.Linear(64 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.output_network(x.view(x.size(0), -1))


# default weight init
D1 = Discriminator()

# Fake some x and x_hat data
x = torch.randn(5, 1, 64, 64)
x_hat = torch.randn(5, 1, 64, 64)

##### CAT method

# discriminator intermediate feature layer output
x_batch = torch.cat((x, x_hat), 0)

# forward pass bogus discriminator, with all data in 1 batch
dout = D1(x_batch)

# fish out part beloning to x, and x_hat
h = dout.size(0) // 2
y_x = dout[:h]
y_draw_hat = dout[h:]

# bce over the labels for the discriminator/gan for 'true' and 'fake' labels
bce_disc_y_x = -torch.log(y_x + 1e-3)
bce_disc_y_draw_hat = -torch.log(1 - y_draw_hat + 1e-3)

discriminator_loss = torch.sum(bce_disc_y_x) + torch.sum(bce_disc_y_draw_hat)

print(discriminator_loss.item())


#### 2 pass. Note, there has been no backward

y_x = D1(x)
y_draw_hat = D1(x_hat)

# bce over the labels for the discriminator/gan for 'true' and 'fake' labels
bce_disc_y_x = -torch.log(y_x + 1e-3)
bce_disc_y_draw_hat = -torch.log(1 - y_draw_hat + 1e-3)


discriminator_loss = torch.sum(bce_disc_y_x) + torch.sum(bce_disc_y_draw_hat)

print(discriminator_loss.item())
