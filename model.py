# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    # https://github.com/praeclarumjj3/VQ-VAE-on-MNIST/blob/master/modules.py

# "q(z = kjx) is deterministic, and by defining a simple uniform prior over z we obtain
# a KL divergence constant and equal to logK."

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, *args, transposed, activation="relu"):
        super().__init__()

        self.activation = activation

        if transposed:
            self.conv = nn.ConvTranspose2d(*args, bias=False)
        else:
            self.conv = nn.Conv2d(*args, bias=False)
        self.norm = nn.BatchNorm2d(self.conv.out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # "Implemented as ReLU, 3x3 conv, ReLU, 1x1 conv"
        self.layers = nn.Sequential(
            # nn.ReLU(),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            # nn.BatchNorm2d(hidden_dim),
            # nn.ReLU(),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            # nn.BatchNorm2d(hidden_dim),
            ConvBlock(hidden_dim, hidden_dim, 3, 1, 1, transposed=False),
            ConvBlock(hidden_dim, hidden_dim, 1, 1, 0, transposed=False),
        )

    def forward(self, x):
        return x + self.layers(x)


class Encoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        # "The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4,
        # followed by two residual 3 × 3 blocks.
        # "We use a field of 32 × 32 latents for ImageNet, or 8 × 8 × 10 for CIFAR10.
        self.layers = nn.Sequential(
            # nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(hidden_dim),
            # nn.ReLU(),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            # ResBlock(hidden_dim),
            # ResBlock(hidden_dim),
            ConvBlock(channels, hidden_dim, 4, 2, 1, transposed=False),
            ConvBlock(hidden_dim, hidden_dim, 4, 2, 1, transposed=False),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

    def forward(self, x):
        # "The model takes an input $x$, that is passed through an encoder producing output $z_{e}(x)$.
        return self.layers(x) # "$z_{e}(x)$"


class Decoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        # The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions
        # with stride 2 and window size 4 × 4.
        self.layers = nn.Sequential(
            # ResBlock(hidden_dim),
            # ResBlock(hidden_dim),
            # nn.ReLU(),
            # nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(hidden_dim),
            # nn.ReLU(),
            # nn.ConvTranspose2d(hidden_dim, channels, kernel_size=4, stride=2, padding=1),
            # nn.Tanh(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ConvBlock(hidden_dim, hidden_dim, 4, 2, 1, transposed=True),
            ConvBlock(hidden_dim, channels, 4, 2, 1, transposed=True, activation="tanh"),
        )

    def forward(self, x):
        return self.layers(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, hidden_dim):
        super().__init__()

        self.embed_space = nn.Embedding(n_embeds, hidden_dim) # "$e \in \mathbb{R}^{K \times D}$"
        self.embed_space.weight.data.uniform_(-1 / n_embeds, 1 / n_embeds)

    def forward(self, x): # (b, `hidden_dim`, h, w)
        ori_shape = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        squared_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up
        # using the shared embedding space $e$.
        argmin = torch.argmin(squared_dist, dim=1)
        # "The input to the decoder is the corresponding embedding vector $e_{k}$."
        # x = torch.index_select(input=self.embed_space.weight, dim=0, index=argmin)
        x = self.embed_space(argmin) # "$z_{q}(x)$", (b, h, w, `hidden_dim`)
        x = x.view(ori_shape) # (b, `hidden_dim`, h, w)
        return x


class VQVAE(nn.Module):
    def __init__(self, channels, n_embeds, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim)
        self.vect_quant = VectorQuantizer(n_embeds=n_embeds, hidden_dim=hidden_dim)
        self.dec = Decoder(channels=channels, hidden_dim=hidden_dim)

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, ori_image):
        # "The model takes an input $x$, that is passed through an encoder producing output $z_{e}(x)$.
        z_e = self.encode(ori_image) # "$z_{e}(x)$"
        z_q = self.vect_quant(z_e)
        x = self.decode(z_q)
        return x

    def get_loss(self, ori_image, commit_weight=0.25):
        z_e = self.encode(ori_image) # "$z_{e}(x)$"
        z_q = self.vect_quant(z_e) # "$z_{q}(x)$"
        # "The VQ objective uses the $l_{2}$ error to move the embedding vectors $e_{i}$
        # towards the encoder outputs $z_{e}(x)$."
        # "$\Vert \text{sg}[z_{e}(x)] - e \Vert^{2}_{2}$"
        vq_loss = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        # "To make sure the encoder commits to an embedding and its output does not grow,
        # we add a commitment loss."
        # "$\beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$"
        commit_loss = commit_weight * F.mse_loss(z_e, z_q.detach(), reduction="mean")
        recon_image = self.decode(z_q)
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss


if __name__ == "__main__":
    img_size = 32
    channels = 3
    n_embeds = 512
    hidden_dim = 256

    ori_image = torch.randn(4, channels, img_size, img_size)
    model = VQVAE(
        channels=channels, n_embeds=32, hidden_dim=hidden_dim,
    )
    pred = model(ori_image)
    pred.shape
