# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    # https://github.com/praeclarumjj3/VQ-VAE-on-MNIST/blob/master/modules.py

# "q(z = kjx) is deterministic, and by defining a simple uniform prior over z we obtain
# a KL divergence constant and equal to logK."

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.layers(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            ResBlock(embed_dim),
            ResBlock(embed_dim),
        )

    def forward(self, x):
        # "The model takes an input $x$, that is passed through an encoder producing output $z_{e}(x)$.
        return self.layers(x) # "$z_{e}(x)$"


class Decoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
            ResBlock(embed_dim),
            ResBlock(embed_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, embed_dim):
        """
        Args:
            n_embeds (int): "The size of the discrete latent space $K$".
            embed_dim (int): "The dimensionality of each latent embedding vector $e_{i}$".
        """
        super().__init__()

        self.embed_space = nn.Embedding(n_embeds, embed_dim) # "$e \in \mathbb{R}^{K \times D}$"
        self.embed_space.weight.data.uniform_(-1 / n_embeds, 1 / n_embeds)

    def forward(self, x): # (b, `embed_dim`, h, w)
        ori_shape = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        squared_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up
        # using the shared embedding space $e$.
        argmin = torch.argmin(squared_dist, dim=1)
        # "The input to the decoder is the corresponding embedding vector $e_{k}$."
        # x = torch.index_select(input=self.embed_space.weight, dim=0, index=argmin)
        x = self.embed_space(argmin) # "$z_{q}(x)$", (b, h, w, `embed_dim`)
        x = x.view(ori_shape) # (b, `embed_dim`, h, w)
        return x


class VQVAE(nn.Module):
    def __init__(self, input_dim, n_embeds, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.enc = Encoder(input_dim=input_dim, embed_dim=embed_dim)
        self.vect_quant = VectorQuantizer(n_embeds=n_embeds, embed_dim=embed_dim)
        self.dec = Decoder(input_dim=input_dim, embed_dim=embed_dim)

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
    input_dim = 1
    img_size = 64
    embed_dim = 256
    # recon_weight = 0.1
    device = torch.device("cpu")

    ori_image = torch.randn(4, input_dim, img_size, img_size).to(device)
    model = VQVAE(
        input_dim=input_dim, n_embeds=32, embed_dim=embed_dim,
    ).to(device)
    # encoded = model.encode(ori_image)
    # encoded.shape

    # out = model(ori_image)
    # out.shape

    loss = model.get_loss(ori_image, commit_weight=commit_weight)
