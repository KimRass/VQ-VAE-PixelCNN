# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    # https://github.com/praeclarumjj3/VQ-VAE-on-MNIST/blob/master/modules.py

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

    def forward(self, x): # (B, `embed_dim`, H, W)
        # n_embeds = 30
        # embed_dim = 128
        # embed_space = nn.Embedding(n_embeds, embed_dim)
        # x = torch.randn(2, embed_dim, 16, 16)

        b, _, h, w = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        sq_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up
        # using the shared embedding space $e$.
        post_cat_dist = torch.argmin(sq_dist, dim=1)
        # "The input to the decoder is the corresponding embedding vector $e_{k}$."
        x = torch.index_select(input=self.embed_space.weight, dim=0, index=post_cat_dist)
        x = rearrange(x, pattern="(b h w) c -> b c h w", b=b, h=h, w=w)
        return x
        # post_cat_dist = min_dist_idx.view(b, h, w)
        # x = self.embed_space(post_cat_dist) # "$z_{q}(x)$", (B, H, w, `embed_dim`)
        # x = x.permute(0, 3, 1, 2) # (B, `embed_dim`, H, W)


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
        x = self.encode(ori_image)
        x = self.vect_quant(x)
        x = self.decode(x)
        return x

    def get_loss(self, ori_image, beta):
        x = self.encode(ori_image)

        quant = self.vect_quant(x)
        # "The VQ objective uses the L2 error to move the embedding vectors $e_{i}$
        # towards the encoder outputs $z_{e}(x)$."
        # "$\beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$"
        vq_loss = F.mse_loss(quant.detach(), x, reduction="mean")
        # "To make sure the encoder commits to an embedding and its output does not grow,
        # we add a commitment loss."
        commit_loss = beta * F.mse_loss(quant, x.detach(), reduction="mean")

        recon_image = self.decode(quant)
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss
        # return recon_loss


if __name__ == "__main__":
    input_dim = 1
    img_size = 64
    embed_dim = 256
    # recon_weight = 0.1
    beta = 3
    device = torch.device("cpu")

    ori_image = torch.randn(4, input_dim, img_size, img_size).to(device)
    model = VQVAE(
        input_dim=input_dim, n_embeds=32, embed_dim=embed_dim,
    ).to(device)
    # encoded = model.encode(ori_image)
    # encoded.shape

    # out = model(ori_image)
    # out.shape

    loss = model.get_loss(ori_image, beta=beta)
