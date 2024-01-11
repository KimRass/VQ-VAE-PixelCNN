# References:
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    # https://github.com/praeclarumjj3/VQ-VAE-on-MNIST/blob/master/modules.py

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, channels, out_channels, stride=2, activ=True, transposed=False):
        super().__init__()

        self.activ = activ

        if transposed:
            self.conv = nn.ConvTranspose2d(
                channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1,
            )
        else:
            self.conv = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
            )
        self.norm = nn.BatchNorm2d(out_channels)
        if activ:
            self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activ:
            x = self.leaky_relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_block1 = ConvBlock(channels, channels, stride=1, transposed=False)
        self.conv_block2 = ConvBlock(channels, channels, stride=1, transposed=False)

    def forward(self, x):
        skip = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x + skip


class Encoder(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()

        self.conv_block1 = ConvBlock(channels, embed_dim, transposed=False)
        self.conv_block2 = ConvBlock(embed_dim, embed_dim, transposed=False)
        self.res_block1 = ResBlock(channels=embed_dim)
        self.res_block2 = ResBlock(channels=embed_dim)

    def forward(self, x):
        # "The model takes an input $x$, that is passed through an encoder producing output $z_{e}(x)$.
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x # "$z_{e}(x)$"


class Decoder(nn.Module):
    def __init__(self, channels, embed_dim):
        super().__init__()

        self.res_block1 = ResBlock(channels=embed_dim)
        self.res_block2 = ResBlock(channels=embed_dim)
        self.conv_block1 = ConvBlock(embed_dim, embed_dim, transposed=True)
        self.conv_block2 = ConvBlock(embed_dim, channels, activ=False,transposed=True)
        # self.conv = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # x = self.conv(x)
        x = self.tanh(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, embed_dim):
        """
        Args:
            n_embeds (int): "The size of the discrete latent space $K$".
            embed_dim (int): "The dimensionality of each latent embedding vector $e_{i}$".
        """
        super().__init__()

        # n_embeds = 4
        # embed_dim = 8
        self.embed_space = nn.Embedding(n_embeds, embed_dim) # "$e \in \mathbb{R}^{K \times D}$"

    def forward(self, x): # (B, `embed_dim`, H, W)
        # x = torch.randn(2, embed_dim, 16, 16)
        # x.shape, embed_space.weight.shape

        b, _, h, w = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        dist_square = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up using the shared embedding space $e$.
        min_dist_idx = torch.argmin(dist_square, dim=1)
        post_cat_dist = min_dist_idx.view(b, h, w)
        
        # "The input to the decoder is the corresponding embedding vector $e_{k}$."
        x = self.embed_space(post_cat_dist) # "$z_{q}(x)$"
        x = x.view(b, -1, h, w)
        return x


class VQVAE(nn.Module):
    def __init__(self, channels, n_embeds, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        self.enc = Encoder(channels=channels, embed_dim=embed_dim)
        self.vect_quant = VectorQuantizer(n_embeds=n_embeds, embed_dim=embed_dim)
        self.dec = Decoder(channels=channels, embed_dim=embed_dim)

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.vect_quant(x)
        x = self.decode(x)
        return x

    def reconstruct(self, ori_image):
        x = self.encode(ori_image)
        x = self.vect_quant(x)
        x = self.decode(x)
        return x

    def get_loss(self, ori_image, beta):
        x = self.encode(ori_image)
        # tot_vq_loss = self.vect_quant.get_loss(x, beta=beta)
        # return tot_vq_loss


        # "To make sure the encoder commits to an embedding and its output does not grow, we add a commitment loss."
        b, _, _, _ = x.shape

        quant = self.vect_quant(x)
        # "The VQ objective uses the L2 error to move the embedding vectors ei towards the encoder outputs $z_{e}(x)$."
        # "$\beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$"
        vq_loss = F.mse_loss(quant.detach(), x, reduction="mean")
        commit_loss = beta * F.mse_loss(quant, x.detach(), reduction="mean")

        recon_image = self.decode(quant)
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss

    # def sample(self, n_samples, device):
    #     z = torch.randn(size=(n_samples, self.embed_dim), device=device)
    #     quant = self.vect_quant(z)
    #     x = self.decode(quant)
    #     return x


if __name__ == "__main__":
    channels = 1
    img_size = 64
    embed_dim = 256
    # recon_weight = 0.1
    beta = 3
    device = torch.device("cpu")

    ori_image = torch.randn(4, channels, img_size, img_size).to(device)
    model = VQVAE(
        channels=channels, n_embeds=32, embed_dim=embed_dim,
    ).to(device)
    # encoded = model.encode(ori_image)
    # encoded.shape

    # out = model(ori_image)
    # out.shape

    loss = model.get_loss(ori_image, beta=beta)
