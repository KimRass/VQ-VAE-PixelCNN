# References:
    # https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
torch.autograd.set_detect_anomaly(True)


class Encoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        # "The encoder consists of 2 strided convolutional layers with stride 2 and window size 4 × 4,
        # followed by two residual 3 × 3 blocks.
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        # "We use a field of 32 × 32 latents for ImageNet, or 8 × 8 × 10 for CIFAR10.
        x = self.conv_block(x)
        x = x + self.res_block(x)
        x = self.bn(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeds, hidden_dim):
        super().__init__()

        self.embed_space = nn.Embedding(n_embeds, hidden_dim) # "$e \in \mathbb{R}^{K \times D}$"
        self.embed_space.weight.data.uniform_(-1 / n_embeds, 1 / n_embeds) # Uniform distribution??

    def forward(self, x): # (b, `hidden_dim`, h, w)
        b, _, h, w = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        squared_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up
        # using the shared embedding space $e$.
        argmin = torch.argmin(squared_dist, dim=1)
        # "The input to the decoder is the corresponding embedding vector $e_{k}$."
        x = self.embed_space(argmin) # (b, h, w, `hidden_dim`)
        x = rearrange(x, pattern="(b h w) c -> b c h w", b=b, h=h, w=w) # (b, `hidden_dim`, h, w)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()

        # The decoder similarly has two residual 3 × 3 blocks, followed by two transposed convolutions
        # with stride 2 and window size 4 × 4.
        self.res_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, channels, 4, 2, 1, bias=True),
            nn.BatchNorm2d(channels),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        x = self.conv_block(x)
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
        # This term trains the vector quantizer.
        vq_loss = F.mse_loss(z_e.detach(), z_q, reduction="mean")
        # "To make sure the encoder commits to an embedding and its output does not grow,
        # we add a commitment loss."
        # "$\beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$"
        # This term trains the encoder.
        commit_loss = commit_weight * F.mse_loss(z_e, z_q.detach(), reduction="mean")
        z_q = z_e + (z_q - z_e).detach() # Preserve gradient??

        recon_image = self.decode(z_q)
        # This term trains the decoder.
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss


if __name__ == "__main__":
    img_size = 32
    channels = 3
    n_embeds = 2
    hidden_dim = 3

    ori_image = torch.randn(4, channels, img_size, img_size)
    model = VQVAE(
        channels=channels, n_embeds=n_embeds, hidden_dim=hidden_dim,
    )

    x = torch.randn(1, hidden_dim, 4, 4)
    x = rearrange(x, pattern="b c h w -> (b h w) c")
    embed_space = nn.Embedding(n_embeds, hidden_dim)
    # embed_space.weight.data.uniform_(-1 / n_embeds, 1 / n_embeds)
    # embed_space.weight.shape
    # x.unsqueeze(1).shape, embed_space.weight.unsqueeze(0).shape, (x.unsqueeze(1) - embed_space.weight.unsqueeze(0)).shape
    # x.unsqueeze(1)[0, 0, :] - embed_space.weight.unsqueeze(0)[0, 0, :]
    # (x.unsqueeze(1) - embed_space.weight.unsqueeze(0))[0, 0, :]
    
    squared_dist = ((x.unsqueeze(1) - embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
    argmin = torch.argmin(squared_dist, dim=1)
    embed_space(argmin)[:, 0, :]
    argmin
    
    argmin = torch.argmin(squared_dist, dim=1).unsqueeze(1)
    min_encodings = torch.zeros(argmin.shape[0], n_embeds)
    min_encodings.scatter_(1, argmin, 1)
    z_q = torch.matmul(min_encodings, embed_space.weight)
    z_q