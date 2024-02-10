# References:
    # https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
    # https://keras.io/examples/generative/vq_vae/

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from pathlib import Path

torch.set_printoptions(linewidth=70)


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

    def vector_quantize(self, x): # (b, `n_embeds`, h, w)
        b, _, h, w = x.shape
        x = rearrange(x, pattern="b c h w -> (b h w) c")
        squared_dist = ((x.unsqueeze(1) - self.embed_space.weight.unsqueeze(0)) ** 2).sum(dim=2)
        # "The discrete latent variables $z$ are then calculated by a nearest neighbour look-up
        # using the shared embedding space $e$.
        argmin = torch.argmin(squared_dist, dim=1) # (b * h * w,)
        q = argmin.view(b, h, w) # (b, h, w)
        return q

    def forward(self, x):
        q = self.vector_quantize(x) # (b, h, w)
        x = self.embed_space(q) # (b, h, w, `hidden_dim`)
        return x.permute(0, 3, 1, 2)


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


# "When predicting the R channel for the current pixel xi, only the generated pixels left and above of xi can be used as context. When predicting the G channel, the value of the R channel can also be used as context in addition to the previously generated pixels. Likewise, for the B channel, the values of both the R and G channels can be used. To restrict connections in the network to these dependencies, we apply a mask to the in_imageto- state convolutions and to other purely convolutional layers in a PixelRNN."
# "Mask A is applied only to the first convolutional layer in a PixelRNN and restricts the connections to those neighboring pixels and to those colors in the current pixels that have already been predicted."
class MaskedConv(nn.Conv2d):
    def __init__(self, *args, mask_type, **kwargs):
        super().__init__(*args, **kwargs)

        self.mask = torch.zeros_like(self.weight)
        self.mask[..., : self.mask.shape[2] // 2, :] = 1
        self.mask[..., self.mask.shape[2] // 2, : self.mask.shape[3] // 2] = 1
        if mask_type == "B":
            self.mask[..., self.mask.shape[2] // 2, self.mask.shape[3] // 2] = 1
        self.weight.data *= self.mask


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, mask_type):
        super().__init__()

        self.layers1 = nn.Sequential(
            MaskedConv(
                hidden_dim, hidden_dim // 2, 1, 1, 0, bias=False, mask_type=mask_type,
            ),
            # nn.Conv2d(hidden_dim, hidden_dim // 2, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            MaskedConv(
                hidden_dim // 2, hidden_dim // 2, 3, 1, 1, bias=False, mask_type=mask_type,
            ),
            # nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            MaskedConv(
                hidden_dim // 2, hidden_dim, 1, 1, 0, bias=False, mask_type=mask_type,
            ),
            # nn.Conv2d(hidden_dim // 2, hidden_dim, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.layers2 = nn.Sequential(
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.layers1(x)
        x = self.layers2(x)
        return x


class PixelCNN(nn.Module):
    def __init__(self, n_embeds, hidden_dim, n_res_blocks=2):
        super().__init__()

        self.n_embeds = n_embeds
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_embeds, hidden_dim)
        self.conv_block1 = nn.Sequential(
            MaskedConv(hidden_dim, hidden_dim, 7, 1, 3, bias=False, mask_type="A"),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[
                ResBlock(hidden_dim=hidden_dim, mask_type="B")
                for _ in range(n_res_blocks)
            ]
        )
        self.conv_block2 = nn.Sequential(
            MaskedConv(hidden_dim, hidden_dim, 1, 1, 0, bias=False, mask_type="B"),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.conv = MaskedConv(hidden_dim, n_embeds, 1, 1, 0, mask_type="B")
        # self.conv = nn.Conv2d(hidden_dim, n_embeds, 1, 1, 0)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_block1(x)
        x = self.res_blocks(x)
        x = self.conv_block2(x)
        x = self.conv(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, channels, n_embeds, hidden_dim, n_pixelcnn_res_blocks):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.enc = Encoder(channels=channels, hidden_dim=hidden_dim)
        self.vect_quant = VectorQuantizer(n_embeds=n_embeds, hidden_dim=hidden_dim)
        self.dec = Decoder(channels=channels, hidden_dim=hidden_dim)

        self.pixelcnn = PixelCNN(
            n_embeds=n_embeds, hidden_dim=hidden_dim, n_res_blocks=n_pixelcnn_res_blocks,
        )
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def encode(self, x):
        x = self.enc(x)
        return x

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, ori_image):
        # "The model takes an input $x$, that is passed through an encoder
        # producing output $z_{e}(x)$.
        z_e = self.encode(ori_image) # "$z_{e}(x)$"
        z_q = self.vect_quant(z_e)
        x = self.decode(z_q)
        return x

    def save_model_params(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_path))

    def load_model_params(self, model_params, strict):
        state_dict = torch.load(model_params, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)

    def get_vqvae_loss(self, ori_image, commit_weight=0.25):
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
        z_q = z_e + (z_q - z_e).detach() # Straight-through estimator.(?)

        recon_image = self.decode(z_q)
        # This term trains the decoder.
        recon_loss = F.mse_loss(recon_image, ori_image, reduction="mean")
        return recon_loss + vq_loss + commit_loss

    def get_prior_q(self, ori_image):
        z_e = self.encode(ori_image)
        q = self.vect_quant.vector_quantize(z_e) # "$q(z \vert x)$"
        return q

    def get_pixelcnn_loss(self, ori_image):
        with torch.no_grad():
            q = self.get_prior_q(ori_image)
        pred_q = self.pixelcnn(q.detach())
        return self.ce(
            rearrange(pred_q, pattern="b c h w -> (b h w) c"), q.view(-1,),
        )

    @staticmethod
    def deterministically_sample(x):
        return torch.argmax(x, dim=1)

    @staticmethod
    def stochastically_sample(x, temp=1):
        b, c, h, w = x.shape
        prob = F.softmax(x / temp, dim=1)
        # print(prob[0, :, 0, 0])
        sample = torch.multinomial(prob.view(-1, c), num_samples=1, replacement=True)
        return sample.view(b, h, w)

    def q_to_image(self, q):
        x = self.vect_quant.embed_space(q)
        return self.decode(x.permute(0, 3, 1, 2))

    def reconstruct(self, ori_image, temp=0):
        with torch.no_grad():
            q = self.get_prior_q(ori_image)
        pred_q = self.pixelcnn(q.detach())
        if temp == 0:
            recon_q = self.deterministically_sample(pred_q)
        elif temp > 0:
            recon_q = self.stochastically_sample(pred_q)
        return self.q_to_image(recon_q)

    @torch.no_grad()
    def sample_post_q(self, batch_size, q_size, device, temp=0):
        sampled_q = torch.zeros(
            size=(batch_size, q_size, q_size), dtype=torch.int64, device=device,
        )
        for row in range(q_size):
            for col in range(q_size):
                pred_q = self.pixelcnn(sampled_q.detach())
                # print(torch.max(F.softmax(pred_q[..., row, col] / temp, dim=1), dim=1)[0])
                # F.softmax(pred_q)
                if temp == 0:
                    recon_q = self.deterministically_sample(pred_q)
                elif temp > 0:
                    recon_q = self.stochastically_sample(pred_q)
                sampled_q[:, row, col] = recon_q[:, row, col]
        return sampled_q

    def sample(self, batch_size, q_size, device, temp=0):
        post_q = self.sample_post_q(
            batch_size=batch_size, q_size=q_size, device=device, temp=temp,
        )
        return self.q_to_image(post_q)


if __name__ == "__main__":
    # img_size = 28
    # channels = 1
    # n_embeds = 128
    # hidden_dim = 256
    # DEVICE = torch.device("cpu")
    # VQVAE_PARAMS = "/Users/jongbeomkim/Documents/vqvae/pixelcnn/fashion_mnist/only_masked_conv/epoch=4-val_loss=0.00008.pth"

    model = VQVAE(1, n_embeds, hidden_dim, 2).to(DEVICE)
    state_dict = torch.load(VQVAE_PARAMS, map_location=DEVICE)
    model.load_state_dict(state_dict)

    sampled_image = model.sample(batch_size=1, q_size=7, device=DEVICE, temp=0.4)
    sampled_grid = image_to_grid(sampled_image, n_cols=1)
    sampled_grid.show()

    # q = torch.tensor(
    #     [[ 77,  38,  38,  41,  41,  41,  77],
    #     [ 77,  38,  38,  15,  82,  15,  26],
    #     [ 77,  38,  38,  16,  31,  51, 117],
    #     [ 77,  38,  38,  15,  51,  51, 117],
    #     [ 41,  28,  62,  62,  51,  51, 117],
    #     [ 62,  51,  51,  51,  51,  51,  51],
    #     [ 77,  99,  41,  41,  41,  41,  41]]
    # )[None, ...]
    # image = model.q_to_image(q)
    # image_to_grid(image, n_cols=1).show()

    # train_dl, val_dl, test_dl = get_dls(
    #     data_dir="/Users/jongbeomkim/Documents/datasets",
    #     batch_size=4,
    #     n_cpus=1,
    #     val_ratio=0.2,
    #     seed=888,
    # )
    for ori_image, x in train_dl:
        with torch.no_grad():
            recon_image = model.reconstruct(ori_image)
            recon_grid = image_to_grid(recon_image, n_cols=2)
            recon_grid.show()
            break
