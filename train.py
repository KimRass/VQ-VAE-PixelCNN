# References:
    # https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

import torch
from torch.optim import AdamW
from pathlib import Path
import math
import argparse
from tqdm import tqdm
import multiprocessing as mp

from utils import get_device, set_seed, image_to_grid, save_image
from mnist import get_mnist_dls
from model import VQVAE


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_epochs", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--lr", type=float, default=0.0005, required=False)
    parser.add_argument("--n_embeds", type=int, default=512, required=False)
    parser.add_argument("--embed_dim", type=int, default=64, required=False)
    parser.add_argument("--beta", type=float, default=0.25, required=False)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def train_single_step(ori_image, model, optim, beta, device):
    ori_image = ori_image.to(device)
    loss = model.get_loss(ori_image=ori_image, beta=beta)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


@torch.no_grad()
def validate(val_dl, model, beta, device):
    model.eval()

    cum_loss = 0
    # cum_recon_loss = 0
    # cum_kld_loss = 0
    for ori_image, _ in val_dl:
        ori_image = ori_image.to(device)
        loss = model.get_loss(ori_image=ori_image, beta=beta)

        cum_loss += loss.item()
        # cum_recon_loss += recon_loss.item()
        # cum_kld_loss += kld_loss.item()
    val_loss = cum_loss / len(val_dl)

    model.train()
    return val_loss


def train(n_epochs, train_dl, val_dl, model, optim, save_dir, beta, device):
    # best_val_recon_loss = math.inf
    best_val_loss = math.inf
    for epoch in range(1, n_epochs + 1):
        cum_recon_loss = 0
        # cum_kld_loss = 0
        for ori_image, _ in tqdm(train_dl, leave=False):
            loss = train_single_step(
                ori_image=ori_image,
                model=model,
                optim=optim,
                beta=beta,
                device=device,
            )
            cum_recon_loss += loss.item()
            # cum_kld_loss += kld_loss.item()
        train_recon_loss = cum_recon_loss / len(train_dl)
        # train_kld_loss = cum_kld_loss / len(train_dl)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ Train loss: {train_recon_loss:.4f} ]"
        # log += f"[ Train KLD loss: {train_kld_loss:.4f} ]"
        print(log)

        val_loss = validate(
            val_dl=val_dl, model=model, beta=beta, device=device,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                str(Path(save_dir)/f"epoch_{epoch}-val_loss_{val_loss:.4f}.pth"),
            )

        log = f"[ Val loss: {val_loss:.4f} ]"
        # log += f"[ Val kld loss: {val_kld_loss:.4f} ]"
        # log += f"[ Best val recon loss: {best_val_recon_loss:.4f} ]"
        log += f"[ Best val loss: {best_val_loss:.4f} ]"
        print(log)

        ori_image = ori_image.to(device)
        recon_image = model.reconstruct(ori_image)
        recon_grid = image_to_grid(recon_image, n_cols=8)
        save_image(recon_grid, Path(save_dir)/f"epoch_{epoch}.jpg")


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()
    N_CPUS = mp.cpu_count() - 1
    # N_CPUS = 0
    print(f"[ DEVICE: {DEVICE} ][ N_CPUS: {N_CPUS} ]")

    train_dl, val_dl, _ = get_mnist_dls(
        data_dir=args.DATA_DIR, batch_size=args.BATCH_SIZE, n_cpus=N_CPUS,
    )

    model = VQVAE(channels=1, n_embeds=args.N_EMBEDS, embed_dim=args.EMBED_DIM).to(DEVICE)
    optim = AdamW(model.parameters(), lr=args.LR)

    train(
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optim=optim,
        save_dir=args.SAVE_DIR,
        beta=args.BETA,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
