import torch
from torch.optim import AdamW
from pathlib import Path
import math
import argparse
from tqdm import tqdm
import re

from utils import get_device, set_seed, image_to_grid, save_image
from model import VQVAE


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--vqvae_params", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--n_embeds", type=int, default=128, required=False)
    # "All having 256 hidden units."
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--n_pixelcnn_res_blocks", type=int, default=2, required=False)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_cpus", type=int, default=0, required=False)
    # "Evaluate the performance after 250,000 steps with batch-size 128."
    parser.add_argument("--n_epochs", type=int, default=2000, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    # "With learning rate 2e-4."
    parser.add_argument("--lr", type=float, default=0.0002, required=False)
    parser.add_argument("--val_ratio", type=float, default=0.2, required=False)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


class Trainer(object):
    def __init__(self, train_dl, val_dl, test_dl, device):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device

    def train_single_step(self, ori_image, model, optim):
        ori_image = ori_image.to(self.device)
        loss = model.get_pixelcnn_loss(ori_image)

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    @torch.no_grad()
    def validate(self, model):
        model.eval()

        cum_val_loss = 0
        for ori_image, _ in self.val_dl:
            ori_image = ori_image.to(self.device)
            loss = model.get_pixelcnn_loss(ori_image)
            cum_val_loss += loss.item()
        val_loss = cum_val_loss / len(self.val_dl)

        model.train()
        return val_loss

    def save_model_params(self, model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_path))

    def train(self, init_epoch, n_epochs, save_dir, model, optim):
        test_ori_image, _ = next(iter(self.test_dl))
        test_ori_image = test_ori_image.to(self.device).detach()
        test_ori_grid = image_to_grid(
            test_ori_image, n_cols=int(self.train_dl.batch_size ** 0.5),
        )
        save_image(test_ori_grid, save_path=Path(save_dir)/f"test_ori_image.jpg")

        recon_image1 = model(test_ori_image.detach())
        recon_grid1 = image_to_grid(
            recon_image1, n_cols=int(self.train_dl.batch_size ** 0.5),
        )
        save_image(
            recon_grid1, save_path=Path(save_dir)/f"recon_image.jpg",
        )

        best_val_loss = math.inf
        for epoch in range(init_epoch, init_epoch + n_epochs):
            cum_train_loss = 0
            for ori_image, _ in tqdm(self.train_dl, leave=False):
                loss = self.train_single_step(ori_image, model=model, optim=optim)
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(self.train_dl)

            val_loss = self.validate(model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                filename = f"epoch={epoch}-val_loss={val_loss:.5f}.pth"
                self.save_model_params(model, save_path=Path(save_dir)/filename)

            log = f"""[ {epoch}/{n_epochs} ]"""
            log += f"[ Train loss: {train_loss:.5f} ]"
            log += f"[ Val loss: {val_loss:.5f} | Best: {best_val_loss:.5f} ]"
            print(log)

            with torch.no_grad():
                recon_image = model.reconstruct(test_ori_image.detach())
                recon_grid = image_to_grid(
                    recon_image, n_cols=int(self.train_dl.batch_size ** 0.5),
                )
                save_image(
                    recon_grid,
                    save_path=Path(save_dir)/f"epoch={epoch}-recon_image.jpg",
                )


def ckpt_path_to_init_epoch(ckpt_path):
    return int(re.search(pattern=r"epoch=(\d+)-", string=ckpt_path).group(1)) + 1


def main():
    args = get_args()
    set_seed(args.SEED)
    # DEVICE = get_device()
    DEVICE = torch.device("cpu")

    print(f"[ DEVICE: {DEVICE} ][ N_CPUS: {args.N_CPUS} ]")

    if args.DATASET == "fashion_mnist":
        from fashion_mnist import get_dls
        CHANNELS = 1
    elif args.DATASET == "cifar10":
        from cifar10 import get_dls
        CHANNELS = 3
    train_dl, val_dl, test_dl = get_dls(
        data_dir=args.DATA_DIR,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        val_ratio=args.VAL_RATIO,
        seed=args.SEED,
    )

    model = VQVAE(
        channels=CHANNELS,
        n_embeds=args.N_EMBEDS,
        hidden_dim=args.HIDDEN_DIM,
        n_pixelcnn_res_blocks=args.N_PIXELCNN_RES_BLOCKS,
    ).to(DEVICE)
    state_dict = torch.load(args.VQVAE_PARAMS, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    if args.RESUME_FROM:
        state_dict = torch.load(args.RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(state_dict)
    optim = AdamW(model.parameters(), lr=args.LR)

    if args.RESUME_FROM:
        init_epoch = ckpt_path_to_init_epoch(args.RESUME_FROM)
    else:
        init_epoch = 1
    trainer = Trainer(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        device=DEVICE,
    )
    trainer.train(
        init_epoch=init_epoch,
        n_epochs=args.N_EPOCHS,
        save_dir=args.SAVE_DIR,
        model=model,
        optim=optim,
    )

if __name__ == "__main__":
    main()
