import torch
from torch.optim import AdamW
from pathlib import Path
import math
import argparse
from tqdm import tqdm

from utils import get_device, set_seed, image_to_grid, save_image
from model import VQVAE


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--n_embeds", type=int, default=512, required=False)
    # "All having 256 hidden units."
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--commit_weight", type=float, default=0.25, required=False)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_cpus", type=int, default=0, required=False)
    # "Evaluate the performance after 250,000 steps."
    parser.add_argument("--n_epochs", type=int, default=2000, required=False)
    # "With batch-size 128."
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    # "With learning rate 2e-4."
    parser.add_argument("--lr", type=float, default=0.0002, required=False)
    parser.add_argument("--val_ratio", type=float, default=0.2, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def train_single_step(ori_image, model, optim, commit_weight, device):
    ori_image = ori_image.to(device)
    loss = model.get_loss(ori_image, commit_weight=commit_weight)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


@torch.no_grad()
def validate(val_dl, model, commit_weight, device):
    model.eval()

    cum_val_loss = 0
    for ori_image, _ in val_dl:
        ori_image = ori_image.to(device)
        loss = model.get_loss(ori_image, commit_weight=commit_weight)
        cum_val_loss += loss.item()
    val_loss = cum_val_loss / len(val_dl)

    model.train()
    return val_loss


def save_state_dict(state_dict, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(save_path))


def train(
    n_epochs, train_dl, val_dl, test_dl, model, optim, save_dir, commit_weight, device,
):
    test_di = iter(test_dl)

    best_val_loss = math.inf
    for epoch in range(1, n_epochs + 1):
        cum_train_loss = 0
        for ori_image, _ in tqdm(train_dl, leave=False):
            loss = train_single_step(
                ori_image=ori_image,
                model=model,
                optim=optim,
                commit_weight=commit_weight,
                device=device,
            )
            cum_train_loss += loss.item()
        train_loss = cum_train_loss / len(train_dl)

        val_loss = validate(
            val_dl=val_dl, model=model, commit_weight=commit_weight, device=device,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            filename = f"epoch={epoch}-train_loss={train_loss:.3f}-val_loss={val_loss:.3f}.pth"
            save_state_dict(model.state_dict(), save_path=Path(save_dir)/filename)

        log = f"""[ {epoch}/{n_epochs} ]"""
        log += f"[ Train loss: {train_loss:.3f} ]"
        log += f"[ Val loss: {val_loss:.3f} | {best_val_loss:.3f} ]"
        print(log)

        test_ori_image, _ = next(test_di)
        test_ori_image = test_ori_image.to(device)
        test_ori_grid = image_to_grid(test_ori_image, n_cols=int(train_dl.batch_size ** 0.5))
        save_image(test_ori_grid, save_path=Path(save_dir)/f"epoch={epoch}-test_ori_image.jpg")

        recon_image = model(test_ori_image)
        recon_grid = image_to_grid(recon_image, n_cols=int(train_dl.batch_size ** 0.5))
        save_image(recon_grid, save_path=Path(save_dir)/f"epoch={epoch}-recon_image.jpg")


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

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
        channels=CHANNELS, n_embeds=args.N_EMBEDS, hidden_dim=args.HIDDEN_DIM,
    ).to(DEVICE)
    # "We use the ADAM optimiser."
    optim = AdamW(model.parameters(), lr=args.LR)

    train(
        n_epochs=args.N_EPOCHS,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        model=model,
        optim=optim,
        save_dir=args.SAVE_DIR,
        commit_weight=args.COMMIT_WEIGHT,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
