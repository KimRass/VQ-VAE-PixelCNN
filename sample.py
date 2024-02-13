import torch
import argparse

from utils import get_device, set_seed, image_to_grid, save_image, load_model_params
from model import VQVAE


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--data_dir", type=str, required=True)
    # parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--temp", type=float, default=1, required=False)

    # Architecture
    parser.add_argument("--n_embeds", type=int, default=128, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--n_pixelcnn_res_blocks", type=int, required=False)
    parser.add_argument("--n_pixelcnn_conv_blocks", type=int, required=False)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


def main():
    args = get_args()
    # set_seed(args.SEED)
    DEVICE = get_device()
    # DEVICE = torch.device("cpu")

    if args.DATASET == "fashion_mnist":
        CHANNELS = 1
        Q_SIZE = 7
    elif args.DATASET == "cifar10":
        CHANNELS = 3
        Q_SIZE = 8

    model = VQVAE(
        channels=CHANNELS,
        n_embeds=args.N_EMBEDS,
        hidden_dim=args.HIDDEN_DIM,
        n_pixelcnn_res_blocks=args.N_PIXELCNN_RES_BLOCKS,
        n_pixelcnn_conv_blocks=args.N_PIXELCNN_CONV_BLOCKS,
    ).to(DEVICE)
    # load_model_params(model=model, model_params=args.MODEL_PARAMS, device=DEVICE, strict=True)
    load_model_params(model=model, model_params=args.MODEL_PARAMS, device=DEVICE, strict=False)

    sampled_image = model.sample(
        batch_size=args.BATCH_SIZE, q_size=Q_SIZE, device=DEVICE, temp=args.TEMP,
    )
    sampled_grid = image_to_grid(sampled_image, n_cols=int(args.BATCH_SIZE ** 0.5))
    sampled_grid.show()


if __name__ == "__main__":
    main()
