import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
from moviepy.video.io.bindings import mplfig_to_npimage
from pathlib import Path
import os
import numpy as np
import random


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def denorm(tensor):
    tensor /= 2
    tensor += 0.5
    return tensor


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def to_pil(img):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img)
        return image
    else:
        return img


def save_image(image, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(str(save_path), quality=100)


def plt_to_pil(fig):
    img = mplfig_to_npimage(fig)
    image = to_pil(img)
    return image


def save_model_params(model, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"Saved model params as '{str(save_path)}'.")


def load_model_params(model, model_params, device, strict):
    state_dict = torch.load(model_params, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model params from '{str(model_params)}'.")
    # return model
