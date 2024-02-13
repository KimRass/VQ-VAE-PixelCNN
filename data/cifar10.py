import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np

from data.fashion_mnist import dses_to_dls


IMG_SIZE = 32

def load_data(data_path):
    with open(data_path, mode="rb") as f:
        data_dic = pickle.load(f, encoding="bytes")

    imgs = data_dic[b"data"]
    imgs = imgs.reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    imgs = imgs.transpose(0, 2, 3, 1)

    labels = data_dic[b"labels"]
    labels = np.array(labels)
    return imgs, labels


def load_train_val_data(data_dir):
    ls_imgs = list()
    ls_labels = list()
    for idx in range(1, 6):
        imgs, labels = load_data(Path(data_dir)/f"data_batch_{idx}")
        ls_imgs.append(imgs)
        ls_labels.append(labels)
    imgs = np.concatenate(ls_imgs, axis=0)
    labels = np.concatenate(ls_labels, axis=0)
    return imgs, labels


def load_all_data(data_dir, val_ratio, seed):
    train_val_imgs, train_val_labels = load_train_val_data(data_dir)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_val_imgs,
        train_val_labels,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )
    test_imgs, test_labels = load_data(Path(data_dir)/"test_batch")
    return (
        train_imgs,
        train_labels,
        val_imgs,
        val_labels,
        test_imgs,
        test_labels,
    )


class CIFAR10DS(Dataset):
    def __init__(self, imgs, labels, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()

        self.imgs = imgs
        self.labels = labels

        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image = Image.fromarray(img, mode="RGB")
        image = self.transform(image)

        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label


def get_dls(data_dir, batch_size, n_cpus, val_ratio, seed):
    (
        train_imgs,
        train_labels,
        val_imgs,
        val_labels,
        test_imgs,
        test_labels,
    ) = load_all_data(
        data_dir=data_dir, val_ratio=val_ratio, seed=seed,
    )
    train_ds = CIFAR10DS(imgs=train_imgs, labels=train_labels)
    val_ds = CIFAR10DS(imgs=val_imgs, labels=val_labels)
    test_ds = CIFAR10DS(imgs=test_imgs, labels=test_labels)
    return dses_to_dls(train_ds, val_ds, test_ds, batch_size=batch_size, n_cpus=n_cpus)


if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_dls(
        data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py",
        batch_size=4,
        n_cpus=1,
    )
    a, b = next(iter(train_dl))
    print(a.shape, b.shape)
    print(b)
