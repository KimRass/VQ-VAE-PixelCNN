from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def get_dls(data_dir, batch_size, n_cpus, val_ratio=0.2):
    transformer = T.Compose(
        # [T.Pad(padding=2), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
    )
    train_val_ds = FashionMNIST(root=data_dir, train=True, download=True, transform=transformer)
    train_ds, val_ds = random_split(train_val_ds, lengths=(1 - val_ratio, val_ratio))
    test_ds = FashionMNIST(root=data_dir, train=False, download=True, transform=transformer)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=n_cpus,
    )
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets"
