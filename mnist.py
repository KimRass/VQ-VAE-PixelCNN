from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def get_mnist_dls(data_dir, batch_size, n_cpus, val_ratio=0.1):
    transformer = T.Compose(
        # [T.Pad(padding=2), T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
    )
    train_val_ds = MNIST(root=data_dir, train=True, download=True, transform=transformer)
    train_ds, val_ds = random_split(train_val_ds, lengths=(1 - val_ratio, val_ratio))
    test_ds = MNIST(root=data_dir, train=False, download=True, transform=transformer)

    train_dl = DataLoader(
        train_ds,
        # random_split(train_ds, lengths=(batch_size, len(train_ds) - batch_size))[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=True,
        drop_last=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpus,
        pin_memory=False,
        drop_last=True,
    )
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    data_dir = "/Users/jongbeomkim/Documents/datasets"
