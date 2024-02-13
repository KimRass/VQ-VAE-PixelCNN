# 1. Pre-trained Models
<!-- | Model params | SEED | LR |N_EMBEDS | HIDDEN_DIM | N_PIXELCNN_RES_BLOCKS | N_PIXELCNN_CONV_BLOCKS | N_EPOCHS | Validation loss |
|-|-|-|-|-|-|-|-|-|
| [vqvae_fashion_mnist.pth](https://drive.google.com/file/d/1eR3jIti3uXCGO8ejbT1mnHLxxCdzdUxe/view?usp=sharing) | 888 | 0.0002 | 128 | 256 | 2 | 2 | 47 using VQ-VAE loss<br>14 using PixelCNN loss | 0.145<br>1.279 |
| [vqvae_cifar10.pth](https://drive.google.com/file/d/1_x5LPfxdWDa-gdhlFhGR0UE9jOwkbqM9/view?usp=sharing) | 888 | 0.0002 | 128 | 256 | ? | ? | 271 using VQ-VAE loss<br>? using PixelCNN loss | 0.141<br>? | -->

- vqvae_cifar10.pth
    - Trained on CIFAR-10.
    - Trained for 80 epochs.
    - Validation loss: 0.164
    ```bash
    !python3 train_pixelcnn.py\
        --dataset="cifar10"\
        --data_dir="/content/drive/MyDrive"\
        --save_dir="/content/drive/MyDrive/vqvae/pixelcnn-cifar10/"\
        --vqvae_params="/content/drive/MyDrive/vqvae/vqvae-cifar10/epoch=80-val_loss=0.164.pth"\
        --n_epochs=300\
        --batch_size=128\
        --n_cpus=2\
        --lr=0.0002\
        --n_embeds=32\
        --hidden_dim=128\
        --n_pixelcnn_res_blocks=3\
        --n_pixelcnn_conv_blocks=2
    ```


# 2. Implementation Details
## 1) `detach()`
- VQ-VAE 학습에서 Loss 계산 시 `z_q = z_e + (z_q - z_e).detach()`를 추가할 시 학습이 더 빨라지는 것을 확인했으나, 정확히 어떤 기능을 하는지까지는 알지 못했습니다.
