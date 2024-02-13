# 1. Pre-trained Models
## 1) On Fashion MNIST
- [vqvae-fashion_mnist.pth](https://drive.google.com/file/d/177ZdygNvstZgM539HFObUjG46rJTCrWc/view?usp=sharing)
- Trained VQ-VAE for 47 epochs. (Validation loss: 0.145)
    ```bash
    dataset="fashion_mnist"
    batch_size=128
    lr=0.0002
    n_embeds=128
    hidden_dim=256
    n_pixelcnn_res_blocks=2
    n_pixelcnn_conv_blocks=2
    ```
- Then trained PixelCNN for 14 epochs. (Validataion loss: 1.279)
    ```bash
    dataset="fashion_mnist"
    batch_size=128
    lr=0.0002
    n_embeds=128
    hidden_dim=256
    n_pixelcnn_res_blocks=2
    n_pixelcnn_conv_blocks=2
    ```
## 2) On CIFAR-10
- [vqvae-cifar10.pth](https://drive.google.com/file/d/1kyTe8U0frTDCPDdYxyuHiyFYzFEbmDVy/view?usp=sharing)
- Trained VQ-VAE for 80 epochs. (Validation loss: 0.164)
    ```bash
    dataset="cifar10"
    batch_size=128
    lr=0.0002
    n_embeds=32
    hidden_dim=128
    n_pixelcnn_res_blocks=3
    n_pixelcnn_conv_blocks=2
    ```
- Then trained PixelCNN for 60 epochs. (Validataion loss: 2.469)
    ```bash
    dataset="cifar10"
    batch_size=128
    lr=0.0002
    n_embeds=32
    hidden_dim=128
    n_pixelcnn_res_blocks=3
    n_pixelcnn_conv_blocks=2
    ```
- `n_embeds=32`는 너무 작은 값인 것 같습니다.

# 2. Samples
- <img src="https://github.com/KimRass/KimRass/assets/67457712/4d1a8d21-c589-43b1-b37f-dde2d5e4b7de" width="500">
- <img src="https://github.com/KimRass/KimRass/assets/67457712/9c0570d7-9d25-457b-923b-83a3f0481389" width="600">

# 3. Implementation Details
## 1) `detach()`
- VQ-VAE 학습에서 Loss 계산 시 `z_q = z_e + (z_q - z_e).detach()`를 추가할 시 학습이 더 빨라지는 것을 확인했으나, 정확히 어떤 기능을 하는지까지는 알지 못했습니다.
