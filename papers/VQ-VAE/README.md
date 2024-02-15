- CIFAR-10에 대해 학습시키는 상황을 가정하겠습니다.

# 1st Training Stage
- VQ-VAE를 통해서 원본 이미지를 그대로 복원합니다. Categorical distribution $q(z \vert x)$으로부터 이미지를 생성하는 Decoder를 학습시킵니다. Standard VAE처럼 Decoder를 위해서 Encoder를 함께 사용합니다.
- Input image $x$: (b, 3, 32, 32)
- Encoder의 출력 $z_{e}(x)$: (b, $D$, 8, 8) (Continous)
- Embedding space: $D$차원의 $K$개의 Vectors $e_{1}, e_{2}, \ldots, e_{K}$로 구성됩니다.
- $z_{e}(x)$의 64 (8 × 8)개의 Vector 각각에 대해서 $e_{1}, e_{2}, \ldots, e_{K}$ 중 가장 가까운 것의 Index를 찾습니다.
- 이 Index가 Posterior distribution $q(z \vert x)$ (Categorical)입니다.
- $q(z \vert x)$: (b, 8, 8)
- Decoder의 입력 $z_{q}(x)$: $q(z \vert x)$를 가지고 Embedding space에서 Indexing (b, $D$, 8, 8) (Continuous)
- Decoder를 통해서 (b, 3, 32, 32)으로 복원
$$L = \log p(x \vert z_{q}(x)) + \Vert \text{sg}[z_{e}(x)] - e \Vert^{2}_{2} + \beta \Vert z_{e}(x) - \text{sg}[e] \Vert^{2}_{2}$$
- `z_q = z_e + (z_q - z_e).detach()`
- The 1st term: Reconstruction loss
    - $x$와 Decoder out 간의 MSE loss
    - Decoder를 훈련시키고 Gradient copying을 통해 Encoder도 훈련시킵니다.
- The 2nd term:
    - Encoder output을 고정시키고 Embedding space의 각 Vectors를 이동시킵니다.
- The 3rd term:
    - Embedding space의 각 Vectors를 고정시키고 Encoder output을 이동시킵니다.

# 2nd Training Stage
- 이제 모델이 본 적 없는 이미지를 생성할 수 있도록 학습시켜야 합니다. PixelCNN에 Categorical distribution $q(z \vert x)$을 학습시킵니다.
- Categorical distribution을 따르는 (b, 8, 8)의 텐서를 생성할 수 있다면 이를 Decoder에 입력하여 새로운 이미지를 생성할 수 있습니다.
- DALL-E처럼 Transformer 기반의 모델을 사용할 수도 있지만 논문에서는 PixelCNN을 사용했다고 합니다.
- 여러 개의 Conv layers를 통과하면서 Spatial dimension은 항상 그대로 유지되며 채널의 수만 변합니다.
- PixelCNN에 Categorical tensor를 입력할 것이므로 맨 처음 One-hot encoding을 하거나 Embedding layer를 사용합니다.
- CIFAR-10의 훈련 셋의 이미를 VQ-VAE를 고정한 후 VQ-VAE에 입력해서 Posterior distribution을 얻고 이를 Ground truth로 삼습니다. 이것을 PixelCNN에 입력하여 나온 출력을 에측 값으로 해서 Cross entropy loss를 계산합니다. Transformer 기반의 모델처럼 입력 데이터를 한 Timestep씩 뒤로 미뤄서 Ground truth를 만들 필요가 없습니다.

# Sampling
- 모델의 출력은 Shape이 (b, $K$, 8, 8)이 되고 이 중에서 생성할 픽셀에 대해서 샘플링합니다.
- 샘플링에는 시행횟수 1, 경우의 수가 $K$인 다항분포를 사용할 수 있습니다.
- 64 (8 × 8)번의 Inference를 통해 $z$를 생성한 후 VQ-VAE의 Decoder에 입력하여 이미지를 생성합니다.
