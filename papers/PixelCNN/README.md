- References:
    - https://github.com/singh-hrituraj/PixelCNN-Pytorch?tab=readme-ov-file

# 1. Introduction
- 2가지 Architecture를 제안합니다.
- PixelRNN:
    - 최대 12개의 2D LSTM layers로 구성되며 Convolutional layers도 사용합니다.
    - 레이어는 2가지입니다; Row LSTM layer and Diagonal BiLSTM layer.
    - Residual connection도 사용합니다.
- PixelCNN:
    - Masked convolutions를 사용함으로써 CNN을 Sequence modeling에 사용합니다.
    - 15개의 레이어들로 이루어진 Fully-convolutional network입니다. 데이터는 모델을 통과하는 내내 동일한 해상도를 유지하고 모델의 출력은 각 위치에서의 조건부 확률 분포입니다.
- 두 가지 Architectures 모두 다 픽셀간의 그리고 픽셀 내에서 RGB 색 공간 값 사이의 의존성을 모델링 가능합니다.
- Softmax layer를 통해 Multinomial distribution을 모델링합니다.

# 2. Model
- 목적:
    - 이미지에 대한 분포를 추정함으로써 Likelihood를 계산합니다.
    - 새로운 이미지를 생성합니다.
- 모델은 이미지를 한 번에 한 행씩 그리고 행에서는 한 픽셀씩 스캔합니다. 이미 스캔한 문맥을 조건으로 해서 이번에 스캔하는 픽셀의 조건부 확률 분포를 예측합니다. 스캔에 사용되는 Model parameters는 위치에 관계 없이 동일합니다.
## 2.1) Generating an Image Pixel by Pixel
- 이미지 $\textbf{x}$에 대한 Joint probability $p(\textbf{x})$를 Conditional probability distribution의 곱으로 분해합니다.
$$p(\textbf{x}) = \prod^{n^{2}}_{i = 1}p(x_{i} \vert x_{1}, \ldots, x_{i - 1})$$
- 각 픽셀마다 RGB 색 공간의 3개의 값이 결정되어야만 하나의 픽셀의 분포가 결정됩니다. RGB 순서대로 모델이 값을 예측하며 G의 값을 예측 시에는 R의 값을 문맥으로 사용하고, B의 값을 예측 시에는 R과 G의 값을 문맥으로 사용합니다.
- 계산 자체는 병렬적으로 이루어지지만 픽셀을 하나씩 생성하는 과정은 순차적입니다.
## 2.2) Pixels as Discrete Variables
- 모델의 출력을 Continuous distribution이 아닌 Softmax layer를 통한 Multinomial distribution으로 모델링하는 이유는 이 편이 실험 결과 학습이 더 잘 되고 모델의 성능도 좋았기 때문입니다.

# 3. Pixel Recurrent Neural Networks
## 3.3) Residual Connections
- ResNet에서 사용되었던 Residual connections를 사용합니다. Residual block 내에서 체널의 수가 절반으로 줄었다가 다시 원래대로 돌아오고 해상도는 전후에 동일하게 유지됩니다.
## 3.4) Masked Convolution