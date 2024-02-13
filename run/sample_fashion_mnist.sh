#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../sample.py\
    --dataset="fashion_mnist"\
    --model_params="/Users/jongbeomkim/Documents/vqvae/vqvae_fashion_mnist.pth"\
    --save_dir="/Users/jongbeomkim/Desktop/workspace/VQ-VAE/samples"\
    --batch_size=100\
    --n_samples=10\
    --temp=1\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
