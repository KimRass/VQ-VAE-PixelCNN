#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../sample.py\
    --dataset="cifar10"\
    --save_dir="/Users/jongbeomkim/Desktop/workspace/VQ-VAE/samples"\
    --model_params="/Users/jongbeomkim/Documents/vqvae/vqvae-cifar10.pth"\
    --batch_size=100\
    --n_samples=10\
    --temp=1\
    --n_embeds=32\
    --hidden_dim=128\
    --n_pixelcnn_res_blocks=3\
    --n_pixelcnn_conv_blocks=2\
