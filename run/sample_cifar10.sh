#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../sample.py\
    --dataset="cifar10"\
    --save_dir="/Users/jongbeomkim/Desktop/workspace/VQ-VAE/samples"\
    --model_params="/Users/jongbeomkim/Documents/vqvae/epoch=96-val_loss=2.226.pth"\
    --batch_size=100\
    --n_samples=10\
    --temp=1\
    --n_embeds=128\
    --hidden_dim=64\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
