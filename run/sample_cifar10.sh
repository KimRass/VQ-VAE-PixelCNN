#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../sample.py\
    --dataset="fashion_mnist"\
    --model_params="/Users/jongbeomkim/Documents/vqvae/pixelcnn_fashion_mnist/epoch=4-val_loss=1.342.pth"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn_fashion_mnist"\
    --batch_size=100\
    --temp=1\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
