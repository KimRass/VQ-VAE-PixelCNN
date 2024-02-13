#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

# For Fashion MNIST
python3 sample.py\
    --dataset="fashion_mnist"\
    --model_params="/Users/jongbeomkim/Documents/vqvae/pixelcnn_fashion_mnist/epoch=2-val_loss=1.4125908.pth"\
    --temp=1\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
    # --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    # --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn_fashion_mnist"\

# # For CIFAR10
