#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_vqvae.py\
    --dataset="cifar10"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/vqvae-cifar10/"\
    --n_epochs=300\
    --batch_size=128\
    --n_cpus=7\
    --lr=0.0002\
    --n_embeds=32\
    --hidden_dim=128\
    --n_pixelcnn_res_blocks=3\
    --n_pixelcnn_conv_blocks=2\
