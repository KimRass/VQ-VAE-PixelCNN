#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_vqvae.py\
    --dataset="cifar10"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/cifar10/"\
    --n_cpus=7\
    --n_epochs=300\
    --batch_size=128\
    --n_embeds=128\
    --hidden_dim=256\
    --n_pixelcnn_res_blocks=2\
