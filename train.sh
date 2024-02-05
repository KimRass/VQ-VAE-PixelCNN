#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

# python3 train.py\
#     --dataset="fashion_mnist"\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/"\
#     --save_dir="/Users/jongbeomkim/Documents/vqvae/fashion_mnist/embed_dim=100"\
#     --n_cpus=7\
#     --embed_dim=100\

python3 train.py\
    --dataset="cifar10"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/cifar10"\
    --n_cpus=7\
    --n_embeds=512\
    --hidden_dim=256\
    --n_epochs=2000\
    --batch_size=128\
