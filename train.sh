#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

# # For Fashion MNIST
# python3 train.py\
#     --dataset="fashion_mnist"\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/"\
#     --save_dir="/Users/jongbeomkim/Documents/vqvae/fashion_mnist/experiment"\
#     --n_cpus=7\
#     --n_embeds=128\
#     --hidden_dim=256\
#     --n_epochs=50\
#     --batch_size=128\

# For CIFAR10
python3 train.py\
    --dataset="cifar10"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/cifar10/"\
    --n_cpus=7\
    --n_embeds=128\
    --hidden_dim=256\
    --n_epochs=2000\
    --batch_size=128\
    # --resume_from="/Users/jongbeomkim/Documents/vqvae/cifar10/model6/epoch=10-train_loss=0.121-val_loss=0.120.pth"\
