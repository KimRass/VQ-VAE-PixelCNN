#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_pixelcnn.py\
    --dataset="cifar10"\
    --vqvae_params="/Users/jongbeomkim/Downloads/vqvae_cifar10.pth"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn_cifar10"\
    --n_cpus=7\
    --n_epochs=70\
    --batch_size=128\
    --n_embeds=128\
    --hidden_dim=256\
    --n_pixelcnn_res_blocks=3\
    --n_pixelcnn_conv_blocks=2\
