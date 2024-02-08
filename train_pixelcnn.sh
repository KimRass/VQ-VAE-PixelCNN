#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

# For Fashion MNIST
python3 train_pixelcnn.py\
    --dataset="fashion_mnist"\
    --vqvae_params="/Users/jongbeomkim/Downloads/vqvae_fashion_mnist.pth"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn/fashion_mnist/only_masked_conv"\
    --n_cpus=7\
    --n_embeds=128\
    --hidden_dim=256\
    --n_epochs=5\
    --batch_size=128\

# # For CIFAR10
# python3 train_pixelcnn.py\
#     --dataset="cifar10"\
#     --vqvae_params="/Users/jongbeomkim/Downloads/vqvae_cifar10.pth"\10.pth"\
#     --data_dir="/Users/jongbeomkim/Documents/datasets/cifar-10-batches-py"\
#     --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn/cifar10/"\
#     --n_cpus=7\
#     --n_embeds=128\
#     --hidden_dim=256\
#     --n_epochs=50\
#     --batch_size=128\

# [ 77,  38,  38,  41,  41,  41,  77],
# [ 77,  38,  38,  15,  82,  15,  26],
# [ 77,  38,  38,  16,  31,  51, 117],
# [ 77,  38,  38,  15,  51,  51, 117],
# [ 41,  28,  62,  62,  51,  51, 117],
# [ 62,  51,  51,  51,  51,  51,  51],
# [ 77,  99,  41,  41,  41,  41,  41]