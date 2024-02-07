#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

# For Fashion MNIST
python3 train_pixelcnn.py\
    --dataset="fashion_mnist"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae-pixelcnn/fashion_mnist/"\
    --n_cpus=7\
    --n_embeds=128\
    --hidden_dim=256\
    --n_epochs=50\
    --batch_size=128\
