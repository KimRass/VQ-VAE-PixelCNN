#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train_pixelcnn.py\
    --dataset="fashion_mnist"\
    --vqvae_params="/Users/jongbeomkim/Documents/vqvae/vqvae_fashion_mnist/epoch=47-val_loss=0.145.pth"\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/pixelcnn_fashion_mnist"\
    --n_cpus=7\
    --n_epochs=20\
    --batch_size=128\
    --n_embeds=128\
    --hidden_dim=256\
    --n_pixelcnn_res_blocks=2\
    --n_pixelcnn_conv_blocks=2\
