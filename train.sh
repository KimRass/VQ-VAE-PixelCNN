#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

python3 train.py\
    --data_dir="/Users/jongbeomkim/Documents/datasets/"\
    --save_dir="/Users/jongbeomkim/Documents/vqvae/fashion_mnist"\
    --n_cpus=7\
    # --resume_from="/Users/jongbeomkim/Documents/pixelcnn/epoch=20-train_loss=4.984.pth"\
