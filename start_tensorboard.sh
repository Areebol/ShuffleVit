#!/bin/bash

# Start a tensorboard connection
source /mnt/cephfs/home/areebol/miniconda3/bin/activate vit
LOG=./
PORT=1234
tensorboard --logdir $LOG --port $PORT
