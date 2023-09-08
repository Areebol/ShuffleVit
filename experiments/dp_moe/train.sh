#! ~/bin/bash
source /mnt/cephfs/home/areebol/miniconda3/bin/activate vit
MODEL_NAME="moe"
GPU=6
TRAIN_FILE=../../main_ece.py
echo "model: $MODEL_NAME, args: in GPU $GPU" 
python $TRAIN_FILE --model-name $MODEL_NAME  --gpu $GPU