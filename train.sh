#! ~/bin/bash
source /mnt/cephfs/home/areebol/miniconda3/bin/activate vit
MODEL_NAME="shuffle_vit"
SHUFFLE_NUM=1
SHUFFLE_CHANNEL=64
GPU=5
TRAIN_FILE=main_ece.py
echo "model: $MODEL_NAME, args: $SHUFFLE_NUM shuffle-channel $SHUFFLE_CHANNEL in GPU $GPU" 
python $TRAIN_FILE --model-name $MODEL_NAME --shuffle-num $SHUFFLE_NUM --shuffle-channel $SHUFFLE_CHANNEL --gpu $GPU 
