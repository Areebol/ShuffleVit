#! ~/bin/bash
source /mnt/cephfs/home/areebol/miniconda3/bin/activate vit
MODEL_NAME="tutel_vit_moe"
GPU=1
TRAIN_FILE=../../main_ece.py
echo "model: $MODEL_NAME, args: in GPU $GPU" 
python $TRAIN_FILE --model-name $MODEL_NAME  --gpu $GPU --num-experts 16 --top-k 2
