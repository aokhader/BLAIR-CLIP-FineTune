#!/bin/bash

# activate fv

# Randomly set a port number to avoid clashes between concurrent runs.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple host threads for data loading.
export OMP_NUM_THREADS=8

# Launch distributed multimodal pre-training (8-GPU default; adjust as needed).
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_port $PORT_ID \
    train.py \
    --model_family blair_clip \
    --model_name_or_path roberta-base \
    --mm_clip_model_name openai/clip-vit-base-patch16 \
    --mm_projection_dim 512 \
    --mm_text_text_weight 0.5 \
    --train_file clean_review_meta_with_images.tsv \
    --image_column image_path \
    --image_root ./blair_clip_images \
    --output_dir checkpoints/blair-clip-base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --learning_rate 3e-5 \
    --max_seq_length 64 \
    --eval_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --eval_steps 100 \
    --metric_for_best_model cl_loss \
    --load_best_model_at_end \
    --pooler_type cls \
    --temp 0.05 \
    --mm_temperature_init 0.07 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_mlm \
    --fp16 \
    "$@"

