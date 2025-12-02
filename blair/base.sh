#!/bin/bash

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# 2 * A100 (80G)
    # --per_device_train_batch_size 384 \
    # 48
    # --eval_steps 1000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=8 \
--master_port $PORT_ID \
train.py \
    --model_name_or_path roberta-base \
    --train_file clean_review_meta.tsv \
    --output_dir checkpoints/blair-roberta-base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 48 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --eval_strategy steps \
    --metric_for_best_model cl_loss \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_mlm \
    --fp16 \
    "$@"
