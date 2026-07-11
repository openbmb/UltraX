#!/bin/bash
# UltraX: Train the refinement model (Qwen3-0.6B full-parameter SFT)
# Framework: ms-swift with DeepSpeed ZeRO3
# Hardware: 8x GPU

# Activate your conda environment and navigate to ms-swift directory
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env
# cd /path/to/ms-swift

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model Qwen/Qwen3-0.6B \
    --train_type full \
    --loss_scale ignore_empty_think \
    --dataset /path/to/train_data_with_instruction \
    --torch_dtype bfloat16 \
    --max_steps 5530 \
    --streaming true \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 3 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 3e-6}' \
    --packing false \
    --max_length 20480 \
    --attn_impl flash_attn \
    --save_steps 2765 \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir /path/to/output_model \
    --deepspeed zero3 \
    --use_liger_kernel true
