#!/bin/bash -e
run_name=# Run name
export WANDB_NAME=${run_name}
export WANDB_PROJECT=#Your project
SCRIPT_DIR=# Path to root directory of this repo

train_paths=# Path to training data
valid_paths=# Path to validation data
base_model_path=# Path to base model to load


accelerate launch --main_process_port 29500 --config_file deepspeed_instruction_tuning_config.yaml  ${SCRIPT_DIR}/finetune.py \
    --model_name "${base_model_path}" \
    --tokenizer_name "meta-llama/Llama-2-7b-hf" \
    --low_cpu_mem_usage \
    --train_path "${train_paths}" \
    --valid_path "${valid_paths}" \
    --train_dataset_type chat \
    --valid_dataset_type chat \
    --report_to "wandb" \
    --seed 42 \
    --max_seq_len 2048 \
    --save_final_model \
    --num_train_epochs 2 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --output_dir "checkpoints/${run_name}" \
    --logging_steps 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --learning_rate 2e-5  \
    --lr_scheduler_type "cosine" \
    --scheduler_lr_end 2e-6 \
    --warmup_ratio 0.01 \
    --bf16 True \
    --use_dynamic_padding \
    --dataloader_drop_last True \
    --dataloader_pin_memory True \
    --weight_decay 0.05 \
    --torch_dtype "bfloat16"