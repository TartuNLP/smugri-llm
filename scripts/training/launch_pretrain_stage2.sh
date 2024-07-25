#!/bin/bash -e

run_name=# Run name
export WANDB_NAME=${run_name}
export WANDB_PROJECT=#Your project
SCRIPT_DIR=# Path to root directory of this repo

train_paths=# Path to training data
valid_paths=# Path to validation data
stage_1_model_path=# Path to base model to load



accelerate launch --main_process_port 29500 --config_file deepspeed_pretrain_stage2_config.yaml  ${SCRIPT_DIR}/finetune.py \
--model_name ${stage_1_model_path} \
--tokenizer_name "meta-llama/Llama-2-7b-hf" \
--low_cpu_mem_usage \
--train_path "${train_paths}" \
--valid_path "${valid_paths}" \
--wrap_dataset_with_hf \
--disable_train_shuffle True \
--train_dataset_type culturax \
--valid_dataset_type culturax \
--report_to "wandb" \
--force_end_save_and_evaluate \
--seed 42 \
--max_seq_len 2048 \
--num_train_epochs 1 \
--save_final_model \
--eval_steps 0.25 \
--save_steps 0.25 \
--evaluation_strategy "steps" \
--save_strategy "steps" \
--output_dir "checkpoints/${run_name}" \
--logging_steps 1 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--learning_rate 2e-5  \
--lr_scheduler_type "cosine" \
--scheduler_lr_end 2e-6 \
--warmup_ratio 0.01 \
--disable_padding \
--bf16 True \
--weight_decay 0.05 \
--torch_dtype "bfloat16" \
--use_flash_attention_2
