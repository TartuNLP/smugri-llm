run_name=${1}
data_path=${2}
base_model_path=${3}


port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
accelerate launch --main_process_port ${port} --config_file deepspeed_train_config_bf16_instruct.yaml  finetune.py \
--model_name ${base_model_path} \
--tokenizer_name "meta-llama/Llama-2-7b-hf" \
--train_path ${data_path} \
--train_dataset_type alpaca \
--alpaca_prompt_format_path alpaca_prompt_simple.json \
--low_cpu_mem_usage \
--seed 42 \
--max_seq_len 1024 \
--num_train_epochs 3 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--output_dir "checkpoints/${run_name}" \
--logging_steps 10 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--learning_rate 5e-6  \
--lr_scheduler_type "polynomial" \
--scheduler_lr_end 5e-7 \
--bf16 True \
--use_dynamic_padding \
--dataloader_drop_last True \
--dataloader_pin_memory True \
--weight_decay 0.1