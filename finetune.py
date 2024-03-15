# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/train.py
# and https://github.com/facebookresearch/llama-recipes
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple

import torch
from datasets import interleave_datasets, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizer, default_data_collator, DataCollatorForSeq2Seq, get_polynomial_decay_schedule_with_warmup, \
    PreTrainedModel

from torch.utils.data import Dataset
from transformers.utils import PaddingStrategy

from training_datasets import InstructionDataset, ConstantLengthDataset, ChatDataset


@dataclass
class ScriptArguments:
    train_path: str = field(metadata={"help": "Training data path"})
    valid_path: Optional[str] = field(metadata={"help": "Validation data path"}, default=None)
    train_dataset_type: str = field(
        default="alpaca",
        metadata={"help": "Training dataset type"}
    )
    valid_dataset_type: str = field(
        default="alpaca",
        metadata={"help": "Validation dataset type"}
    )
    alpaca_prompt_format_path: Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    use_dynamic_padding: bool = field(default=False)
    disable_padding: bool = field(default=False)
    use_new_pad_token: bool = field(default=False)
    interleave_probs: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)
    use_flash_attention_2: bool = field(default=False)
    save_final_model: bool = field(default=False)


@dataclass
class LoraArguments:
    peft_name: Optional[str] = field(
        default=None,
        metadata={"help": "The pre-trained PEFT model name"}
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )


@dataclass
class QuantizationArguments:
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )


# https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/utils.py#L116C1-L125C43
def get_chars_per_token(dataset: Dataset, tokenizer: PreTrainedTokenizer, data_column: str, nb_examples: int = 500):
    """
    Estimate the average number of characters per token in the dataset.
    """
    logging.info("Estimating average number of characters per token in the dataset...")
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer.encode(example[data_column], truncation=False, padding=False))

    return total_characters / total_tokens


def create_dataset(tokenizer: PreTrainedTokenizer, args: ScriptArguments, dataset_type: str, path: str, seed: int):
    if dataset_type == "alpaca":
        logging.info(f"Loading dataset from: {path}")
        dataset = InstructionDataset(
            data_path=path,
            tokenizer=tokenizer,
            padding=not args.use_dynamic_padding and not args.disable_padding,
            prompt_format_path=args.alpaca_prompt_format_path,
        )
    elif dataset_type == "culturax":
        dataset_paths = list(path.split(","))
        logging.info(f"Loading datasets from: {dataset_paths}")

        culturax_datasets = []
        for ds_path in dataset_paths:
            logging.info(f"Loading {ds_path}")
            if ":" in ds_path:
                ds_name, lang = ds_path.split(":")
                culturax_dataset = load_dataset(ds_name, lang, streaming=True, split="train")
            else:
                culturax_dataset = load_dataset(ds_path, streaming=True, split="train")
            culturax_datasets.append(culturax_dataset)

        if len(culturax_datasets) == 0:
            raise ValueError("No datasets found")
        elif len(culturax_datasets) == 1:
            logging.info("Loaded 1 dataset")
            raw_dataset = culturax_datasets[0]
        else:
            logging.info(f"Loaded {len(culturax_datasets)} culturax datasets")
            if args.interleave_probs:
                interleave_probs = [float(p) for p in args.interleave_probs.split(",")]
            else:
                interleave_probs = [1 / len(culturax_datasets)] * len(culturax_datasets)

            logging.info(f"Interleave probabilities: {interleave_probs}")

            raw_dataset = interleave_datasets(
                culturax_datasets,
                probabilities=interleave_probs,
                seed=seed,
                stopping_strategy="all_exhausted"
            )

        chars_per_token = get_chars_per_token(raw_dataset, tokenizer, "text")
        logging.info(f"Chars per token: {chars_per_token}")
        dataset = ConstantLengthDataset(
            tokenizer,
            raw_dataset,
            infinite=True,
            seq_length=args.max_seq_length,
            chars_per_token=chars_per_token,
            dataset_text_field="text",
            shuffle=True,
        )
    elif dataset_type == "chat":
        if not args.use_dynamic_padding and not args.disable_padding:
            raise ValueError(
                "Constant padding to max model length is not implemented for chat datasets, use --use_dynamic_padding."
            )
        dataset = ChatDataset(
            data_path=path,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset


def create_datasets(tokenizer: PreTrainedTokenizer, args: ScriptArguments):
    train_dataset = create_dataset(tokenizer, args, args.train_dataset_type, args.train_path, seed=training_args.seed)
    if args.valid_path is None:
        return train_dataset, None

    valid_dataset = create_dataset(tokenizer, args, args.valid_dataset_type, args.valid_path, seed=training_args.seed)
    return train_dataset, valid_dataset


def create_and_prepare_model(args: ScriptArguments, training_args: TrainingArguments, lora_args: LoraArguments,
                             quant_args: QuantizationArguments) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_map = None
    bnb_config = None
    load_in_8bit = quant_args.use_8bit_quantization

    if quant_args.use_4bit_quantization:
        logging.info("Using 4bit quantization")
        compute_dtype = getattr(torch, quant_args.bnb_4bit_compute_dtype)

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_args.use_4bit_quantization,
            bnb_4bit_quant_type=quant_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and quant_args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if quant_args.use_4bit_quantization or quant_args.use_8bit_quantization:
        device_map = "auto"

    if args.torch_dtype is None or args.torch_dtype == "auto":
        torch_dtype = args.torch_dtype
    else:
        torch_dtype = getattr(torch, args.torch_dtype)

    model_kwargs = {}

    if args.use_flash_attention_2:
        logging.info("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not training_args.gradient_checkpointing,
        trust_remote_code=True,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        **model_kwargs,
    )

    if lora_args.use_peft_lora or lora_args.peft_name is not None:
        logging.info("Using PEFT LoRA")
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if lora_args.peft_name is not None:
            logging.info(f"Loading pre-trained LoRA weights and congfig from {lora_args.peft_name}")
            model = PeftModel.from_pretrained(model, lora_args.peft_name, is_trainable=True)
        else:
            peft_config = LoraConfig(
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                r=lora_args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_args.lora_target_modules.split(","),
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if args.tokenizer_name is None else args.tokenizer_name,
        model_max_length=args.max_seq_length,
        padding_side="right",
    )
    if tokenizer.pad_token_id is not None:
        logging.info(f"Tokenizer pad_token_id={tokenizer.pad_token_id}")
        return model, tokenizer

    if args.use_new_pad_token:
        logging.info("No pad token found, adding <pad> to vocabulary")
        tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.padding_idx = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
    else:
        logging.info("No pad token found, using pad_token_id=0")
        tokenizer.pad_token_id = 0

    return model, tokenizer


def peft_module_casting_to_bf16(model, args: TrainingArguments):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    scheduler_lr_end: float = None


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, lr_init: float = 1,
        lr_end: float = 0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    relative_lr_end = lr_end / lr_init
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (
            1 - relative_lr_end) + relative_lr_end


def get_cosine_schedule_with_warmup_end_lr(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        lr_end: float = 0,
):
    lr_init = optimizer.defaults["lr"]
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        lr_init=lr_init,
        lr_end=lr_end,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if (
                self.lr_scheduler is None and
                isinstance(self.args, CustomTrainingArguments) and
                self.args.scheduler_lr_end is not None
        ):
            logging.info(
                f"Using {self.args.lr_scheduler_type} with learning rate with end lr {self.args.scheduler_lr_end}"
            )
            if self.args.lr_scheduler_type == "polynomial":
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            elif self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup_end_lr(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    lr_end=self.args.scheduler_lr_end,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            else:
                raise ValueError(f"lr scheduler {self.args.lr_scheduler_type} not supported with scheduler_lr_end")
            self._created_lr_scheduler = True
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)


def main(script_args: ScriptArguments, training_args: TrainingArguments, quantization_args: QuantizationArguments,
         lora_args: LoraArguments):
    torch.cuda.manual_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer = create_and_prepare_model(
        script_args, training_args, lora_args, quantization_args
    )
    model.config.use_cache = False

    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    if args.use_dynamic_padding:
        logging.info("Using dynamic padding")
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=PaddingStrategy.LONGEST,
            max_length=args.max_seq_length
        )
    else:
        logging.info("Using max length padding to max_seq_length")
        collator = default_data_collator

    try:
        logging.info(f"Train dataset with {len(train_dataset)} examples")
    except:
        logging.info(f"Train dataset with unknown number of examples")

    if eval_dataset is not None:
        logging.info(f"Validation dataset with {len(eval_dataset)} examples")
    else:
        logging.info(f"No validation dataset")

    logging.info(f"Max sequence length: {tokenizer.model_max_length}")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if lora_args.use_peft_lora:
        trainer.model.print_trainable_parameters()
        peft_module_casting_to_bf16(trainer.model, training_args)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if script_args.save_final_model:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )

    parser = HfArgumentParser([ScriptArguments, CustomTrainingArguments, QuantizationArguments, LoraArguments])
    args, training_args, quant_args, lora_args = parser.parse_args_into_dataclasses()
    main(
        script_args=args,
        training_args=training_args,
        quantization_args=quant_args,
        lora_args=lora_args,
    )
