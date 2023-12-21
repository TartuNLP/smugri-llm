import copy
import logging
import warnings
import random

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import LlamaTokenizer

from utils import read_json

logger = logging.getLogger(__name__)

class ChatDataset(Dataset):
    SYSTEM_PREFIX = "<|system|>\n"
    SYSTEM_SUFFIX = "\n"
    ASSISTANT_PREFIX = "<|assistant|>\n"
    ASSISTANT_SUFFIX = "\n"
    USER_PREFIX = "<|user|>\n"
    USER_SUFFIX = "\n"

    def __init__(self, data_path: str, tokenizer: LlamaTokenizer):
        self.dataset = read_json(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def _concat_messages(self, messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += self.SYSTEM_PREFIX + message["content"].strip() + self.SYSTEM_SUFFIX
            elif message["role"] == "user":
                message_text += self.USER_PREFIX + message["content"].strip() + self.USER_SUFFIX
            elif message["role"] == "assistant":
                message_text += self.ASSISTANT_PREFIX + message["content"].strip() + self.tokenizer.eos_token \
                                + self.ASSISTANT_SUFFIX
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    def __getitem__(self, index):
        IGNORE_INDEX = -100

        messages = self.dataset[index]["messages"]
        if len(messages) == 0:
            raise ValueError('messages field is empty.')

        example_text = self._concat_messages(messages).strip()
        tokenized_example = self.tokenizer(example_text, return_tensors='pt',
                                           max_length=self.tokenizer.model_max_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = self.tokenizer(
                        self._concat_messages(messages[:message_idx]),
                        return_tensors='pt',
                        max_length=self.tokenizer.model_max_length,
                        truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = self._concat_messages(messages[:message_idx + 1]) + self.ASSISTANT_PREFIX
                else:
                    messages_so_far = self._concat_messages(messages[:message_idx + 1])
                message_end_idx = self.tokenizer(
                    messages_so_far,
                    return_tensors='pt',
                    max_length=self.tokenizer.model_max_length,
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

                if message_end_idx >= self.tokenizer.model_max_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }


# implementation from TRL: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
            self,
            tokenizer,
            dataset,
            dataset_text_field=None,
            formatting_func=None,
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6,
            eos_token_id=0,
            shuffle=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            formatting_func_signature = formatting_func.__code__.co_varnames
            if len(formatting_func_signature) > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: LlamaTokenizer, padding=True, prompt_format_path: str = None):
        self.dataset = read_json(data_path)
        self.padding = padding
        self.tokenizer = tokenizer
        self.prompt_format = PROMPT_DICT

        if prompt_format_path is not None:
            self.prompt_format = read_json(prompt_format_path)
            logger.info(f"Prompt format loaded from {prompt_format_path}:\n{self.prompt_format}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        example_text = self.dataset[index]
        if example_text.get("input", "") == "":
            prompt = self.prompt_format["prompt_no_input"].format_map(example_text)
        else:
            prompt = self.prompt_format["prompt_input"].format_map(example_text)
        example = prompt + example_text["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt, truncation=True), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, truncation=True)
        if len(example) < self.tokenizer.model_max_length:
            example.append(self.tokenizer.eos_token_id)

        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.tokenizer.model_max_length - len(example)
        if padding > 0 and self.padding:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = self.tokenizer.pad_token_id
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }
