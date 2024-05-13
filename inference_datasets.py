import logging

from torch.utils.data import Dataset

from utils import read_json

logger = logging.getLogger(__name__)

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
    def __init__(self, data, prompt_format_path: str = None):
        self.data = data

        self.prompt_format = PROMPT_DICT

        if prompt_format_path is not None:
            self.prompt_format = read_json(prompt_format_path)
            logger.info(f"Prompt format loaded from {prompt_format_path}:\n{self.prompt_format}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        if item.get("input", "") == "":
            prompt = self.prompt_format["prompt_no_input"].format_map(item)
        else:
            prompt = self.prompt_format["prompt_input"].format_map(item)

        return {"prompts": prompt}


class SimpleDataset(Dataset):
    def __init__(self, data, prompt_field: str = "text"):
        self.data = data
        self.prompt_field = prompt_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"prompts": self.data[index][self.prompt_field]}


class ChatDataset(Dataset):
    SYSTEM_PREFIX = "<|system|>\n"
    SYSTEM_SUFFIX = "\n"
    ASSISTANT_PREFIX = "<|assistant|>\n"
    ASSISTANT_SUFFIX = "\n"
    USER_PREFIX = "<|user|>\n"
    USER_SUFFIX = "\n"

    def __init__(self, data, tokenizer):
        self.dataset = data
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
        messages = self.dataset[index]["messages"]
        if len(messages) == 0:
            raise ValueError('messages field is empty.')

        example_text = self._concat_messages(messages) + self.ASSISTANT_PREFIX
        return {"prompts": example_text}


class EstQADataset(InstructionDataset):
    def __getitem__(self, index):
        item = self.data[index]

        if item.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(item)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(item)

        return {"prompts": prompt, "ids": item["id"]}
