import os
from pathlib import Path
from typing import List
import json


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip() for line in f]


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def create_instructions(in_path_src, in_path_tgt, out_path, prompt, no_input=False, n_examples=None):
    src_lines = read_lines(in_path_src)
    if in_path_tgt is None:
        tgt_lines = [""] * len(src_lines)
    else:
        tgt_lines = read_lines(in_path_tgt)

    assert len(src_lines) == len(tgt_lines)
    if n_examples is not None:
        src_lines = src_lines[:n_examples]
        tgt_lines = tgt_lines[:n_examples]

    if no_input:
        instructions = [
            {
                "instruction": prompt.format(input=src_sentence),
                "input": "",
                "output": tgt_sentence
            }
            for src_sentence, tgt_sentence in zip(src_lines, tgt_lines)
        ]
    else:
        instructions = [
            {
                "instruction": prompt.format(src_sentence=src_sentence, tgt_sentence=tgt_sentence),
                "input": src_sentence,
                "output": tgt_sentence
            }
            for src_sentence, tgt_sentence in zip(src_lines, tgt_lines)
        ]
    print(f"Writing {len(instructions)} instructions to {out_path}")
    write_json(instructions, out_path)


def run():
    in_dir = "UT-L2-training"
    out_dir = "UT-L2-training-instructions"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    create_instructions(
        os.path.join(in_dir, "train.est0_Latn-est_Latn.est0_Latn"),
        os.path.join(in_dir, "train.est0_Latn-est_Latn.est_Latn"),
        os.path.join(out_dir, "instruction.est0_Latn-est_Latn.json"),
        "Reply with a corrected version of the input sentence in Estonian with all grammatical and spelling errors fixed. If there are no errors, reply with a copy of the original sentence."
    )
