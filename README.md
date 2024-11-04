# LLMs for Extremely Low-Resource Finno-Ugric Languages
This repository contains the implementation used for training and evaluating language models for extremely low-resource Finno-Ugric languages.

## Models
**Pre-trained**:
* [tartuNLP/Llama-SMUGRI-7B](https://huggingface.co/tartuNLP/Llama-SMUGRI-7B)

**Instruction-tuned**:
* [tartuNLP/Llama-SMUGRI-7B-Instruct-MTI](https://huggingface.co/tartuNLP/Llama-SMUGRI-7B-Instruct-MTI) (_SupInst+TrAlpaca_)
* [Llama-SMUGRI-7B-Instruct-MTI-Tr](https://huggingface.co/tartuNLP/Llama-SMUGRI-7B-Instruct-MTI-Tr) (_SupInst+TrAlpaca+TrInst_)
* [tartuNLP/Llama-SMUGRI-7B-Instruct-LLMTI](https://huggingface.co/tartuNLP/Llama-SMUGRI-7B-Instruct-LLMTI) (_SupInst+LLMTrAlpaca_)
* [tartuNLP/Llama-SMUGRI-7B-Instruct-LLMTI-Tr](https://huggingface.co/tartuNLP/Llama-SMUGRI-7B-Instruct-LLMTI-Tr) (_SupInst+LLMTrAlpaca+TrInst_)

## Evaluation
Belebele-SMUGRI:
* https://huggingface.co/datasets/tartuNLP/belebele-smugri

SIB-SMUGRI:
* https://huggingface.co/datasets/tartuNLP/sib-smugri

## Usage
Scripts for launching training are provided in:
* [scripts/training](scripts/training)

LM-eval-harness configurations:
* [scripts/evaluation/lm_eval_harness_configs](scripts/evaluation/lm_eval_harness_configs)

## Citation
```
@misc{purason2024llmsextremelylowresourcefinnougric,
      title={LLMs for Extremely Low-Resource Finno-Ugric Languages}, 
      author={Taido Purason and Hele-Andra Kuulmets and Mark Fishel},
      year={2024},
      eprint={2410.18902},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.18902}, 
}
```

## Acknowledgements
The implementation is built on [github.com/TartuNLP/llammas](https://github.com/TartuNLP/llammas).

