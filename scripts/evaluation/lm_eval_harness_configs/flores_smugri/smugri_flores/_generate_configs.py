"""
Adapted from https://github.com/sambanova/lm-evaluation-harness
"""

import yaml
from tqdm import tqdm

lang_map = {
    "eng_Latn": "English",
    "vro_Latn": "Võro",
    "liv_Latn": "Livonian",
    "kpv_Cyrl": "Komi",
    "est_Latn": "Estonian",
    "fin_Latn": "Finnish",
    "rus_Cyrl": "Russian",
}

if __name__ == "__main__":

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = "_default_template_yaml"
    with open(base_yaml_name, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    src_langs = ["eng_Latn", "est_Latn", "rus_Cyrl"]
    tgt_langs = ["vro_Latn", "liv_Latn", "kpv_Cyrl", "est_Latn", "fin_Latn"]
    bidirectional = True

    dataset_configs = [(src, tgt) for src in src_langs for tgt in tgt_langs if src != tgt]
    if bidirectional:
        dataset_configs += [(tgt, src) for src, tgt in dataset_configs]

    for (src_lang, tgt_lang) in tqdm(dataset_configs):
        # Ignore splits that are not parallel in two languages
        dataset_config = f"{src_lang}-{tgt_lang}"

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"flores_smugri_{dataset_config}",
            "doc_to_decontamination_query": "{{sentence_" + src_lang + "}}\t{{sentence_" + tgt_lang + "}}",
            "dataset_name": f"{dataset_config}",
            "doc_to_text": f"{lang_map[src_lang]}: " + "{{sentence_" + src_lang + "}}\n" + f"{lang_map[tgt_lang]}:",
            "doc_to_target": "{{sentence_" + tgt_lang + "}}"
        }

        file_save_path = f"flores_smugri_{dataset_config}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
