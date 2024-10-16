"""
Adapted from https://github.com/sambanova/lm-evaluation-harness
"""
import yaml
from tqdm import tqdm



if __name__ == "__main__":
    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = "_default_template_yaml"
    with open(base_yaml_name, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    passage_langs = ["vro_Latn", "kpv_Cyrl", "liv_Latn"]

    for lang in tqdm(passage_langs):
        yaml_dict = {
            "include": base_yaml_name,
            "task": f"belebele_{lang}",
            "test_split": "test",
            "fewshot_split": "test",
            "dataset_name": lang,
        }

        file_save_path = f"belebele_{lang}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
