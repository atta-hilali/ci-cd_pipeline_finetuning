from dataclasses import dataclass
from typing import Dict, Any
from datasets import Dataset

@dataclass
class Template:
    version: str = "v1"

    def format_example(self, prompt: str, completion: str) -> str:
        # Simple SFT format; you can swap to ChatML later and bump template_version
        return f"{prompt}\n\nAssistant: {completion}"

def tokenize_sft(
    ds: Dataset,
    tokenizer,
    max_seq_len: int,
    truncation: str = "right",
    template_version: str = "v1",
) -> Dataset:
    template = Template(version=template_version)
    tokenizer.truncation_side = truncation

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        text = template.format_example(ex["prompt"], ex["completion"])
        toks = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    return ds.map(_map, remove_columns=ds.column_names)

def tokenize_dpo(
    ds: Dataset,
    tokenizer,
    max_seq_len: int,
    truncation: str = "right",
) -> Dataset:
    # Expect columns: prompt, chosen, rejected (we’ll build this dataset for DPO)
    tokenizer.truncation_side = truncation

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        # TRL DPOTrainer expects raw strings fields; no need to tokenize here.
        # Keep for compatibility if you later want pre-tokenization.
        return ex

    return ds.map(_map)