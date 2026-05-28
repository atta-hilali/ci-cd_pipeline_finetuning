import argparse
import inspect
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA/DoRA adapter into its base model.")
    parser.add_argument("--base-model", required=True, help="Base HF model id or local path.")
    parser.add_argument("--adapter-dir", required=True, help="PEFT checkpoint directory with adapter_config.json.")
    parser.add_argument("--output-dir", required=True, help="Directory where the merged standalone model will be saved.")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-safe-merge", action="store_true", help="Disable PEFT safe_merge checks when supported.")
    return parser.parse_args()


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def assert_adapter_dir(adapter_dir):
    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    if not any((adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Missing adapter weights in {adapter_path}")


def main():
    args = parse_args()
    assert_adapter_dir(args.adapter_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=token,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading base model from {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        token=token,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)

    print("Merging adapter into base model")
    merge_kwargs = {}
    if "safe_merge" in inspect.signature(model.merge_and_unload).parameters:
        merge_kwargs["safe_merge"] = not args.no_safe_merge
    merged_model = model.merge_and_unload(**merge_kwargs)

    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
