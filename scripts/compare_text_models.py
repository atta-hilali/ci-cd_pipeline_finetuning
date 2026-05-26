import argparse
import gc
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel


DEFAULT_PROMPTS = [
    "What are common causes of tooth sensitivity?",
    "A patient has bleeding gums while brushing. What should they do?",
    "Explain the difference between gingivitis and periodontitis.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a PEFT fine-tuned model with another text model.")
    parser.add_argument("--ft-base-model", required=True, help="Base model id/path used for the adapter.")
    parser.add_argument("--ft-adapter-dir", required=True, help="LoRA/DoRA adapter checkpoint directory.")
    parser.add_argument("--baseline-model", required=True, help="Baseline model id/path, for example a local MedGemma path.")
    parser.add_argument("--prompt-file", help="Optional txt/jsonl file with prompts.")
    parser.add_argument("--output-jsonl", default="comparison_outputs.jsonl", help="Where to save generations.")
    parser.add_argument("--max-new-tokens", type=int, default=489)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def read_prompts(prompt_file):
    if not prompt_file:
        return DEFAULT_PROMPTS

    path = Path(prompt_file)
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                obj = json.loads(line)
                prompts.append(obj.get("prompt") or obj.get("question") or line)
            else:
                prompts.append(line)
    return prompts


def prompt_for_sft(text):
    if "Assistant:" in text:
        return text
    return f"{text}\n\nAssistant:"


def load_causal_model(model_id, dtype_name, trust_remote_code):
    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype(dtype_name),
        device_map="auto",
        token=token,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_batch(label, tokenizer, model, prompts, max_new_tokens, temperature):
    rows = []
    for prompt in prompts:
        text = prompt_for_sft(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded[len(text):].strip() if decoded.startswith(text) else decoded.strip()
        rows.append({"model": label, "prompt": prompt, "response": response})
    return rows


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    output_path = Path(args.output_jsonl)

    print("Loading fine-tuned adapter model...")
    ft_tokenizer, ft_base = load_causal_model(args.ft_base_model, args.dtype, args.trust_remote_code)
    ft_model = PeftModel.from_pretrained(ft_base, args.ft_adapter_dir)
    ft_rows = generate_batch(
        "fine_tuned",
        ft_tokenizer,
        ft_model,
        prompts,
        args.max_new_tokens,
        args.temperature,
    )
    unload(ft_model)
    unload(ft_base)

    print("Loading baseline model...")
    baseline_tokenizer, baseline_model = load_causal_model(args.baseline_model, args.dtype, args.trust_remote_code)
    baseline_rows = generate_batch(
        "baseline",
        baseline_tokenizer,
        baseline_model,
        prompts,
        args.max_new_tokens,
        args.temperature,
    )
    unload(baseline_model)

    with output_path.open("w", encoding="utf-8") as handle:
        for ft_row, baseline_row in zip(ft_rows, baseline_rows):
            handle.write(json.dumps(ft_row, ensure_ascii=False) + "\n")
            handle.write(json.dumps(baseline_row, ensure_ascii=False) + "\n")

    for idx, prompt in enumerate(prompts, start=1):
        ft_response = ft_rows[idx - 1]["response"]
        baseline_response = baseline_rows[idx - 1]["response"]
        print(f"\n=== Prompt {idx} ===")
        print(prompt)
        print("\n--- Fine-tuned ---")
        print(ft_response)
        print("\n--- Baseline ---")
        print(baseline_response)

    print(f"\nSaved comparison rows to {output_path}")


if __name__ == "__main__":
    main()
