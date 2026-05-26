import argparse
import gc
import json
import os
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "What are common causes of tooth sensitivity?",
    "A patient has bleeding gums while brushing. What should they do?",
    "Explain the difference between gingivitis and periodontitis.",
    "What warning signs mean a dental patient should seek urgent care?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a local PEFT adapter against an OpenAI-style server.")
    parser.add_argument("--ft-base-model", required=True)
    parser.add_argument("--ft-adapter-dir", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--server-model", default="medgemma")
    parser.add_argument("--prompt-file", help="Optional txt/jsonl file containing prompts.")
    parser.add_argument("--output-jsonl", default="ft_vs_server_comparison.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=250)
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


def load_ft_model(base_model, adapter_dir, dtype_name, trust_remote_code):
    token = os.getenv("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype(dtype_name),
        device_map="auto",
        token=token,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def generate_ft(tokenizer, model, prompt, max_new_tokens, temperature):
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
    return decoded[len(text):].strip() if decoded.startswith(text) else decoded.strip()


def query_openai_server(server_url, server_model, prompt, max_tokens, temperature):
    payload = {
        "model": server_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    text_parts = []
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8", errors="replace")

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "token":
            text_parts.append(obj.get("text", ""))
            continue

        choices = obj.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            delta = choices[0].get("delta") or {}
            text_parts.append(message.get("content") or delta.get("content") or choices[0].get("text") or "")

    return "".join(text_parts).strip()


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file)

    print("Loading fine-tuned adapter model...")
    tokenizer, ft_model = load_ft_model(
        args.ft_base_model,
        args.ft_adapter_dir,
        args.dtype,
        args.trust_remote_code,
    )

    rows = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"Running prompt {index}/{len(prompts)}...")
        ft_response = generate_ft(tokenizer, ft_model, prompt, args.max_new_tokens, args.temperature)
        server_response = query_openai_server(
            args.server_url,
            args.server_model,
            prompt,
            args.max_new_tokens,
            args.temperature,
        )
        rows.append(
            {
                "prompt": prompt,
                "fine_tuned": ft_response,
                "server_model": args.server_model,
                "server_response": server_response,
            }
        )

    del ft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path = Path(args.output_jsonl)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for index, row in enumerate(rows, start=1):
        print(f"\n=== Prompt {index} ===")
        print(row["prompt"])
        print("\n--- Fine-tuned Qwen adapter ---")
        print(row["fine_tuned"])
        print(f"\n--- {args.server_model} server ---")
        print(row["server_response"])

    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
