import argparse
import gc
import json
import os
import statistics
import time
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
    parser.add_argument("--ft-base-model")
    parser.add_argument("--ft-adapter-dir")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--server-model", default="medgemma")
    parser.add_argument("--prompt-file", help="Optional txt/jsonl file containing prompts.")
    parser.add_argument("--run", choices=("both", "ft", "server"), default="both")
    parser.add_argument("--output-jsonl", default="ft_vs_server_comparison.jsonl")
    parser.add_argument("--summary-json", default="ft_vs_server_summary.json")
    parser.add_argument("--edge-report-md", default="edge_readiness_report.md")
    parser.add_argument("--edge-vram-gb", default="8,12,16,24,32")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.run in {"both", "ft"} and (not args.ft_base_model or not args.ft_adapter_dir):
        parser.error("--ft-base-model and --ft-adapter-dir are required when --run is 'both' or 'ft'.")
    return args


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


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def describe_model(model):
    total_params = 0
    trainable_params = 0
    param_bytes = 0
    for param in model.parameters():
        total_params += param.numel()
        param_bytes += param.numel() * param.element_size()
        if param.requires_grad:
            trainable_params += param.numel()

    return {
        "parameters_total_b": round(total_params / 1_000_000_000, 3),
        "parameters_trainable_m": round(trainable_params / 1_000_000, 3),
        "parameter_memory_gb": round(param_bytes / (1024**3), 3),
    }


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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync_cuda()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_cuda()
    elapsed_sec = time.perf_counter() - start

    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(output.shape[-1])
    generated_tokens = max(output_tokens - input_tokens, 0)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded[len(text):].strip() if decoded.startswith(text) else decoded.strip()
    metrics = {
        "latency_sec": round(elapsed_sec, 3),
        "generated_tokens": generated_tokens,
        "tokens_per_sec": round(generated_tokens / elapsed_sec, 3) if elapsed_sec > 0 else None,
    }
    if torch.cuda.is_available():
        metrics.update(
            {
                "cuda_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
                "cuda_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
            }
        )
    return response, metrics


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
    done_metrics = {}
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read().decode("utf-8", errors="replace")
    wall_sec = time.perf_counter() - start

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
        if obj.get("type") == "done":
            done_metrics = {
                "server_reported_tokens_generated": obj.get("tokens_generated"),
                "server_reported_elapsed_sec": obj.get("elapsed_sec"),
                "server_reported_tokens_per_sec": obj.get("tokens_per_sec"),
                "server_reported_vram_used_gb": obj.get("vram_used_gb"),
            }
            continue

        choices = obj.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            delta = choices[0].get("delta") or {}
            text_parts.append(message.get("content") or delta.get("content") or choices[0].get("text") or "")

    metrics = {"wall_latency_sec": round(wall_sec, 3)}
    metrics.update({key: value for key, value in done_metrics.items() if value is not None})
    return "".join(text_parts).strip(), metrics


def mean_numeric(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(statistics.mean(values), 3) if values else None


def max_numeric(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(max(values), 3) if values else None


def build_summary(args, rows, ft_model_stats):
    return {
        "fine_tuned": {
            "base_model": args.ft_base_model,
            "adapter_dir": args.ft_adapter_dir,
            "model_stats": ft_model_stats,
            "avg_latency_sec": mean_numeric(rows, ("fine_tuned_metrics", "latency_sec")),
            "avg_tokens_per_sec": mean_numeric(rows, ("fine_tuned_metrics", "tokens_per_sec")),
            "max_cuda_peak_reserved_gb": max_numeric(rows, ("fine_tuned_metrics", "cuda_peak_reserved_gb")),
            "max_cuda_peak_allocated_gb": max_numeric(rows, ("fine_tuned_metrics", "cuda_peak_allocated_gb")),
        },
        "server": {
            "model": args.server_model,
            "url": args.server_url,
            "avg_wall_latency_sec": mean_numeric(rows, ("server_metrics", "wall_latency_sec")),
            "avg_reported_tokens_per_sec": mean_numeric(rows, ("server_metrics", "server_reported_tokens_per_sec")),
            "max_reported_vram_used_gb": max_numeric(rows, ("server_metrics", "server_reported_vram_used_gb")),
        },
    }


def fit_label(required_gb, device_gb):
    if required_gb is None:
        return "unknown"
    if required_gb <= device_gb * 0.85:
        return "fits"
    if required_gb <= device_gb:
        return "tight"
    if required_gb * 0.35 <= device_gb:
        return "quantization likely needed"
    return "too large"


def write_edge_report(path, summary, edge_vram_gb):
    ft_required = summary["fine_tuned"].get("max_cuda_peak_reserved_gb")
    if ft_required is None:
        ft_required = summary["fine_tuned"].get("model_stats", {}).get("parameter_memory_gb")
    server_required = summary["server"].get("max_reported_vram_used_gb")

    lines = [
        "# Edge Readiness Report",
        "",
        "Measured on the current DGX environment. Edge performance must still be validated on the target device.",
        "",
        "## Summary",
        "",
        f"- Fine-tuned adapter base: `{summary['fine_tuned']['base_model']}`",
        f"- Fine-tuned average tokens/sec: `{summary['fine_tuned']['avg_tokens_per_sec']}`",
        f"- Fine-tuned peak reserved VRAM GB: `{ft_required}`",
        f"- Server model: `{summary['server']['model']}`",
        f"- Server average reported tokens/sec: `{summary['server']['avg_reported_tokens_per_sec']}`",
        f"- Server reported VRAM GB: `{server_required}`",
        "",
        "## Edge Fit Estimate",
        "",
        "| Edge VRAM/RAM GB | Fine-tuned Qwen adapter | MedGemma server |",
        "| ---: | --- | --- |",
    ]

    for device_gb in edge_vram_gb:
        lines.append(
            f"| {device_gb:g} | {fit_label(ft_required, device_gb)} | {fit_label(server_required, device_gb)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- A PEFT adapter does not make the base model smaller at inference time; Qwen3-8B still needs the Qwen3-8B base plus the adapter.",
            "- If a model is marked `quantization likely needed`, test 4-bit or 8-bit inference before deciding it can run on edge hardware.",
            "- Server timings include HTTP and streaming overhead. Fine-tuned timings are direct local generation timings.",
            "- For fair GPU numbers, benchmark one model at a time. If the MedGemma Docker server is running while Qwen is loaded, both can compete for VRAM and compute.",
            "",
        ]
    )

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    run_ft = args.run in {"both", "ft"}
    run_server = args.run in {"both", "server"}

    tokenizer = None
    ft_model = None
    ft_model_stats = {}
    if run_ft:
        print("Loading fine-tuned adapter model...")
        tokenizer, ft_model = load_ft_model(
            args.ft_base_model,
            args.ft_adapter_dir,
            args.dtype,
            args.trust_remote_code,
        )
        ft_model_stats = describe_model(ft_model)

    rows = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"Running prompt {index}/{len(prompts)}...")
        row = {"prompt": prompt}
        if run_ft:
            ft_response, ft_metrics = generate_ft(tokenizer, ft_model, prompt, args.max_new_tokens, args.temperature)
            row["fine_tuned"] = ft_response
            row["fine_tuned_metrics"] = ft_metrics
        if run_server:
            server_response, server_metrics = query_openai_server(
                args.server_url,
                args.server_model,
                prompt,
                args.max_new_tokens,
                args.temperature,
            )
            row["server_model"] = args.server_model
            row["server_response"] = server_response
            row["server_metrics"] = server_metrics
        rows.append(row)

    if ft_model is not None:
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
        if "fine_tuned" in row:
            print("\n--- Fine-tuned Qwen adapter ---")
            print(row["fine_tuned"])
            print(f"\nMetrics: {row['fine_tuned_metrics']}")
        if "server_response" in row:
            print(f"\n--- {args.server_model} server ---")
            print(row["server_response"])
            print(f"\nMetrics: {row['server_metrics']}")

    print(f"\nSaved comparison to {output_path}")
    summary = build_summary(args, rows, ft_model_stats)
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved benchmark summary to {args.summary_json}")
    if args.edge_report_md:
        edge_vram_gb = [float(item.strip()) for item in args.edge_vram_gb.split(",") if item.strip()]
        write_edge_report(args.edge_report_md, summary, edge_vram_gb)
        print(f"Saved edge readiness report to {args.edge_report_md}")


if __name__ == "__main__":
    main()
