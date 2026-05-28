import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DENTIST_COPILOT_INSTRUCTION = (
    "You are a dentist co-pilot assisting a licensed dentist. "
    "Give concise, clinically useful support for triage, differential considerations, "
    "chairside next steps, patient education, and documentation. "
    "Do not claim to replace professional judgment."
)


DEFAULT_PROMPTS = [
    (
        "A 34-year-old patient reports sharp pain on cold drinks in the upper right molar region. "
        "No swelling, no fever. Help the dentist structure the likely causes, key questions, exam checks, "
        "and initial chairside management."
    ),
    (
        "A patient has bleeding gums while brushing and generalized plaque accumulation. "
        "Prepare a dentist co-pilot response with differential considerations, periodontal assessment steps, "
        "patient education, and when to schedule scaling/root planing."
    ),
    (
        "During a consultation, a patient asks whether their gum problem is gingivitis or periodontitis. "
        "Draft a dentist-facing explanation, what findings distinguish them, and what should be documented."
    ),
    (
        "A patient calls after extraction with increasing pain on day 3, bad taste, and no fever. "
        "Help the dentist triage dry socket versus infection, list red flags, and suggest next actions."
    ),
    (
        "A child presents with dental trauma after a fall. The parent says a front tooth is loose. "
        "Provide a dentist co-pilot checklist for urgent questions, primary versus permanent tooth considerations, "
        "and immediate advice before examination."
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two OpenAI-compatible chat/completions servers.")
    parser.add_argument("--server-a-url", required=True)
    parser.add_argument("--server-a-model", required=True)
    parser.add_argument("--server-a-label", default="server_a")
    parser.add_argument("--server-b-url", required=True)
    parser.add_argument("--server-b-model", required=True)
    parser.add_argument("--server-b-label", default="server_b")
    parser.add_argument(
        "--task-instruction",
        default=DENTIST_COPILOT_INSTRUCTION,
        help="Instruction prepended to every prompt. Use '' to disable.",
    )
    parser.add_argument("--prompt-file", help="Optional txt/jsonl file containing prompts.")
    parser.add_argument("--output-jsonl", default="server_comparison.jsonl")
    parser.add_argument("--summary-json", default="server_comparison_summary.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[], help="Optional stop string. Repeat for multiple stops.")
    return parser.parse_args()


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


def apply_task_instruction(prompt, task_instruction):
    if not task_instruction:
        return prompt
    return f"{task_instruction}\n\nDentist co-pilot task:\n{prompt}"


def parse_response_body(body):
    text_parts = []
    metrics = {}

    stripped = body.strip()
    if stripped.startswith("{"):
        obj = json.loads(stripped)
        choices = obj.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text_parts.append(message.get("content") or choices[0].get("text") or "")
        usage = obj.get("usage") or {}
        server_metrics = obj.get("metrics") or {}
        if usage.get("completion_tokens") is not None:
            metrics["server_reported_tokens_generated"] = usage.get("completion_tokens")
        if server_metrics.get("elapsed_sec") is not None:
            metrics["server_reported_elapsed_sec"] = server_metrics.get("elapsed_sec")
        if server_metrics.get("tokens_per_sec") is not None:
            metrics["server_reported_tokens_per_sec"] = server_metrics.get("tokens_per_sec")
        if server_metrics.get("vram_used_gb") is not None:
            metrics["server_reported_vram_used_gb"] = server_metrics.get("vram_used_gb")
        return "".join(text_parts).strip(), metrics

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
            metrics.update({key: value for key, value in done_metrics.items() if value is not None})
            continue

        choices = obj.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            delta = choices[0].get("delta") or {}
            text_parts.append(message.get("content") or delta.get("content") or choices[0].get("text") or "")

    return "".join(text_parts).strip(), metrics


def query_server(url, model, prompt, max_tokens, temperature, top_p, stop_strings):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop_strings:
        payload["stop"] = stop_strings

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code} for {url}. Response body: {error_body}") from exc
    wall_sec = time.perf_counter() - start

    text, metrics = parse_response_body(body)
    metrics["wall_latency_sec"] = round(wall_sec, 3)
    return text, metrics


def mean_metric(rows, server_key, metric_key):
    values = []
    for row in rows:
        value = row.get(server_key, {}).get("metrics", {}).get(metric_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(statistics.mean(values), 3) if values else None


def max_metric(rows, server_key, metric_key):
    values = []
    for row in rows:
        value = row.get(server_key, {}).get("metrics", {}).get(metric_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(max(values), 3) if values else None


def build_summary(args, rows):
    def server_summary(server_key, label, model, url):
        return {
            "label": label,
            "model": model,
            "url": url,
            "avg_wall_latency_sec": mean_metric(rows, server_key, "wall_latency_sec"),
            "avg_reported_elapsed_sec": mean_metric(rows, server_key, "server_reported_elapsed_sec"),
            "avg_reported_tokens_per_sec": mean_metric(rows, server_key, "server_reported_tokens_per_sec"),
            "max_reported_vram_used_gb": max_metric(rows, server_key, "server_reported_vram_used_gb"),
        }

    return {
        "benchmark_config": {
            "task_instruction": args.task_instruction,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": args.stop,
            "batch_size": 1,
            "prompt_count": len(rows),
        },
        "server_a": server_summary("server_a", args.server_a_label, args.server_a_model, args.server_a_url),
        "server_b": server_summary("server_b", args.server_b_label, args.server_b_model, args.server_b_url),
    }


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file)
    rows = []

    for index, prompt in enumerate(prompts, start=1):
        print(f"Running prompt {index}/{len(prompts)}...")
        effective_prompt = apply_task_instruction(prompt, args.task_instruction)
        a_text, a_metrics = query_server(
            args.server_a_url,
            args.server_a_model,
            effective_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.stop,
        )
        b_text, b_metrics = query_server(
            args.server_b_url,
            args.server_b_model,
            effective_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.stop,
        )
        row = {
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "server_a": {
                "label": args.server_a_label,
                "model": args.server_a_model,
                "response": a_text,
                "metrics": a_metrics,
            },
            "server_b": {
                "label": args.server_b_label,
                "model": args.server_b_model,
                "response": b_text,
                "metrics": b_metrics,
            },
        }
        rows.append(row)

        print(f"\n=== Prompt {index} ===")
        print(prompt)
        print(f"\n--- {args.server_a_label} ---")
        print(a_text)
        print(f"Metrics: {a_metrics}")
        print(f"\n--- {args.server_b_label} ---")
        print(b_text)
        print(f"Metrics: {b_metrics}")

    output_path = Path(args.output_jsonl)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(args, rows)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {output_path}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
