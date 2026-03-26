import os, json

def write_summary(out_dir: str, summary: dict) -> str:
    p = os.path.join(out_dir, "summaries")
    os.makedirs(p, exist_ok=True)
    fp = os.path.join(p, "metrics_summary.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return fp