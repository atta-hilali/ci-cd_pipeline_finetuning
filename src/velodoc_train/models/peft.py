from peft import LoraConfig, get_peft_model

def apply_peft(model, peft_cfg: dict):
    method = peft_cfg["method"].lower()
    if method not in ("lora", "dora"):
        raise ValueError(f"Unsupported PEFT method: {method}")

    # DoRA is enabled via use_dora flag in LoRA config in PEFT
    use_dora = (method == "dora")

    target_modules = peft_cfg.get("target_modules", "auto")
    # "auto" means let PEFT infer (works for many HF models). You can hardcode later.
    if target_modules == "auto":
        target_modules = None

    lora = LoraConfig(
        r=int(peft_cfg["r"]),
        lora_alpha=int(peft_cfg["alpha"]),
        lora_dropout=float(peft_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_dora=use_dora,
    )
    return get_peft_model(model, lora)