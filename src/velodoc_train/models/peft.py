from peft import LoraConfig, get_peft_model


def _infer_target_modules(model):
    model_type = str(getattr(model.config, "model_type", "")).lower()
    module_class_names = {module.__class__.__name__ for module in model.modules()}
    if "gemma4" in model_type or "Gemma4ClippableLinear" in module_class_names:
        return [
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ]
    if "gemma" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "qwen" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return None


def apply_peft(model, peft_cfg: dict):
    method = peft_cfg["method"].lower()
    if method not in ("lora", "dora"):
        raise ValueError(f"Unsupported PEFT method: {method}")

    # DoRA is enabled via use_dora flag in LoRA config in PEFT
    use_dora = (method == "dora")

    target_modules = peft_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = _infer_target_modules(model)

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
