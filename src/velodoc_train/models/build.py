from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_causal_lm(model_id: str, dtype: str, trust_remote_code: bool = True):
    kwargs = {"trust_remote_code": trust_remote_code}
    if dtype == "bf16":
        import torch
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        import torch
        kwargs["torch_dtype"] = torch.float16
    else:
        # fp32 default
        pass
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)