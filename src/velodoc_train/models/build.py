import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def _remote_code_allowed(trust_remote_code: bool) -> bool:
    if not trust_remote_code:
        return False
    allow = os.getenv("VELODOC_ALLOW_REMOTE_CODE", "").lower() in {"1", "true", "yes"}
    if not allow:
        raise RuntimeError(
            "Refusing to load a model with trust_remote_code=true. "
            "Set VELODOC_ALLOW_REMOTE_CODE=1 only after reviewing the model repository code."
        )
    return True


def _dtype(dtype: str):
    dtype = dtype.lower()
    if dtype == "bf16":
        import torch
        return torch.bfloat16
    if dtype == "fp16":
        import torch
        return torch.float16
    if dtype == "fp32":
        return None
    raise ValueError(f"Unsupported dtype: {dtype}")


def _auth_token():
    return os.getenv("HF_TOKEN") or None

def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=_remote_code_allowed(trust_remote_code),
        token=_auth_token(),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_causal_lm(
    model_id: str,
    dtype: str,
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
):
    kwargs = {
        "trust_remote_code": _remote_code_allowed(trust_remote_code),
        "token": _auth_token(),
    }
    torch_dtype = _dtype(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
