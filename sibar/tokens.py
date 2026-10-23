"""Token cost c(f) of a serialized fact.

SIBAR_TOKENIZER picks the tokenizer: a tiktoken encoding (default cl100k_base) or a HuggingFace
tokenizer id such as Qwen/Qwen2.5-14B-Instruct, which counts tokens as the paper's agent sees them.
"""
from __future__ import annotations

import functools
import os

SIGNATURE_TOKENS = 4  # serialized provenance signature s_f
DEFAULT_TOKENIZER = "cl100k_base"


def tokenizer_name() -> str:
    from .llm import load_env
    load_env()
    return os.environ.get("SIBAR_TOKENIZER", "").strip() or DEFAULT_TOKENIZER


class _HF:
    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


@functools.lru_cache(maxsize=1)
def _encoder():
    name = tokenizer_name()
    try:
        if "/" in name:
            return _HF(name)
        import tiktoken
        return tiktoken.get_encoding(name)
    except Exception:  # offline or not installed: fall back to a character heuristic
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return max(1, ascii_chars // 4 + (len(text) - ascii_chars))


def truncate_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    if enc is not None:
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        # drop a multi-byte character cut in half at the boundary
        return enc.decode(ids[:max_tokens]).rstrip("�")
    while count_tokens(text) > max_tokens and text:
        text = text[:-1]
    return text


def serialize_fact(name: str, attr: str, value: str) -> str:
    return f"{name}|{attr}|{value}"


def fact_cost(name: str, attr: str, value: str) -> int:
    return count_tokens(serialize_fact(name, attr, value)) + SIGNATURE_TOKENS
