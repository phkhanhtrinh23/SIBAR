"""Backend factory driven by .env. Every LLM call goes through OpenRouter with one key.

AGENT_BACKEND / PREDICTOR_BACKEND: oracle | openrouter
AGENT_MODEL / PREDICTOR_MODEL: OpenRouter model ids; OPENROUTER_API_KEY: the single key.
AGENT_BASE_URL / PREDICTOR_BASE_URL: optional OpenAI-compatible endpoint for that role, e.g. a local
vLLM server for Qwen2.5-14B-Instruct, the paper's agent, which OpenRouter does not serve.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import ChatBackend, DiskCache, extract_json

DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_backend(role: str = "AGENT", cache_dir: str = ".cache") -> Optional[ChatBackend]:
    """Returns None for the deterministic 'oracle' backend (no LLM calls)."""
    load_env()
    kind = os.environ.get(f"{role}_BACKEND", "oracle").strip().lower()
    if kind in ("", "oracle", "none"):
        return None
    if kind != "openrouter":
        raise ValueError(f"unknown backend {kind!r} for {role}_BACKEND (use oracle | openrouter)")
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is empty: fill it in .env")
    from .openrouter_backend import OPENROUTER_BASE_URL, OpenRouterBackend
    model = os.environ.get(f"{role}_MODEL") or DEFAULT_MODEL
    cache = DiskCache(os.path.join(cache_dir, "llm_cache.sqlite"))
    base_url = (os.environ.get(f"{role}_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
                or OPENROUTER_BASE_URL)
    return OpenRouterBackend(model, cache, key, base_url)


__all__ = ["ChatBackend", "DiskCache", "extract_json", "get_backend", "load_env"]
