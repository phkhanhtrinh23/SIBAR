"""OpenRouter backend (OpenAI-compatible API, one key for every model)."""
from __future__ import annotations

from openai import OpenAI

from .base import ChatBackend, DiskCache

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ChatBackend):
    name = "openrouter"

    def __init__(self, model: str, cache: DiskCache, api_key: str, base_url: str = OPENROUTER_BASE_URL,
                 temperature: float = 0.0):
        super().__init__(model, cache)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""
