"""Fireworks client wrapper + token accounting.

ALL inference must go through FIREWORKS_BASE_URL or it scores zero. This module
is the single choke point for that. It also tallies prompt+completion tokens so
we can mirror the judging proxy locally during development.
"""

import asyncio
import os
import threading

from openai import AsyncOpenAI


class TokenMeter:
    """Thread/async-safe running total of tokens seen (mirrors the proxy)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0
            self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class FireworksClient:
    """Async wrapper around the OpenAI-compatible Fireworks endpoint."""

    def __init__(self, meter: TokenMeter):
        api_key = os.environ.get("FIREWORKS_API_KEY")
        base_url = os.environ.get("FIREWORKS_BASE_URL")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set in environment")
        if not base_url:
            raise RuntimeError("FIREWORKS_BASE_URL not set in environment")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._meter = meter

    async def complete(self, model: str, system: str, user: str,
                       max_tokens: int, temperature: float,
                       timeout: float):
        """One chat completion. Records tokens on the meter and also returns
        per-call usage. Returns (text, prompt_tokens, completion_tokens).
        Raises on failure/timeout."""
        resp = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout,
        )
        usage = getattr(resp, "usage", None)
        ptok = getattr(usage, "prompt_tokens", 0) if usage else 0
        ctok = getattr(usage, "completion_tokens", 0) if usage else 0
        self._meter.add(ptok, ctok)
        text = (resp.choices[0].message.content or "").strip()
        return text, ptok, ctok
