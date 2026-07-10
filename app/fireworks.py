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
                       timeout: float, reasoning_effort: str = "none",
                       stop=None):
        """One chat completion. Records tokens on the meter and also returns
        per-call usage. Returns (text, prompt_tokens, completion_tokens).
        Raises on failure/timeout.

        reasoning_effort="none" tells reasoning models (e.g. minimax-m3) to answer
        DIRECTLY instead of spending the whole token budget on an internal <think>
        block — which otherwise truncates to EMPTY content and both fails the
        accuracy gate and wastes tokens. Models that don't support the parameter
        are retried once without it."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        stop_kw = {"stop": stop} if stop else {}

        async def _call(extra):
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                    **stop_kw, **extra),
                timeout=timeout,
            )
            return resp

        extra = {"extra_body": {"reasoning_effort": reasoning_effort}} if reasoning_effort else {}
        try:
            resp = await _call(extra)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise
        except Exception:  # noqa: BLE001 - model may reject reasoning_effort; retry plain
            if not extra:
                raise
            resp = await _call({})

        usage = getattr(resp, "usage", None)
        ptok = getattr(usage, "prompt_tokens", 0) if usage else 0
        ctok = getattr(usage, "completion_tokens", 0) if usage else 0
        self._meter.add(ptok, ctok)
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        # If a reasoning model still returned empty content but exposed its
        # reasoning channel, fall back to that so we never emit a blank answer.
        if not text:
            text = (getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None) or "").strip()
        return text, ptok, ctok
