"""Strategy profiles — dialable token/accuracy knobs, kestrel-style.

The Docker image bakes a `PROFILE` build-arg into `AGENT_PROFILE`; this module
turns that into a set of knob overrides applied at import time. That lets us A/B
different token-minimization strategies by rebuilding with a different arg (no
code change), then tag each image by profile (`:lean`, `:floor`, ...).

Scoring counts RAW Fireworks tokens (input+output), so the knobs here all reduce
COUNT, never "cost": output caps + length hints (fewer output tokens) and input
caps (fewer input tokens). Model choice is deliberately NOT a knob — a cheaper
model emits the same token count for the same answer, so it can't move the score.

Profiles are layered over `config`'s baselines; "A0" is the current safe baseline
(no overrides), so an unset/unknown AGENT_PROFILE changes nothing.
"""

import os

from app import router

PROFILE = (os.environ.get("AGENT_PROFILE") or "A0").strip() or "A0"

# Knob namespaces. Any category absent from a namespace falls back to config's
# baseline for that knob.
#   max_tokens   -> overrides config.MAX_TOKENS ceiling (lower = harder output cap)
#   length_hint  -> short instruction appended to the system prompt to shape
#                   output length (e.g. "Limit to 2 sentences.")
#   stop         -> stop sequences that end generation early
#   input_cap    -> max INPUT tokens; longer inputs get compressed to this budget
_PROFILES = {
    # Baseline: exactly today's behavior. Safe fallback.
    "A0": {},

    # Output-minimized (SAFE, shippable): control only summarisation output —
    # a 2-sentence hint (skipped when the task states its own length) plus a
    # matching ceiling. Validated: no accuracy regression, real cut on long
    # summaries. Deliberately does NOT cap code/math max_tokens: a cap there
    # can't reduce tokens without TRUNCATING the answer, which broke code_debug
    # (100%->84%) and nicked math. max_tokens is a safety ceiling, not a lever.
    "lean": {
        "max_tokens": {
            router.SUMMARISATION: 200,
        },
        "length_hint": {
            router.SUMMARISATION: " Limit to 2 sentences.",
        },
        "stop": {},
        "input_cap": {},
    },

    # EXPERIMENTAL — launch-day only, do NOT ship blind. Rewrites prompts to
    # delete non-scored verbosity (sentiment "reason", code docstrings, factual
    # prose). On our dev key these terse prompts trigger minimax-m3's reasoning
    # leak (empty content -> we emit the reasoning channel), which HURTS. But the
    # real ALLOWED_MODELS also has non-reasoning gemma models that should follow
    # terse prompts cleanly — so this profile is worth A/B-ing on launch day when
    # text categories can route to gemma. No max_tokens caps here: caps only
    # truncate (they broke code_debug), they don't reduce tokens.
    "floor": {
        "system": {
            router.SENTIMENT: "Reply with exactly one word — Positive, Negative, or Neutral. No preamble.",
            router.FACTUAL: "Answer in as few words as possible — just the answer, no sentence. No preamble.",
            router.CODE_GEN: "Output only the code — no comments, docstrings, or explanation. No preamble.",
            router.CODE_DEBUG: "Output only the corrected code — no comments or explanation. No preamble.",
        },
        "max_tokens": {
            router.SUMMARISATION: 200,
        },
        "length_hint": {
            router.SUMMARISATION: " Limit to 2 sentences.",
        },
        "stop": {},
        "input_cap": {},
    },
}

_ACTIVE = _PROFILES.get(PROFILE, _PROFILES["A0"])


def _ns(name: str) -> dict:
    return _ACTIVE.get(name, {})


def system_override(category: str):
    """Profile's full system-prompt replacement for a category, or None."""
    return _ns("system").get(category)


def max_tokens_override(category: str):
    """Profile's max_tokens ceiling for a category, or None to use the baseline."""
    return _ns("max_tokens").get(category)


def length_hint(category: str) -> str:
    """Text appended to the system prompt to shape output length ('' if none)."""
    return _ns("length_hint").get(category, "")


def stop_sequences(category: str):
    """Stop sequences for a category, or None."""
    return _ns("stop").get(category) or None


def input_cap(category: str) -> int:
    """Max input tokens for a category (0 = uncapped)."""
    return int(_ns("input_cap").get(category, 0))
