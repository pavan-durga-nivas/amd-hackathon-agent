"""Per-category configuration: prompts, output caps, and tier selection.

Terse, answer-only system prompts keep OUTPUT tokens low (the dominant lever).
`route_table.json` maps each category -> a tier ("cheap" | "strong"); tiers are
resolved to concrete model IDs at runtime from ALLOWED_MODELS. In the PoC the
tier->model map is a simple heuristic (cheap = first allowed, strong = last);
Phase-2 calibration replaces it with measured cheapest-sufficient models.
"""

import json
import os

from app import router

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROUTE_TABLE_PATH = os.path.join(_HERE, "route_table.json")

# Terse, intent-focused system prompts per category. English only.
SYSTEM_PROMPTS = {
    router.FACTUAL: "You are precise. Answer the question correctly and concisely. No preamble, no restating the question.",
    router.MATH: "Solve the problem. Reason internally but output ONLY the final answer (a number or short value), nothing else.",
    router.SENTIMENT: "Classify the sentiment as Positive, Negative, or Neutral. Output the label followed by a one-line justification.",
    router.SUMMARISATION: "Summarise the text. Obey any length or format constraint in the request exactly. Output only the summary.",
    router.NER: 'Extract named entities. Output a JSON array of {"text","type"} where type is one of Person, Organization, Location, Date. Output only the JSON.',
    router.CODE_DEBUG: "Identify the bug and return the corrected code. Output only the corrected code, no explanation unless asked.",
    router.LOGIC: "Solve the problem. Think step by step, then end your reply with a final line exactly of the form 'Answer: X' where X is your choice (the option letter for multiple-choice, otherwise the answer).",
    router.CODE_GEN: "Write a correct, well-structured solution that meets the spec. Output only code, no commentary.",
}

# Output-token ceilings per category. Set to the calibration-sweep levels that
# reached 100% — a *ceiling*, not a target: lean models stop early when done, so
# raising these prevents truncation without inflating tokens on easy tasks.
MAX_TOKENS = {
    router.FACTUAL: 400,
    router.MATH: 512,
    router.SENTIMENT: 300,
    router.SUMMARISATION: 400,
    router.NER: 400,
    router.CODE_DEBUG: 1024,
    router.LOGIC: 768,
    router.CODE_GEN: 1024,
}

# Sampling temperature per category — low/zero for deterministic tasks.
TEMPERATURE = {
    router.FACTUAL: 0.0, router.MATH: 0.0, router.SENTIMENT: 0.0,
    router.SUMMARISATION: 0.2, router.NER: 0.0, router.CODE_DEBUG: 0.0,
    router.LOGIC: 0.0, router.CODE_GEN: 0.1,
}

_PREFIX = "accounts/fireworks/models/"

# Context windows (tokens) for the second routing axis: never send an input that
# can't fit. Unknown models default to a conservative-but-large window (modern
# Fireworks models are 128k+); the gate only escalates when an input *provably*
# won't fit the picked model, so a wrong-small default can't cause false escalations.
_DEFAULT_CONTEXT = 131072
_CONTEXT_WINDOW = {
    "deepseek-v4-flash": 131072,
    "deepseek-v4-pro": 131072,
    "kimi-k2p7-code": 131072,
    "kimi-k2p6": 131072,
    "gpt-oss-120b": 131072,
    "glm-5p1": 131072,
    "glm-5p2": 131072,
}


def context_window(model: str) -> int:
    """Context window for a model id (short or full), default large if unknown."""
    short = model.split("/")[-1]
    return _CONTEXT_WINDOW.get(short, _DEFAULT_CONTEXT)

# Calibrated cheapest-sufficient model per category (measured on the dev set;
# see eval/calibrate.py). Concrete IDs — route_table.json overrides these.
_DEFAULT_ROUTE_TABLE = {
    router.FACTUAL: _PREFIX + "deepseek-v4-flash",
    router.MATH: _PREFIX + "kimi-k2p7-code",
    router.SENTIMENT: _PREFIX + "deepseek-v4-flash",
    router.SUMMARISATION: _PREFIX + "deepseek-v4-flash",
    router.NER: _PREFIX + "deepseek-v4-flash",
    router.CODE_DEBUG: _PREFIX + "kimi-k2p7-code",
    router.LOGIC: _PREFIX + "kimi-k2p7-code",
    router.CODE_GEN: _PREFIX + "deepseek-v4-flash",
}

# Preference order for fallback (leanest/most-reliable first). Used when the
# calibrated model isn't in the launch-day ALLOWED_MODELS, or on error.
_PREFERRED_ORDER = ["deepseek-v4-flash", "kimi-k2p7-code", "deepseek-v4-pro",
                    "gpt-oss-120b", "glm-5p1", "glm-5p2", "kimi-k2p6"]


def _full(name: str) -> str:
    return name if name.startswith("accounts/") else _PREFIX + name


def load_route_table() -> dict:
    """Load category->model_id map from route_table.json, else calibrated defaults."""
    try:
        with open(_ROUTE_TABLE_PATH, "r", encoding="utf-8") as f:
            table = json.load(f)
        return {c: _full(table.get(c) or _DEFAULT_ROUTE_TABLE[c]) for c in router.CATEGORIES}
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_ROUTE_TABLE)


def resolve_model(category: str, route_table: dict, allowed: list,
                  input_tokens: int = 0) -> str:
    """Primary model for a category: the calibrated pick if it's in ALLOWED_MODELS,
    otherwise the leanest preferred model that IS available, else the first allowed.
    Keeps us robust if launch-day ALLOWED_MODELS differs from our dev pool.

    Second routing axis (input size): if the input plus this category's output
    budget can't fit the chosen model's context window, escalate to the
    largest-context available model so the request never gets truncated at the
    context boundary. On our dev/bench inputs this never fires (all <2k tokens),
    but it's cheap insurance against an unusually large real task."""
    if not allowed:
        raise ValueError("ALLOWED_MODELS is empty")
    primary = _full(route_table.get(category, ""))
    if primary not in allowed:
        primary = next((_full(n) for n in _PREFERRED_ORDER if _full(n) in allowed),
                       allowed[0])

    # Hard context-window gate. Reserve the category's output budget + a margin.
    needed = input_tokens + MAX_TOKENS.get(category, 512) + 256
    if needed > context_window(primary):
        roomiest = max(allowed, key=context_window)
        if context_window(roomiest) > context_window(primary):
            return roomiest
    return primary


def fallback_models(primary: str, allowed: list, k: int = 1) -> list:
    """Ordered fallbacks from ALLOWED_MODELS (excluding primary) for retry-on-error."""
    ordered = [_full(n) for n in _PREFERRED_ORDER if _full(n) in allowed and _full(n) != primary]
    for m in allowed:  # append any remaining allowed not already covered
        if m != primary and m not in ordered:
            ordered.append(m)
    return ordered[:k]
