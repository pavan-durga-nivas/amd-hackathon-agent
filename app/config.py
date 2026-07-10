"""Per-category configuration: prompts, output caps, and tier selection.

Terse, answer-only system prompts keep OUTPUT tokens low (the dominant lever).
`route_table.json` maps each category -> a tier ("cheap" | "strong"); tiers are
resolved to concrete model IDs at runtime from ALLOWED_MODELS. In the PoC the
tier->model map is a simple heuristic (cheap = first allowed, strong = last);
Phase-2 calibration replaces it with measured cheapest-sufficient models.
"""

import json
import os

from app import profiles, router

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROUTE_TABLE_PATH = os.path.join(_HERE, "route_table.json")

# Terse, intent-focused system prompts per category. English only. Kept SHORT:
# every token here is paid as INPUT on every call, and the instruction shapes
# OUTPUT length. Prompts are aligned with REASONING_EFFORT — categories set to
# "none" must NOT ask for step-by-step (that would dump reasoning into the
# scored output); they demand the bare answer instead.
SYSTEM_PROMPTS = {
    router.FACTUAL: "Answer correctly and concisely. No preamble.",
    router.MATH: "Solve it. Output only the final answer (a number or short value).",
    router.SENTIMENT: "Reply with one label (Positive, Negative, or Neutral) then a brief reason.",
    router.SUMMARISATION: "Summarise. Obey any length/format constraint exactly. Output only the summary.",
    router.NER: 'Extract named entities as a JSON array of {"text","type"} (type: Person, Organization, Location, Date). Only the JSON.',
    router.CODE_DEBUG: "Return only the corrected code.",
    router.LOGIC: "Output only the final answer. For multiple choice, output only the option letter.",
    router.CODE_GEN: "Output only the code.",
}

# Output-token ceilings per category. Set to the calibration-sweep levels that
# reached 100% — a *ceiling*, not a target: lean models stop early when done, so
# raising these prevents truncation without inflating tokens on easy tasks.
MAX_TOKENS = {
    router.FACTUAL: 400,
    router.MATH: 768,
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

# Reasoning-effort per category (for reasoning models like minimax-m3).
# "none" => answer DIRECTLY (no internal <think>): cheaper and avoids the failure
# where the model spends the whole token budget reasoning and returns EMPTY
# content. "" (empty) => don't send the param => natural chain-of-thought, which
# MATH genuinely needs (GSM8K is multi-step: none drops it 96%->48%). Everything
# else is better/cheaper direct: logic 40%->68%, factual/ner flat but leaner.
REASONING_EFFORT = {
    router.FACTUAL: "none",
    router.MATH: "",          # natural CoT — multi-step arithmetic needs it
    router.SENTIMENT: "none",
    router.SUMMARISATION: "none",
    router.NER: "none",
    router.CODE_DEBUG: "",     # code model; leave reasoning to the model
    router.LOGIC: "none",
    router.CODE_GEN: "",
}

# --- Phase 2: local small model (Qwen2.5-3B) answering easy tasks at ZERO
# Fireworks tokens, escalating the hard tail to the API. Enabled only when
# LOCAL_MODEL_URL is set (an in-container OpenAI-compatible endpoint), so absence
# = current Fireworks-only behavior (backward compatible).
#
# LOCAL_CATEGORIES were chosen from a measured bake-off (Qwen2.5-3B Q4, n=15/cat):
# categories where it's BOTH accurate AND emits short output (=> fast, <30s even
# on 2 vCPU). NER 93% / sentiment 80% / logic 73% qualify; math(0%)/code/summ/
# factual escalate (weak and/or long-output). Every local answer is verified and
# falls back to Fireworks on failure, so the accuracy gate is never at their mercy.
LOCAL_MODEL_URL = os.environ.get("LOCAL_MODEL_URL")  # e.g. http://127.0.0.1:8081/v1
LOCAL_MODEL_ID = os.environ.get("LOCAL_MODEL_ID", "local")
LOCAL_MODEL_KEY = os.environ.get("LOCAL_MODEL_KEY", "local")
# Tight cap on a local attempt so a slow generation escalates instead of blowing
# the per-request limit.
LOCAL_TIMEOUT_S = float(os.environ.get("LOCAL_TIMEOUT_S", "14"))
_DEFAULT_LOCAL_CATS = f"{router.NER},{router.SENTIMENT},{router.LOGIC}"
LOCAL_CATEGORIES = {c.strip() for c in
                    os.environ.get("LOCAL_CATEGORIES", _DEFAULT_LOCAL_CATS).split(",")
                    if c.strip()}


def local_enabled() -> bool:
    return bool(LOCAL_MODEL_URL)


_SENTIMENT_LABELS = ("positive", "negative", "neutral")


def verify_local(category: str, answer: str) -> bool:
    """Cheap local sanity check on a local-model answer. Lenient by design: it
    only rejects CLEARLY bad output (empty, wrong shape, refusal) so we escalate
    those to Fireworks while keeping the token win on the rest."""
    a = (answer or "").strip()
    if not a:
        return False
    low = a.lower()
    if low.startswith(("i cannot", "i can't", "i'm sorry", "as an ai", "i am unable")):
        return False
    if category == router.SENTIMENT:
        return any(lbl in low for lbl in _SENTIMENT_LABELS)
    if category == router.NER:
        # expect a JSON-ish list of entities (or an explicit empty list)
        return ("[" in a and "]" in a) or a.startswith("{")
    if category == router.LOGIC:
        return len(a) <= 400  # short deductive answer; reject runaway text
    return True


_PREFIX = "accounts/fireworks/models/"

# Context windows (tokens) for the second routing axis: never send an input that
# can't fit. Unknown models default to a conservative-but-large window (modern
# Fireworks models are 128k+); the gate only escalates when an input *provably*
# won't fit the picked model, so a wrong-small default can't cause false escalations.
_DEFAULT_CONTEXT = 131072
_CONTEXT_WINDOW = {
    # launch-day ALLOWED_MODELS (Track 1 final set)
    "minimax-m3": 131072,
    "kimi-k2p7-code": 131072,
    "gemma-4-31b-it": 131072,
    "gemma-4-26b-a4b-it": 131072,
    "gemma-4-31b-it-nvfp4": 131072,
    # dev pool
    "deepseek-v4-flash": 131072,
    "deepseek-v4-pro": 131072,
    "kimi-k2p6": 131072,
    "gpt-oss-120b": 131072,
    "glm-5p1": 131072,
    "glm-5p2": 131072,
}


def context_window(model: str) -> int:
    """Context window for a model id (short or full), default large if unknown."""
    short = model.split("/")[-1]
    return _CONTEXT_WINDOW.get(short, _DEFAULT_CONTEXT)

# Capability-tiered model preference per category, as ordered *needles* matched
# by SUFFIX against whatever ALLOWED_MODELS the harness injects. We never depend
# on a specific model being present: the resolver returns the EXACT allowed id
# for the first needle that matches, else falls back to any allowed model.
#
# WHY THIS EXISTS: our first submission hard-coded dev models (deepseek-v4-flash)
# in a fixed route table. The real launch-day ALLOWED_MODELS was
# {minimax-m3, kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it,
# gemma-4-31b-it-nvfp4} — deepseek absent — so every TEXT task fell through to
# kimi-k2p7-code (a CODE model) and the accuracy gate failed (73.7%). Selecting
# by capability over the actual allowed set fixes that regardless of the list.
#
# Accuracy-first ordering: a strong general/reasoning model (minimax) leads for
# language+reasoning; the code specialist (kimi) leads for code; instruction-
# following Gemma is the general fallback. Once we clear the gate, easy
# categories can be shifted toward the cheaper Gemma tier for token rank.
_CODE_PREF = ["kimi-k2p7-code", "-code", "coder", "minimax", "deepseek",
              "glm", "gemma-4-31b-it", "gemma"]
_REASON_PREF = ["minimax", "deepseek", "glm-5p2", "glm", "kimi-k2p7-code",
                "gemma-4-31b-it", "gemma"]
_TEXT_PREF = ["minimax", "gemma-4-31b-it", "gemma", "deepseek", "glm", "kimi"]
_CATEGORY_PREFS = {
    router.CODE_DEBUG: _CODE_PREF,
    router.CODE_GEN: _CODE_PREF,
    router.MATH: _REASON_PREF,
    router.LOGIC: _REASON_PREF,
    router.FACTUAL: ["minimax", "deepseek", "glm", "gemma-4-31b-it", "gemma", "kimi"],
    router.SENTIMENT: _TEXT_PREF,
    router.NER: _TEXT_PREF,
    router.SUMMARISATION: _TEXT_PREF,
}
_DEFAULT_PREF = ["minimax", "gemma-4-31b-it", "deepseek", "glm", "kimi", "gemma"]


def _full(name: str) -> str:
    return name if name.startswith("accounts/") else _PREFIX + name


def _short(model_id: str) -> str:
    return str(model_id).strip().rsplit("/", 1)[-1].lower()


def load_route_table() -> dict:
    """Kept for API compatibility (evaluate.py passes it through). The resolver
    now selects by capability over ALLOWED_MODELS, so this is advisory only."""
    try:
        with open(_ROUTE_TABLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def candidate_models(category: str, allowed: list, input_tokens: int = 0) -> list:
    """Ordered list of EXACT ALLOWED_MODELS ids to try for this category,
    capability-first. The first entry is the primary; the rest are fallbacks
    for retry-on-error/empty. Never returns a model outside `allowed`."""
    if not allowed:
        raise ValueError("ALLOWED_MODELS is empty")
    prefs = _CATEGORY_PREFS.get(category, _DEFAULT_PREF)

    picked = []
    for needle in prefs:
        needle = needle.lower()
        for m in allowed:
            if needle in _short(m) and m not in picked:
                picked.append(m)
                break
    # Guarantee every allowed model is reachable as a last-resort fallback.
    for m in allowed:
        if m not in picked:
            picked.append(m)

    # Second routing axis (input size): if the primary can't fit input + output
    # budget, promote the roomiest-context allowed model to the front so the
    # request isn't truncated at the context boundary.
    needed = input_tokens + MAX_TOKENS.get(category, 512) + 256
    if picked and needed > context_window(picked[0]):
        roomiest = max(allowed, key=context_window)
        if context_window(roomiest) > context_window(picked[0]):
            picked.remove(roomiest)
            picked.insert(0, roomiest)
    return picked


def resolve_model(category: str, route_table: dict, allowed: list,
                  input_tokens: int = 0) -> str:
    """Primary (best) model for a category from ALLOWED_MODELS — capability-first,
    robust to whatever the harness injects. `route_table` is ignored (advisory)."""
    return candidate_models(category, allowed, input_tokens)[0]


def fallback_models(primary: str, allowed: list, k: int = 1, category: str = "") -> list:
    """Ordered fallbacks (excluding primary) for retry-on-error, capability-first."""
    ordered = [m for m in candidate_models(category, allowed) if m != primary]
    return ordered[:k]


# --- Profile-aware output plan (call this instead of the raw dicts) ---
# Layers the active AGENT_PROFILE's knobs over the baselines above so token
# strategy is dialable per Docker build. Crucially, it respects a length the
# TASK itself states: our terseness cap is skipped (and any lowered output
# ceiling restored) when the prompt requests a specific length/format, so we
# never get judged-wrong for ignoring a requested word/sentence count.

def output_plan(category: str, prompt: str = ""):
    """Return (system_prompt, max_tokens, stop) for one task under the profile."""
    sys = profiles.system_override(category) or SYSTEM_PROMPTS[category]
    mt = profiles.max_tokens_override(category)
    mt = mt if mt is not None else MAX_TOKENS[category]
    hint = profiles.length_hint(category)

    if router.has_length_constraint(prompt):
        # Task dictates its own length: don't nudge shorter, and never let a
        # profile ceiling truncate a legitimately longer required answer.
        mt = max(mt, MAX_TOKENS[category])
    elif hint:
        sys = sys + hint

    return sys, mt, profiles.stop_sequences(category)
