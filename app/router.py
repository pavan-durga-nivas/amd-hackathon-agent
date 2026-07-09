"""Free, local category router.

Detects which of the 8 capability categories a task belongs to using cheap
rules (regex/keywords/structure). Runs entirely locally -> zero scored tokens.
Category detection is the primary routing signal; input length is the second
axis (see main.py). Embedding fallback is a Phase-2 experiment, not in the PoC.
"""

import re

# Canonical category names used across the app and route_table.json.
FACTUAL = "factual"
MATH = "math"
SENTIMENT = "sentiment"
SUMMARISATION = "summarisation"
NER = "ner"
CODE_DEBUG = "code_debug"
LOGIC = "logic"
CODE_GEN = "code_gen"

CATEGORIES = [
    FACTUAL, MATH, SENTIMENT, SUMMARISATION, NER, CODE_DEBUG, LOGIC, CODE_GEN,
]

# Structural signal: does the prompt contain code? (fenced block or code tokens)
_CODE_FENCE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\breturn\b|;\s*$|=>|\(\)\s*[:{]", re.M)

_MATH_HINT = re.compile(
    r"\b(calculate|compute|how many|how much|percent|percentage|average|"
    r"projection|profit|interest\b|discount|\btax\b|\bsum of\b)\b|\d+\s*%|"
    r"\$\s*\d|\d+\s*[+\-*/x]\s*\d", re.I)
# Summary intent = an explicit condense verb, NOT a mere length constraint.
_SUMMARY_HINT = re.compile(r"\b(summ?aris[ez]e?|summ?ary|tl;?dr|condense|shorten)\b", re.I)
_SENTIMENT_HINT = re.compile(r"\b(sentiment|positive or negative|how (do(es)?|did) .{0,20}feel|"
                             r"tone of|emotion(al)? (of|in))\b", re.I)
_NER_HINT = re.compile(r"named entit|extract .{0,30}?(entit|names?|persons?|organi[sz]|"
                       r"location|dates?)|identify .{0,30}?(person|organi[sz]|location|date)", re.I)
_DEBUG_HINT = re.compile(r"\b(bug|debug|fix (the|this|it)|error in|why does .{0,30}(fail|crash|not work)|"
                         r"corrected? (version|implementation|code)|what'?s wrong|there is a (bug|mistake))", re.I)
_CODEGEN_HINT = re.compile(r"write .{0,30}?(function|program|code|method|script|class)|"
                           r"implement .{0,30}?(function|class|method|algorithm|the)|"
                           r"generate .{0,15}?code|create .{0,25}?(function|class|method)|\bdef \w+\(", re.I)
_LOGIC_HINT = re.compile(r"\b(puzzle|deduce|deduction|riddle|constraint)\b|logic(al)? (puzzle|reasoning)|"
                         r"each (has|have|is|gets|owns)|who (is|are|owns|has|gets|sits|lives|plays)|"
                         r"if .{0,50}? then|different (pet|colou?r|house|job|hat|drink|profession)", re.I)
# Structural signal: a multiple-choice block (>=3 lettered options on their own
# lines). Deductive-reasoning benchmarks (LogiQA etc.) are posed this way and
# carry no keyword cue, so they'd otherwise fall through to the factual default.
_MCQ_OPTION = re.compile(r"(?m)^\s*\(?([A-E])[.)]\s+\S")


def detect_category(prompt: str) -> str:
    """Return the best-guess category for a task prompt (local, free).

    Order matters: structural code signals and specific intents are checked
    before generic fallbacks so 'debug this code' beats 'write code' beats
    'factual question'.
    """
    p = prompt or ""
    has_code = bool(_CODE_FENCE.search(p))

    # Code categories first — a code block plus a fix/bug cue is unambiguous.
    if has_code and _DEBUG_HINT.search(p):
        return CODE_DEBUG
    if _CODEGEN_HINT.search(p):
        return CODE_GEN
    if has_code and _DEBUG_HINT.search(p):
        return CODE_DEBUG

    if _SUMMARY_HINT.search(p):
        return SUMMARISATION
    if _NER_HINT.search(p):
        return NER
    if _SENTIMENT_HINT.search(p):
        return SENTIMENT
    if _LOGIC_HINT.search(p):
        return LOGIC
    # A multiple-choice block with >=3 distinct options is a deductive-reasoning task.
    if len({m.group(1) for m in _MCQ_OPTION.finditer(p)}) >= 3:
        return LOGIC
    if _MATH_HINT.search(p):
        return MATH

    # Fallback: treat as a factual/open question.
    return FACTUAL
