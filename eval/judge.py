"""Mock accuracy judge — approximates the hackathon's LLM-judge gate.

Deterministic-first: math -> numeric match, sentiment -> label match,
NER -> entity coverage, code -> run against unit tests, factual/logic ->
key-term coverage. Open-ended summaries use an LLM-judge (a strong model),
with a key-term fallback when no judge client is available.

Each grader returns (passed: bool, detail: str).
"""

import re
import subprocess
import sys
import tempfile
import os


def _numbers(text: str):
    out = []
    for tok in re.findall(r"[-+]?\d[\d,]*\.?\d*", text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def grade_numeric(answer: str, expected: float):
    nums = _numbers(answer)
    if not nums:
        return False, f"no number found (expected {expected})"
    best = min(nums, key=lambda x: abs(x - expected))
    ok = abs(best - expected) <= max(0.05, abs(expected) * 0.001)
    return ok, f"got {best}, expected {expected}"


def grade_label(answer: str, expected: str):
    """The classification is the FIRST sentiment label to appear (the justification
    may mention other label words, e.g. 'Neutral - no positive or negative tone')."""
    a = (answer or "").lower()
    positions = {lbl: a.find(lbl) for lbl in ("positive", "negative", "neutral") if lbl in a}
    if not positions:
        return False, f"no label emitted, expected {expected}"
    classified = min(positions, key=positions.get)  # earliest-appearing label
    ok = classified == expected.lower()
    return ok, f"classified={classified}, expected {expected}"


def grade_alias(answer: str, aliases):
    """Factual (TriviaQA): pass if any gold alias appears in the answer.
    Case/space-insensitive substring match, longest aliases first."""
    a = re.sub(r"\s+", " ", (answer or "").lower())
    for alias in aliases:
        if re.sub(r"\s+", " ", alias.lower()) in a:
            return True, f"matched alias {alias!r}"
    return False, f"no alias matched (of {len(aliases)}, e.g. {aliases[:2]})"


# Explicit "the answer is X" style markers — the model's *conclusion*, which for
# a reasoning dump is what we want (not the first stray "A" = the article).
_ANSWER_MARKER = re.compile(
    r"(?:answer|option|choice|choose|select|correct|pick)(?:\s+\w+){0,3}?"
    r"\s*(?:is|:|=|would be|should be)?\s*\(?\b([A-F])\b\)?", re.I)


def grade_mcq(answer: str, options, answer_index: int):
    """Logic (LogiQA): pass if the answer selects the correct option, either by
    letter (A/B/C/D) or by restating the correct option's text.

    The model often emits a reasoning preamble before its pick, so the FIRST
    standalone letter is usually noise (the article 'A', an enumerated name).
    Resolve the pick as, in order: (1) the LAST explicit answer-marker letter,
    (2) if the reply is just a bare letter, that letter, (3) the LAST standalone
    letter overall, (4) the correct option's text appearing verbatim."""
    letters = ["A", "B", "C", "D", "E", "F"]
    gold_letter = letters[answer_index]
    a = (answer or "").strip()
    au = a.upper()

    marks = _ANSWER_MARKER.findall(au)
    if marks:
        pick = marks[-1]
        return pick == gold_letter, f"picked {pick} (marker), expected {gold_letter}"

    bare = au.strip(" .()")
    if len(bare) == 1 and bare in letters:  # reply is just the letter
        return bare == gold_letter, f"picked {bare}, expected {gold_letter}"

    standalones = re.findall(r"\b([A-F])\b", au)
    if standalones:
        pick = standalones[-1]  # conclusion sits at the END of a reasoning dump
        return pick == gold_letter, f"picked {pick} (last), expected {gold_letter}"

    # fallback: did they quote the correct option text (and not a wrong one)?
    al = a.lower()
    gold_txt = options[answer_index].lower().strip(" .")
    if gold_txt and gold_txt in al:
        return True, f"matched option text {gold_letter}"
    return False, f"no clear choice, expected {gold_letter}"


def grade_contains_all(answer: str, must_include):
    a = (answer or "").lower()
    missing = [t for t in must_include if t.lower() not in a]
    return (len(missing) == 0), (f"missing={missing}" if missing else "all terms present")


def grade_entities(answer: str, entities):
    a = (answer or "").lower()
    missing = [e for e in entities if e.lower() not in a]
    return (len(missing) == 0), (f"missing entities={missing}" if missing else "all entities present")


_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _extract_code(answer: str) -> str:
    m = _CODE_BLOCK.findall(answer or "")
    if m:
        # prefer the longest fenced block (usually the full solution)
        return max(m, key=len)
    return answer or ""


def grade_code(answer: str, tests, timeout: float = 8.0):
    code = _extract_code(answer)
    script = code + "\n\n" + "\n".join(tests) + "\nprint('ALL_TESTS_PASSED')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=timeout)
        ok = proc.returncode == 0 and "ALL_TESTS_PASSED" in proc.stdout
        detail = "tests passed" if ok else (proc.stderr.strip().split("\n")[-1][:120] or "test failed")
        return ok, detail
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"exec error: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def grade_judge(prompt: str, answer: str, rubric: str, judge_client, judge_model: str):
    """LLM-judge for open-ended answers. Falls back handled by caller."""
    sys_p = ("You are a fair grader. Decide if the candidate answer satisfies the "
             "requirement. Judge on meaning, not exact wording; a correct answer "
             "phrased differently still PASSES. Your VERY FIRST word must be PASS or "
             "FAIL, then optionally one short sentence of reason.")
    user_p = (f"Task: {prompt}\n\nRequirement: {rubric}\n\n"
              f"Candidate answer: {answer}\n\nStart your reply with PASS or FAIL.")
    resp = judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        max_tokens=512, temperature=0,
    )
    verdict = (resp.choices[0].message.content or "").strip().upper()
    # Verdict-first prompt: take whichever word appears FIRST, so a truncated
    # reply still yields a verdict instead of a false FAIL.
    ip, if_ = verdict.find("PASS"), verdict.find("FAIL")
    if ip == -1 and if_ == -1:
        passed = False
    elif ip == -1:
        passed = False
    elif if_ == -1:
        passed = True
    else:
        passed = ip < if_
    return passed, f"judge={verdict[:20]!r}"


def grade(task: dict, answer: str, judge_client=None, judge_model: str = ""):
    """Dispatch to the right grader. Returns (passed, detail)."""
    g = task.get("grader")
    if g == "numeric":
        return grade_numeric(answer, task["answer_value"])
    if g == "alias":
        return grade_alias(answer, task["aliases"])
    if g == "mcq":
        return grade_mcq(answer, task["options"], task["answer_index"])
    if g == "label":
        return grade_label(answer, task["expected_label"])
    if g == "contains_all":
        return grade_contains_all(answer, task["must_include"])
    if g == "entities":
        return grade_entities(answer, task["entities"])
    if g == "code":
        return grade_code(answer, task["tests"])
    if g == "judge":
        if judge_client is not None:
            return grade_judge(task["prompt"], answer, task.get("rubric", ""), judge_client, judge_model)
        # offline fallback: key-term coverage
        return grade_contains_all(answer, task.get("must_include", []))
    return False, f"unknown grader {g}"
