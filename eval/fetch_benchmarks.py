"""Build a REAL benchmark set from public datasets (no hand-authored tasks).

Pulls N samples/category from canonical public benchmarks via the HuggingFace
datasets-server HTTP API (no `datasets` lib needed), normalizes each into our
dev_set.json schema (task_id, category, prompt, grader, + ground truth), and
writes eval/bench_set.json.

    venv/bin/python -m eval.fetch_benchmarks --n 15
    venv/bin/python -m eval.fetch_benchmarks --n 30 --out eval/bench_set.json

Category -> benchmark:
    factual       TriviaQA (rc.nocontext)   answer-alias match
    math          GSM8K                     numeric (#### N)
    sentiment     SST-2                     label match (pos/neg)
    summarisation XSum                      LLM-judge vs reference
    ner           CoNLL-2003 (tner)         entity coverage
    logic         LogiQA                    multiple-choice letter
    code_gen      HumanEval                 execute unit tests
    code_debug    HumanEvalPack (fix)       execute unit tests
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS_API = "https://datasets-server.huggingface.co/rows"

CONLL_ID2LABEL = {0: "O", 1: "B-ORG", 2: "B-MISC", 3: "B-PER", 4: "I-PER",
                  5: "B-LOC", 6: "I-ORG", 7: "I-MISC", 8: "I-LOC"}
LETTERS = ["A", "B", "C", "D", "E", "F"]


def fetch_rows(dataset, config, split, offset, length):
    """One page of rows from the datasets-server API, with light retry."""
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                 "split": split, "offset": offset, "length": length})
    url = f"{ROWS_API}?{qs}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            return [row["row"] for row in data.get("rows", [])]
        except Exception:  # noqa: BLE001 - transient rate-limit / empty body
            time.sleep(1.5 * (attempt + 1))
    return []


# ---- per-dataset adapters: raw row -> normalized task (or None to skip) ----

def adapt_trivia(row, i):
    ans = row.get("answer") or {}
    aliases = ans.get("aliases") or []
    if ans.get("value"):
        aliases = [ans["value"]] + aliases
    aliases = sorted({a for a in aliases if a}, key=len, reverse=True)
    if not row.get("question") or not aliases:
        return None
    return {"category": "factual", "prompt": row["question"].strip(),
            "grader": "alias", "aliases": aliases}


def adapt_gsm8k(row, i):
    ans = row.get("answer", "")
    if "####" not in ans:
        return None
    tail = ans.split("####")[-1].strip().replace(",", "")
    try:
        val = float(tail)
    except ValueError:
        return None
    return {"category": "math", "prompt": row["question"].strip(),
            "grader": "numeric", "answer_value": val}


def adapt_sst2(row, i):
    s = (row.get("sentence") or "").strip()
    if not s or row.get("label") not in (0, 1):
        return None
    label = "positive" if row["label"] == 1 else "negative"
    return {"category": "sentiment",
            "prompt": f"Classify the sentiment of this movie review sentence: \"{s}\"",
            "grader": "label", "expected_label": label}


def adapt_xsum(row, i):
    doc = (row.get("document") or "").strip()
    ref = (row.get("summary") or "").strip()
    if not doc or not ref or len(doc) > 6000:  # cap input cost
        return None
    return {"category": "summarisation",
            "prompt": f"Summarise the following article in a single sentence:\n\n{doc}",
            "grader": "judge", "reference": ref}


def adapt_conll(row, i):
    tokens, tags = row.get("tokens") or [], row.get("tags") or []
    if not tokens or len(tokens) != len(tags):
        return None
    # reconstruct entity surface strings from BIO tags
    entities, cur = [], []
    for tok, tid in zip(tokens, tags):
        lab = CONLL_ID2LABEL.get(tid, "O")
        if lab.startswith("B-"):
            if cur:
                entities.append(" ".join(cur))
            cur = [tok]
        elif lab.startswith("I-") and cur:
            cur.append(tok)
        else:
            if cur:
                entities.append(" ".join(cur))
            cur = []
    if cur:
        entities.append(" ".join(cur))
    entities = sorted({e for e in entities if e})
    if not entities:
        return None
    sentence = " ".join(tokens)
    return {"category": "ner",
            "prompt": f"Identify all named entities (people, organizations, "
                      f"locations, misc) in this sentence: {sentence}",
            "grader": "entities", "entities": entities}


def adapt_logiqa(row, i):
    ctx = (row.get("context") or "").strip()
    q = (row.get("query") or "").strip()
    opts = row.get("options") or []
    ci = row.get("correct_option")
    if not q or len(opts) < 2 or not isinstance(ci, int) or ci >= len(opts):
        return None
    labeled = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
    prompt = (f"{ctx}\n\n{q}\n\n{labeled}\n\n"
              f"Answer with the letter of the correct option only.")
    return {"category": "logic", "prompt": prompt.strip(),
            "grader": "mcq", "options": opts, "answer_index": ci}


def adapt_humaneval(row, i):
    entry = row.get("entry_point")
    if not row.get("prompt") or not row.get("test") or not entry:
        return None
    return {"category": "code_gen",
            "prompt": "Complete this Python function. Output only the full function.\n\n"
                      + row["prompt"],
            "grader": "code", "entry_point": entry,
            "tests": [row["test"], f"check({entry})"]}


def adapt_humanevalpack(row, i):
    entry = row.get("entry_point")
    decl, buggy = row.get("declaration", ""), row.get("buggy_solution", "")
    if not entry or not row.get("test") or not buggy:
        return None
    buggy_program = decl + buggy
    return {"category": "code_debug",
            "prompt": "The following Python function has a bug. Fix it and output "
                      "only the corrected full function.\n\n```python\n"
                      + buggy_program + "\n```",
            "grader": "code", "entry_point": entry,
            "tests": [row["test"], f"check({entry})"]}


SPECS = {
    "factual":       ("mandarjoshi/trivia_qa", "rc.nocontext",     "validation", adapt_trivia),
    "math":          ("openai/gsm8k",          "main",             "test",       adapt_gsm8k),
    "sentiment":     ("stanfordnlp/sst2",      "default",          "validation", adapt_sst2),
    "summarisation": ("EdinburghNLP/xsum",     "default",          "validation", adapt_xsum),
    "ner":           ("tner/conll2003",        "conll2003",        "test",       adapt_conll),
    "logic":         ("lucasmccabe/logiqa",    "default",          "test",       adapt_logiqa),
    "code_gen":      ("openai/openai_humaneval", "openai_humaneval", "test",     adapt_humaneval),
    "code_debug":    ("bigcode/humanevalpack", "python",           "test",       adapt_humanevalpack),
}


def build_category(cat, n):
    ds, cfg, split, fn = SPECS[cat]
    out, offset = [], 0
    while len(out) < n and offset < n + 200:
        page = fetch_rows(ds, cfg, split, offset, min(100, n * 2))
        if not page:
            break
        for j, raw in enumerate(page):
            task = fn(raw, offset + j)
            if task:
                task["task_id"] = f"b_{cat}_{len(out)}"
                out.append(task)
                if len(out) >= n:
                    break
        offset += len(page)
    print(f"  {cat:<15} {len(out):>3}/{n}  ({ds})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="samples per category")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "bench_set.json"))
    ap.add_argument("--category", default=None, help="only this category")
    args = ap.parse_args()

    cats = [args.category] if args.category else list(SPECS)
    all_tasks = []
    print(f"fetching {args.n}/category from public benchmarks...")
    for cat in cats:
        all_tasks.extend(build_category(cat, args.n))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {len(all_tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
