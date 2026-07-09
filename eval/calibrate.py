"""Calibration sweep: run EVERY model in ALLOWED_MODELS across sample tasks in
each category, measure accuracy + tokens, and pick the cheapest-sufficient
model per category. Writes app/route_table.json with concrete model IDs.

Usage:
    python -m eval.calibrate                 # 2 tasks/category, all models
    python -m eval.calibrate --limit 3       # more tasks/category (more tokens)
    python -m eval.calibrate --threshold 0.8 # min accuracy to be "sufficient"
    python -m eval.calibrate --write         # write app/route_table.json

Grading is deterministic (no LLM-judge) to keep the sweep cheap; summaries use
key-term coverage as a proxy. Reasoning models that overrun the token cap and
return truncated/empty answers will correctly score low here.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CATS = ["factual", "math", "sentiment", "summarisation", "ner", "code_debug", "logic", "code_gen"]
ABBR = {"factual": "fact", "math": "math", "sentiment": "sent", "summarisation": "summ",
        "ner": "ner", "code_debug": "dbg", "logic": "logic", "code_gen": "gen"}
# Generous caps for FAIR characterization (give reasoning models room to finish),
# while still bounding cost. True tokens used are what we rank on.
SWEEP_CAP = {"factual": 400, "math": 512, "sentiment": 300, "summarisation": 400,
             "ner": 400, "code_debug": 1024, "logic": 512, "code_gen": 1024}


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def pick_tasks(tasks, limit):
    seen = defaultdict(int)
    out = []
    for t in tasks:
        if seen[t["category"]] < limit:
            out.append(t)
            seen[t["category"]] += 1
    return out


async def sweep(models, tasks, client):
    from app import config
    import eval.judge as judge
    sem = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "6")))
    # cell[(model, cat)] = list of (passed, tokens)
    cell = defaultdict(list)

    async def one(model, t):
        async with sem:
            cat = t["category"]
            try:
                ans, p, c = await client.complete(
                    model=model, system=config.SYSTEM_PROMPTS[cat], user=t["prompt"],
                    max_tokens=SWEEP_CAP[cat], temperature=config.TEMPERATURE[cat], timeout=45.0)
                passed, _ = judge.grade(t, ans)  # deterministic (no judge client)
                cell[(model, cat)].append((int(passed), p + c))
            except Exception:  # noqa: BLE001 - a failed call = 0 accuracy, count no tokens
                cell[(model, cat)].append((0, 0))

    await asyncio.gather(*(one(m, t) for m in models for t in tasks))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2, help="tasks per category")
    ap.add_argument("--threshold", type=float, default=1.0, help="min accuracy to be 'sufficient'")
    ap.add_argument("--models", default=None,
                    help="comma-separated substrings to restrict the model pool (e.g. 'flash,k2p7')")
    ap.add_argument("--write", action="store_true", help="write app/route_table.json")
    args = ap.parse_args()

    load_dotenv()
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if not os.environ.get(var):
            print(f"[error] {var} not set.", file=sys.stderr)
            sys.exit(1)

    from app.fireworks import FireworksClient, TokenMeter

    tasks = pick_tasks(json.load(open(os.path.join(ROOT, "eval", "dev_set.json"))), args.limit)
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        models = [m for m in models if any(s in m for s in subs)]
    short = lambda m: m.split("/")[-1]

    meter = TokenMeter()
    client = FireworksClient(meter)

    print(f"sweeping {len(models)} models x {len(tasks)} tasks "
          f"({args.limit}/category) = {len(models)*len(tasks)} calls\n")
    t0 = time.monotonic()
    cell = asyncio.run(sweep(models, tasks, client))
    elapsed = time.monotonic() - t0

    # Aggregate to acc + avg tokens per (model, cat).
    agg = {}  # (model,cat) -> (acc, avgtok)
    for (m, cat), rows in cell.items():
        n = len(rows)
        acc = sum(r[0] for r in rows) / n
        avgtok = sum(r[1] for r in rows) / n
        agg[(m, cat)] = (acc, avgtok)

    # Matrix print: rows models, cols categories, cell "acc%/avgtok".
    hdr = "model".ljust(24) + "".join(ABBR[c].center(11) for c in CATS)
    print(hdr)
    print("-" * len(hdr))
    for m in models:
        row = short(m).ljust(24)
        for cat in CATS:
            acc, tok = agg.get((m, cat), (0, 0))
            row += f"{int(acc*100):>3}/{int(tok):<4}".center(11)
        print(row)

    # Per-category winner: cheapest (fewest tokens) model meeting the accuracy threshold.
    print("\n── cheapest-sufficient model per category "
          f"(accuracy >= {args.threshold:.0%}) ──")
    route = {}
    for cat in CATS:
        cands = [(m, agg[(m, cat)][1]) for m in models
                 if agg.get((m, cat), (0, 0))[0] >= args.threshold]
        if cands:
            best = min(cands, key=lambda x: x[1])
            route[cat] = best[0]
            print(f"  {cat:<14} -> {short(best[0]):<22} ({int(best[1])} tok)")
        else:
            # nobody cleared the bar: fall back to best accuracy, then fewest tokens
            best = max(models, key=lambda m: (agg.get((m, cat), (0, 0))[0], -agg.get((m, cat), (0, 0))[1]))
            route[cat] = best
            a, tk = agg[(best, cat)]
            print(f"  {cat:<14} -> {short(best):<22} ({int(tk)} tok, only {a:.0%} — below threshold!)")

    print(f"\nsweep tokens spent (dev cost): {meter.total} | wall {elapsed:.0f}s")

    if args.write:
        path = os.path.join(ROOT, "app", "route_table.json")
        json.dump(route, open(path, "w"), indent=2)
        print(f"\n[written] {path} with concrete per-category model IDs")
    else:
        print("\n(dry run — pass --write to save app/route_table.json)")


if __name__ == "__main__":
    main()
