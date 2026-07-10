"""Measuring stick: run the agent over the dev set, grade it, report the
two scored axes — accuracy (gate) and total tokens (rank) — per category.

Usage:
    python -m eval.evaluate                 # full dev set, LLM-judge on
    python -m eval.evaluate --no-judge      # deterministic-only (0 judge tokens)
    python -m eval.evaluate --category math # single category
    python -m eval.evaluate --limit 2       # first N per category (quick/cheap)

The agent's own tokens are the score proxy. Judge tokens are reported
separately (they are a dev-only cost, not part of the leaderboard score).
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


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def select_tasks(tasks, category=None, limit=None):
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if limit:
        seen = defaultdict(int)
        out = []
        for t in tasks:
            if seen[t["category"]] < limit:
                out.append(t)
                seen[t["category"]] += 1
        tasks = out
    return tasks


async def run_agent(tasks, allowed, route_table, client, force_model=None):
    from app import config, router
    sem = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "6")))
    records = {}

    async def one(t):
        async with sem:
            prompt = t["prompt"]
            cat = router.detect_category(prompt)
            model = force_model or config.resolve_model(
                cat, route_table, allowed, input_tokens=router.estimate_tokens(prompt))
            rec = {"detected": cat, "model": model, "answer": "", "ptok": 0, "ctok": 0,
                   "latency": 0.0, "err": None}
            t0 = time.monotonic()
            try:
                sys_prompt, out_cap, stop = config.output_plan(cat, prompt)
                ans, p, c = await client.complete(
                    model=model, system=sys_prompt, user=prompt,
                    max_tokens=out_cap, temperature=config.TEMPERATURE[cat],
                    timeout=28.0,
                    reasoning_effort=config.REASONING_EFFORT.get(cat, "none"),
                    stop=stop)
                rec.update(answer=ans, ptok=p, ctok=c)
            except Exception as e:  # noqa: BLE001
                rec["err"] = str(e)[:100]
            rec["latency"] = round(time.monotonic() - t0, 3)
            records[t["task_id"]] = rec

    await asyncio.gather(*(one(t) for t in tasks))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="deterministic graders only")
    ap.add_argument("--judge-all", action="store_true",
                    help="LLM-judge the soft categories (contains_all/judge graders), "
                         "keep ground-truth graders for math/code/ner/sentiment")
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max tasks per category")
    ap.add_argument("--judge-model", default="accounts/fireworks/models/deepseek-v4-pro")
    ap.add_argument("--set", default="dev_set.json",
                    help="task file under eval/ (e.g. bench_set.json for real benchmarks)")
    ap.add_argument("--metrics-out", default=None,
                    help="write per-task metrics (tokens, latency, pass) to this JSON path")
    ap.add_argument("--model", default=None,
                    help="force this model for every task (A/B a route override); "
                         "short name ok, e.g. deepseek-v4-flash")
    ap.add_argument("--show", action="store_true", help="print each answer")
    args = ap.parse_args()

    load_dotenv()
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if not os.environ.get(var):
            print(f"[error] {var} not set (copy .env.example -> .env).", file=sys.stderr)
            sys.exit(1)

    from app import config
    from app.fireworks import FireworksClient, TokenMeter
    import eval.judge as judge

    tasks = json.load(open(os.path.join(ROOT, "eval", args.set)))
    tasks = select_tasks(tasks, args.category, args.limit)

    allowed = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    route_table = config.load_route_table()

    meter = TokenMeter()
    client = FireworksClient(meter)

    # Judge client (separate, sync, not counted in agent tokens).
    judge_client = None
    if not args.no_judge or args.judge_all:
        from openai import OpenAI
        judge_client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"],
                              base_url=os.environ["FIREWORKS_BASE_URL"])

    print(f"routing: calibrated route_table ({len(allowed)} models allowed)")
    print(f"tasks: {len(tasks)}   judge: {'off' if args.no_judge else args.judge_model}\n")

    force_model = None
    if args.model:
        force_model = args.model if args.model.startswith("accounts/") else \
            "accounts/fireworks/models/" + args.model
        print(f"model override: every task -> {force_model}")

    t0 = time.monotonic()
    records = asyncio.run(run_agent(tasks, allowed, route_table, client, force_model))
    elapsed = time.monotonic() - t0

    # Grade + aggregate.
    per_cat = defaultdict(lambda: {"n": 0, "pass": 0, "tok": 0, "route_ok": 0, "lat": []})
    rows = []
    metrics = []  # full per-task record for --metrics-out
    def build_reference(t):
        """A rubric string for the LLM-judge, from whatever reference the task has."""
        g = t["grader"]
        if g == "judge":
            if t.get("reference"):
                return ("A correct answer is a faithful summary conveying the same key "
                        f"facts as this reference summary: {t['reference']}")
            return t.get("rubric", "")
        if g == "contains_all":
            return "A correct answer must convey/include: " + ", ".join(t["must_include"])
        if g == "numeric":
            return f"The correct answer is {t['answer_value']}."
        if g == "label":
            return f"The correct sentiment classification is {t['expected_label']}."
        if g == "entities":
            return "Must identify these entities: " + ", ".join(t["entities"])
        return ""

    # In --judge-all, LLM-judge the soft graders (fuzzy intent); keep ground-truth
    # graders (numeric/label/entities/code) which are objectively verifiable.
    SOFT = {"contains_all", "judge"}
    for t in tasks:
        rec = records[t["task_id"]]
        if rec["err"]:
            passed, detail = False, f"ERROR: {rec['err']}"
        elif judge_client and (t["grader"] == "judge" or (args.judge_all and t["grader"] in SOFT)):
            passed, detail = judge.grade_judge(t["prompt"], rec["answer"],
                                               build_reference(t), judge_client, args.judge_model)
        else:
            passed, detail = judge.grade(t, rec["answer"], judge_client, args.judge_model)
        cat = t["category"]
        tok = rec["ptok"] + rec["ctok"]
        pc = per_cat[cat]
        pc["n"] += 1
        pc["pass"] += int(passed)
        pc["tok"] += tok
        pc["route_ok"] += int(rec["detected"] == cat)
        pc["lat"].append(rec["latency"])
        rows.append((t["task_id"], cat, rec["detected"], passed, tok, detail, rec["answer"]))
        metrics.append({
            "task_id": t["task_id"], "category": cat, "grader": t["grader"],
            "detected": rec["detected"], "route_ok": rec["detected"] == cat,
            "model": rec["model"], "passed": bool(passed),
            "prompt_tokens": rec["ptok"], "completion_tokens": rec["ctok"],
            "total_tokens": tok, "latency_s": rec["latency"],
            "judged_by_llm": t["grader"] == "judge", "detail": detail,
            "error": rec["err"],
        })

    def pctl(xs, q):
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * len(s)))]

    # Report.
    print(f"{'category':<15} {'acc':>6} {'route':>6} {'tokens':>8} {'avgtok':>7} {'avglat':>7} {'p95lat':>7}")
    print("-" * 62)
    tot_n = tot_pass = tot_tok = tot_route = 0
    all_lat = []
    for cat in sorted(per_cat):
        pc = per_cat[cat]
        acc = pc["pass"] / pc["n"]
        route = pc["route_ok"] / pc["n"]
        avg_lat = sum(pc["lat"]) / len(pc["lat"])
        print(f"{cat:<15} {acc:>5.0%} {route:>5.0%} {pc['tok']:>8} {pc['tok']//pc['n']:>7} "
              f"{avg_lat:>6.2f}s {pctl(pc['lat'], 0.95):>6.2f}s")
        tot_n += pc["n"]; tot_pass += pc["pass"]; tot_tok += pc["tok"]; tot_route += pc["route_ok"]
        all_lat += pc["lat"]
    print("-" * 62)
    print(f"{'TOTAL':<15} {tot_pass/tot_n:>5.0%} {tot_route/tot_n:>5.0%} {tot_tok:>8} {tot_tok//tot_n:>7} "
          f"{sum(all_lat)/len(all_lat):>6.2f}s {pctl(all_lat, 0.95):>6.2f}s")
    print(f"\nagent tokens (score proxy): {meter.total}  (in={meter.prompt_tokens} out={meter.completion_tokens})")
    print(f"per-request latency: avg {sum(all_lat)/len(all_lat):.2f}s  p95 {pctl(all_lat,0.95):.2f}s  max {max(all_lat):.2f}s")
    print(f"wall time: {elapsed:.1f}s")

    if args.metrics_out:
        summary = {cat: {"n": pc["n"], "accuracy": round(pc["pass"]/pc["n"], 4),
                         "route_acc": round(pc["route_ok"]/pc["n"], 4),
                         "total_tokens": pc["tok"], "avg_tokens": round(pc["tok"]/pc["n"], 1),
                         "avg_latency_s": round(sum(pc["lat"])/len(pc["lat"]), 3),
                         "p95_latency_s": round(pctl(pc["lat"], 0.95), 3)}
                   for cat, pc in per_cat.items()}
        payload = {"set": args.set, "n_tasks": tot_n,
                   "overall": {"accuracy": round(tot_pass/tot_n, 4),
                               "route_acc": round(tot_route/tot_n, 4),
                               "agent_tokens": meter.total,
                               "prompt_tokens": meter.prompt_tokens,
                               "completion_tokens": meter.completion_tokens,
                               "avg_latency_s": round(sum(all_lat)/len(all_lat), 3),
                               "p95_latency_s": round(pctl(all_lat, 0.95), 3),
                               "max_latency_s": round(max(all_lat), 3),
                               "wall_time_s": round(elapsed, 1)},
                   "per_category": summary, "tasks": metrics}
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"metrics written -> {args.metrics_out}")

    # Failures + optional full answers.
    fails = [r for r in rows if not r[3]]
    if fails:
        print(f"\n── failures ({len(fails)}) ──")
        for tid, cat, det, _p, tok, detail, ans in fails:
            print(f"  {tid:>4} [{cat}] {detail}")
    if args.show:
        print("\n── answers ──")
        for tid, cat, det, p, tok, detail, ans in rows:
            mark = "OK" if p else "XX"
            a = (ans or "").replace("\n", " ")
            print(f"  {mark} {tid:>4} [{cat}/{det}] {tok}tok: {a[:100]}")


if __name__ == "__main__":
    main()
