"""Entry point: read /input/tasks.json -> route -> solve -> write /output/results.json.

PoC pipeline (Phase 0/1):
  1. Load tasks and env config.
  2. Route each task to a category (local, free) and pick a model tier.
  3. Call Fireworks concurrently, with per-task and global time budgets.
  4. Always write a valid results.json with an answer for every task_id.

Later phases add: local formal solvers (sympy/z3/AST), input compression,
free verification + escalation, and calibrated route tables.
"""

import asyncio
import json
import os
import sys
import time

from app import config, router
from app.fireworks import FireworksClient, TokenMeter

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Time budgets (seconds). Rules: total <=10 min, per-request <30s.
GLOBAL_DEADLINE = float(os.environ.get("GLOBAL_DEADLINE_S", "555"))  # ~9m15s safety margin
# Whole-task wall budget (primary + any fallback COMBINED) — kept safely under the
# 30s per-request hard limit so a retry can never push one task past it.
PER_TASK_BUDGET = float(os.environ.get("PER_TASK_BUDGET_S", "28"))
# Cap on any single model call, so the primary can't consume the whole budget and
# leave no room for a fallback attempt.
PER_ATTEMPT_TIMEOUT = float(os.environ.get("PER_ATTEMPT_TIMEOUT_S", "22"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array")
    return tasks


def write_results(path: str, results: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic; avoids half-written malformed JSON


async def solve_task(task: dict, client: FireworksClient, allowed_models: list,
                     route_table: dict, deadline: float) -> dict:
    """Route + answer a single task, retrying on error/empty with a fallback
    model. Never raises; always returns a valid record."""
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category = router.detect_category(prompt)
    # Capability-first ordered picks from ALLOWED_MODELS: primary + one fallback
    # (a second capable model for robustness against 500s/empty replies).
    candidates = config.candidate_models(
        category, allowed_models, input_tokens=router.estimate_tokens(prompt))[:2]

    # Per-task wall deadline: the whole task (all attempts) must fit under the 30s
    # per-request limit, so retries can't accumulate past it.
    task_deadline = time.monotonic() + PER_TASK_BUDGET
    for model in candidates:
        now = time.monotonic()
        # Bound this attempt by whichever runs out first: the per-attempt cap, the
        # remaining per-task budget, or the global deadline.
        timeout = min(PER_ATTEMPT_TIMEOUT, task_deadline - now, deadline - now)
        if timeout <= 1.0:  # no room left for a meaningful attempt
            break
        try:
            sys_prompt, out_cap, stop = config.output_plan(category, prompt)
            answer, _ptok, _ctok = await client.complete(
                model=model,
                system=sys_prompt,
                user=prompt,
                max_tokens=out_cap,
                temperature=config.TEMPERATURE[category],
                timeout=timeout,
                reasoning_effort=config.REASONING_EFFORT.get(category, "none"),
                stop=stop,
            )
            if answer and answer.strip():
                return {"task_id": task_id, "answer": answer}
        except Exception as e:  # noqa: BLE001 - try the fallback model next
            print(f"[warn] task {task_id} model {model} failed: {e}", file=sys.stderr)

    return {"task_id": task_id, "answer": ""}


async def run() -> int:
    start = time.monotonic()
    deadline = start + GLOBAL_DEADLINE

    tasks = load_tasks(INPUT_PATH)
    allowed_models = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    route_table = config.load_route_table()

    meter = TokenMeter()
    client = FireworksClient(meter)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def bounded(t):
        async with sem:
            return await solve_task(t, client, allowed_models, route_table, deadline)

    results = await asyncio.gather(*(bounded(t) for t in tasks))

    write_results(OUTPUT_PATH, results)
    elapsed = time.monotonic() - start
    print(f"[done] {len(results)} tasks | {meter.calls} calls | "
          f"{meter.total} tokens (in={meter.prompt_tokens} out={meter.completion_tokens}) | "
          f"{elapsed:.1f}s", file=sys.stderr)
    return 0


def main():
    try:
        code = asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
