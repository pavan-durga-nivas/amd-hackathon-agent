"""Contract smoke test — the same checks the judging harness applies.

Validates the /output/results.json contract so we catch OUTPUT_MISSING /
INVALID_RESULTS_SCHEMA locally before spending a submission slot:
  - output is a JSON array
  - one entry per input task, no missing/extra task_ids
  - every entry is an object with a task_id AND a non-empty string answer

Two modes:
    python -m eval.smoke_test                       # run the agent on the
        official practice tasks (needs .env), then validate the output
    python -m eval.smoke_test --check-only RESULTS TASKS
        # only validate an existing results.json vs a tasks.json (no run) —
        # used to check the CONTAINER's output on the host

Exits 0 iff the contract holds.
"""

import argparse
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PRACTICE = os.path.join(ROOT, "eval", "practice_tasks.json")


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def validate(results_path: str, tasks_path: str) -> bool:
    """Assert the harness output contract. Returns True iff valid."""
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    want_ids = [t["task_id"] for t in tasks]

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)  # malformed JSON -> raises -> INVALID
    except (OSError, json.JSONDecodeError) as e:
        print(f"[FAIL] cannot read/parse results.json: {e}")
        return False

    errs = []
    if not isinstance(results, list):
        print("[FAIL] results.json is not a JSON array")
        return False

    got_ids = []
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            errs.append(f"entry {i} is not an object")
            continue
        if "task_id" not in r:
            errs.append(f"entry {i} missing task_id")
        else:
            got_ids.append(r["task_id"])
        ans = r.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            errs.append(f"entry {r.get('task_id', i)} has empty/non-string answer")

    missing = set(want_ids) - set(got_ids)
    extra = set(got_ids) - set(want_ids)
    if missing:
        errs.append(f"missing task_ids: {sorted(missing)}")
    if extra:
        errs.append(f"unexpected task_ids: {sorted(extra)}")
    if len(got_ids) != len(set(got_ids)):
        errs.append("duplicate task_ids in output")

    if errs:
        print(f"[FAIL] {len(errs)} contract violation(s):")
        for e in errs:
            print(f"  - {e}")
        return False

    print(f"[PASS] contract OK: {len(results)} entries, all task_ids present, "
          f"all answers non-empty.")
    for r in results:
        a = " ".join((r["answer"] or "").split())
        print(f"  {r['task_id']:>12}: {a[:90]}{'…' if len(a) > 90 else ''}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", nargs=2, metavar=("RESULTS", "TASKS"),
                    help="validate an existing results.json vs tasks.json; no agent run")
    ap.add_argument("--tasks", default=PRACTICE, help="tasks file to run (default: practice)")
    args = ap.parse_args()

    if args.check_only:
        sys.exit(0 if validate(args.check_only[0], args.check_only[1]) else 1)

    load_dotenv()
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if not os.environ.get(var):
            print(f"[error] {var} not set (copy .env.example -> .env).", file=sys.stderr)
            sys.exit(1)

    out_path = os.path.join(ROOT, "eval", "practice_results.json")
    os.environ["INPUT_PATH"] = args.tasks
    os.environ["OUTPUT_PATH"] = out_path

    from app.main import run
    code = asyncio.run(run())
    if code != 0:
        print(f"[FAIL] agent exited non-zero ({code})")
        sys.exit(1)

    sys.exit(0 if validate(out_path, args.tasks) else 1)


if __name__ == "__main__":
    main()
