"""Local dev harness — runs the agent against a local tasks file, no Docker.

Usage:
    python -m eval.run_local                      # uses eval/sample_tasks.json
    python -m eval.run_local path/to/tasks.json

Reads .env if present (simple parser, no dependency). Prints the router's
category decision per task (free/local) and the token/accuracy-relevant summary.
"""

import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_dotenv(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_dotenv()

    tasks_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "eval", "sample_tasks.json")
    out_path = os.path.join(ROOT, "eval", "results.json")
    os.environ["INPUT_PATH"] = tasks_path
    os.environ["OUTPUT_PATH"] = out_path

    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if not os.environ.get(var):
            print(f"[error] {var} not set. Copy .env.example -> .env and fill it in.", file=sys.stderr)
            sys.exit(1)

    # Show routing decisions (local, free) before spending any tokens.
    from app import router
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    print("── routing preview (local, 0 tokens) ──")
    for t in tasks:
        print(f"  {t.get('task_id'):>4}  ->  {router.detect_category(t.get('prompt', ''))}")
    print("───────────────────────────────────────")

    from app.main import run
    code = asyncio.run(run())

    with open(out_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print("\n── answers ──")
    for r in results:
        ans = (r.get("answer") or "").replace("\n", " ")
        print(f"  {r.get('task_id'):>4}: {ans[:120]}{'…' if len(ans) > 120 else ''}")
    print(f"\n[ok] wrote {out_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
