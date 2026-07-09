# Token-Efficient Routing Agent — AMD Developer Hackathon Act II (Track 1)

A containerized general-purpose AI agent that answers tasks across 8 capability
categories (factual, math, sentiment, summarisation, NER, code debugging, logic,
code generation) while minimizing tokens spent through Fireworks AI.

**Core idea:** a free *local* category router picks, per task, the cheapest
Fireworks model that still clears the accuracy gate, with terse answer-only
prompting to keep output tokens low. (This PoC establishes the pipeline; later
phases add local solvers, input compression, and verify-and-escalate.)

## I/O contract

- Reads tasks from `/input/tasks.json`: `[{ "task_id": "t1", "prompt": "..." }, ...]`
- Writes answers to `/output/results.json`: `[{ "task_id": "t1", "answer": "..." }, ...]`
- Exit code `0` on success, non-zero on failure.

## Runtime environment (injected by the harness — do NOT bundle)

| Variable | Purpose |
|---|---|
| `FIREWORKS_API_KEY` | Auth for Fireworks. Use the harness-provided key. |
| `FIREWORKS_BASE_URL` | All inference must route through this URL. |
| `ALLOWED_MODELS` | Comma-separated permitted model IDs (published launch day). |

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env with your dev values
python -m eval.run_local    # runs against eval/sample_tasks.json
```

`.env` is git/Docker-ignored and never shipped in the image.

## Build & push (judging VM is linux/amd64)

```bash
# On Apple Silicon (M1/M2/M3) the --platform flag is required:
docker buildx build --platform linux/amd64 -t <registry>/<image>:latest --push .
```

## Run the container locally

```bash
docker run --rm \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  -v "$PWD/eval:/input" -v "$PWD/eval:/output" \
  <image>   # reads /input/tasks.json, writes /output/results.json
```

## Layout

```
app/
  main.py          entrypoint: read -> route -> solve -> write
  router.py        local category detection (free, 0 tokens)
  config.py        per-category prompts, output caps, tier->model resolution
  fireworks.py     Fireworks client (via FIREWORKS_BASE_URL) + token meter
  route_table.json category -> tier ("cheap"|"strong"), calibrated later
eval/
  run_local.py     local harness (no Docker)
  sample_tasks.json
Dockerfile         linux/amd64
```
