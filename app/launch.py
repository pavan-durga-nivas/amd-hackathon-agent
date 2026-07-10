"""Container entrypoint: boot the local model server, then run the agent.

Starts an in-container OpenAI-compatible llama.cpp server for the bundled
Qwen2.5-3B GGUF, waits until it's healthy (within the 60s ready budget), exports
LOCAL_MODEL_URL so app.main routes LOCAL_CATEGORIES to it at ZERO Fireworks
tokens, then runs the agent. If the model is absent or fails to come up in time,
we DON'T set LOCAL_MODEL_URL — the agent degrades to Fireworks-only rather than
wasting per-task time on a dead endpoint. The server is always torn down at exit.
"""

import asyncio
import os
import subprocess
import sys
import time
import urllib.request

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/model.gguf")
PORT = os.environ.get("LOCAL_MODEL_PORT", "8081")
ALIAS = os.environ.get("LOCAL_MODEL_ID", "local")
THREADS = os.environ.get("LOCAL_THREADS", "2")
CTX = os.environ.get("LOCAL_CTX", "2048")
READY_TIMEOUT_S = float(os.environ.get("LOCAL_READY_TIMEOUT_S", "50"))
_BASE = f"http://127.0.0.1:{PORT}"


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{_BASE}/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_local_server():
    """Launch the llama.cpp server subprocess; return it once healthy, else None."""
    if not os.path.exists(MODEL_PATH):
        print(f"[launch] no local model at {MODEL_PATH}; Fireworks-only", file=sys.stderr)
        return None
    proc = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server",
         "--model", MODEL_PATH, "--model_alias", ALIAS,
         "--n_threads", THREADS, "--n_ctx", CTX,
         "--host", "127.0.0.1", "--port", PORT],
        stdout=subprocess.DEVNULL, stderr=sys.stderr,
    )
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print("[launch] local server exited early; Fireworks-only", file=sys.stderr)
            return None
        if _healthy():
            os.environ["LOCAL_MODEL_URL"] = f"{_BASE}/v1"
            os.environ["LOCAL_MODEL_ID"] = ALIAS
            print(f"[launch] local model ready at {_BASE}/v1 (alias={ALIAS})", file=sys.stderr)
            return proc
        time.sleep(1)
    print(f"[launch] local server not ready in {READY_TIMEOUT_S:.0f}s; Fireworks-only",
          file=sys.stderr)
    proc.terminate()
    return None


def main():
    proc = start_local_server()
    try:
        from app.main import run
        code = asyncio.run(run())
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                proc.kill()
    sys.exit(code)


if __name__ == "__main__":
    main()
