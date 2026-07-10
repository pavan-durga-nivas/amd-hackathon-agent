# Judging VM is linux/amd64 (4GB RAM, 2 vCPU, CPU-only). On Apple Silicon build:
#   docker buildx build --platform linux/amd64 -t <img> --push .
FROM python:3.11-slim

WORKDIR /app

# Runtime libs for the llama.cpp CPU wheel (OpenMP + C++ runtime).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Deps first for layer caching. llama-cpp-python CPU wheels come from the
# maintainer's prebuilt index (avoids a slow, fragile source build); everything
# else (openai, fastapi, hf_hub, ...) resolves from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Bake the local model (Qwen2.5-3B Q4_K_M, ~1.8GB) into the image so the
# container is self-contained and boots without network. Total image ~2.3GB.
ARG LOCAL_MODEL_HF=bartowski/Qwen2.5-3B-Instruct-GGUF
ARG LOCAL_MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf
RUN python -c "import os,shutil; from huggingface_hub import hf_hub_download; \
os.makedirs('/models',exist_ok=True); \
shutil.copy(hf_hub_download('${LOCAL_MODEL_HF}','${LOCAL_MODEL_FILE}'), '/models/model.gguf')"

# App code.
COPY app/ ./app/

# Strategy profile baked at build time (kestrel-style A/B): pick token/accuracy
# knobs without code changes, then tag each image by profile.
ARG PROFILE=A0
ENV AGENT_PROFILE=${PROFILE} \
    LOCAL_MODEL_PATH=/models/model.gguf \
    LOCAL_MODEL_ID=local

# The harness mounts /input and /output at runtime.
# Env (FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS) is injected at
# runtime — never bundled. launch.py boots the local model then runs the agent.
ENTRYPOINT ["python", "-m", "app.launch"]
