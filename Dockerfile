# Judging VM is linux/amd64. On Apple Silicon build with:
#   docker buildx build --platform linux/amd64 -t <img> --push .
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY app/ ./app/

# Strategy profile baked at build time (kestrel-style A/B): pick token/accuracy
# knobs without code changes, then tag each image by profile. Build with e.g.
#   docker buildx build --platform linux/amd64 --build-arg PROFILE=lean -t <img>:lean --push .
# A0 = safe baseline; lean = summarisation output control; floor = experimental.
ARG PROFILE=A0
ENV AGENT_PROFILE=${PROFILE}

# The harness mounts /input and /output at runtime.
# Env (FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS) is injected at runtime —
# never bundled into the image.
ENTRYPOINT ["python", "-m", "app.main"]
