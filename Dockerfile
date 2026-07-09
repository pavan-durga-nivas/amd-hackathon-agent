# Judging VM is linux/amd64. On Apple Silicon build with:
#   docker buildx build --platform linux/amd64 -t <img> --push .
FROM python:3.11-slim

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY app/ ./app/

# The harness mounts /input and /output at runtime.
# Env (FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS) is injected at runtime —
# never bundled into the image.
ENTRYPOINT ["python", "-m", "app.main"]
