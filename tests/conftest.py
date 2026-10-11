import os

os.environ.setdefault("WARMUP", "0")  # tests must not preload models or ping Ollama at API startup
