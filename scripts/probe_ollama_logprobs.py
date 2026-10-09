import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402

from app import config  # noqa: E402


def main():
    client = ollama.Client(host=config.OLLAMA_HOST)
    t = time.time()
    try:
        r = client.chat(model=config.OLLAMA_MODEL, messages=[{"role": "user", "content": "Say hi in 3 words."}],
                        logprobs=True, options={"temperature": 0})
    except Exception as e:  # noqa: BLE001
        print("ERR", type(e).__name__, str(e)[:300])
        return
    lp = getattr(r, "logprobs", None)
    print(f"model={config.OLLAMA_MODEL} latency={time.time() - t:.1f}s answer={r.message.content!r}")
    print("logprobs returned:", bool(lp), "| n tokens:", len(lp) if lp else 0)
    if lp:
        print("first entry:", lp[0])


if __name__ == "__main__":
    main()
