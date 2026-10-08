import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core.adversarial_agent import run_attack_suite  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402


def main():
    init_db()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.ADV_N_CHUNKS
    db = SessionLocal()
    summary = run_attack_suite(db, n)
    db.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
