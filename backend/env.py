"""
NagarAI -- .env loader, stdlib only.

serve.py promises zero pip installs, so this cannot depend on python-dotenv
(which is unused in the codebase despite being listed in requirements.txt).
Existing environment variables always win over the file, so an `export` in
your shell still overrides .env.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(BASE_DIR, ".env")


def load(path=ENV_FILE):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
