#!/bin/bash
# Pre-download models. Run this at home, on good wifi, before the event.
set -e

echo "Fetching sentence embedding model (~90 MB)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Fetching Whisper small (~460 MB)..."
python -c "from faster_whisper import WhisperModel; WhisperModel('small')" || \
  echo "  faster-whisper not installed yet -- skipping"

echo
echo "Done. Models cached in ~/.cache/huggingface and ~/.cache/whisper"
echo "Verify: python3 backend/ml/dedup_engine.py data/seed_complaints.json"
