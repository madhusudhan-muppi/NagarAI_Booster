#!/bin/bash
# Download ML models needed for NagarAI

echo "Downloading Whisper model..."
python -c "import faster_whisper; faster_whisper.WhisperModel('small')"

echo "Downloading sentence-transformers model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Models downloaded successfully!"
