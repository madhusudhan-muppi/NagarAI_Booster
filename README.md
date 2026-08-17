# NagarAI — Civic Complaint Intelligence Engine

An AI layer for municipal grievance management. Citizens file complaints naturally (voice/photo/text in regional languages), officials get a deduplicated, categorized, prioritized, mapped queue with explainable ranking.

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/madhusudhan-muppi/NagarAI_Booster
cd NagarAI_Booster
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
bash download_models.sh
```

### 2. Start Backend
```bash
cd backend
uvicorn main:app --reload
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Run Telegram Bot
```bash
python bot/telegram_bot.py
```

## Project Structure
- `backend/` — FastAPI server, ML inference
- `frontend/` — React + Leaflet dashboard
- `bot/` — Telegram intake bot
- `data/` — Seed data and fixtures
- `schemas.py` — Shared Pydantic models

## Architecture
- **ASR:** faster-whisper + AI4Bharat IndicConformer
- **Vision:** CLIP zero-shot + severity estimation
- **Dedup:** Sentence embeddings + geo-distance clustering
- **Storage:** PostgreSQL + pgvector (Supabase) or SQLite
- **Dashboard:** React + Leaflet + Mapbox

See `docs/` for detailed documentation.

## Team
- Person 1: Telegram bot & intake
- Person 2: ASR + LLM extraction
- Person 3: Vision classification
- Person 4: Dedup + priority engine
- Person 5: Dashboard
- Person 6: Integration & deployment
