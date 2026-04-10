# Multimodal Recommender System (MVP)

This project implements the Phase 1 MVP from `Master.md`:
- CLIP-style image/text encoders (with deterministic fallback)
- User history weighted-average encoder
- Fusion layer
- FAISS retrieval (with numpy fallback)
- FastAPI recommendation endpoints
- Streamlit UI

## 1. Setup

```bash
cd recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (recommended for full quality):
```bash
pip install git+https://github.com/openai/CLIP.git
# plus FAISS GPU via conda
```

## 2. Run Data Pipeline

```bash
python pipeline/simulate_users.py
python pipeline/embed_items.py
python pipeline/build_index.py
```

## 3. Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Run UI

```bash
streamlit run ui/app.py
```

## 5. Tests

```bash
pytest -q
```

## Notes

- If CLIP is unavailable, the code uses deterministic embeddings to keep the pipeline runnable.
- If FAISS is unavailable, a cosine-similarity numpy retriever is used.
- This keeps end-to-end flow executable on most machines while preserving architecture.
