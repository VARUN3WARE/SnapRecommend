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
python pipeline/prepare_training_data.py
python pipeline/train_two_tower.py --epochs 2 --batch-size 64 --device cpu
python pipeline/train_ranker.py --epochs 2 --batch-size 64 --device cpu
```

To enable Phase 2 serving after training, set `PHASE_MODE = "phase2"` and `USE_RANKER = True` in `config.py`. If you also want the sequence user encoder path, set `USER_ENCODER_MODE = "transformer"`.

## 3. Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `POST /recommend/image` - Get recommendations from an image query
- `POST /recommend/text` - Get recommendations from a text query
- `POST /recommend/hybrid` - Get recommendations from both image and text
- `GET /product/{product_id}` - Get product details
- `POST /interaction` - Log a user interaction (click, view, purchase)
- `GET /health` - Health check with model and phase status
- `GET /cache/stats` - Cache performance statistics (hits, misses, hit rate)
- `POST /cache/clear` - Clear all cached results

### Query Parameters for Phase Control

All recommendation endpoints support optional query parameters:
- `?phase_mode=phase1` or `?phase_mode=phase2` - Override phase mode per request
- `?use_ranker=true` or `?use_ranker=false` - Enable/disable ranker per request

Example:
```bash
curl -X POST "http://localhost:8000/recommend/text?phase_mode=phase2&use_ranker=true" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u00001", "query": "red shoes", "top_k": 10}'
```

### Caching Layer

- **Embedding Cache**: TTL 24 hours (86400s)
- **Retrieval Cache**: TTL 1 hour (3600s)
- Cache keys based on user_id + query_vec hash + top_k
- Reduces redundant FAISS queries and model inference
- Expected hit rate: 70%+ on typical production workloads

Check cache stats:
```bash
curl http://localhost:8000/cache/stats
```

Clear cache:
```bash
curl -X POST http://localhost:8000/cache/clear
```

## 4. Run UI

```bash
streamlit run ui/app.py
```

**UI Features:**
- Sidebar: API URL configuration
- Phase Mode selector (phase1/phase2)
- Ranker toggle (enable/disable reranking)
- Query builder: Image, Text, or Hybrid modes
- Results display with scores

## 5. Tests

```bash
pytest -q
```

## Notes

- If CLIP is unavailable, the code uses deterministic embeddings to keep the pipeline runnable.
- If FAISS is unavailable, a cosine-similarity numpy retriever is used.
- This keeps end-to-end flow executable on most machines while preserving architecture.
