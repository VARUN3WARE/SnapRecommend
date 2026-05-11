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

## 6. Docker & Containerization

### Build Docker Image

```bash
docker build -t snaprecommend:latest .
```

### Run API in Docker

```bash
docker run -p 8000:8000 snaprecommend:latest
```

### Run Full Stack with Docker Compose

```bash
# Start all services (API, UI, optional PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

Services:
- **API**: `http://localhost:8000` (with /health endpoint)
- **Streamlit UI**: `http://localhost:8501`
- **PostgreSQL**: `localhost:5432` (optional, for production)

## 7. CI/CD Pipeline

GitHub Actions automatically:
- **Tests**: Runs pytest on Python 3.10, 3.11, 3.12
- **Linting**: Flake8, mypy, black, isort, pylint
- **Code Quality**: Type checking and coverage reports
- **Docker Build**: Builds image on main branch push
- **Artifacts**: Uploads coverage reports to Actions

Trigger: Every push and pull request to `main` or `develop` branches.

View workflow: `.github/workflows/ci.yml`

## 8. Performance Metrics

The system tracks:
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)
- **Diversity**: Pairwise distance between recommendations
- **Coverage**: Catalog coverage percentage
- **Latency**: Query response time (ms)
- **Cache Hit Rate**: Retrieval cache performance

Endpoint: `GET /metrics/summary?last_n=100`

## 9. Database Configuration

### SQLite (Development Default)

The system defaults to SQLite in `data/snaprecommend.db` for development:
```bash
# No configuration needed; works out-of-the-box
python pipeline/simulate_users.py
# ...
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### PostgreSQL (Production)

#### Setup PostgreSQL

```bash
# Install PostgreSQL (macOS)
brew install postgresql@15

# Or Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL
pg_ctl -D /usr/local/var/postgres start  # macOS
# or
sudo service postgresql start  # Linux
```

#### Configure Connection

Create a PostgreSQL database:
```bash
createdb snaprecommend_prod
```

Set the environment variable:
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/snaprecommend_prod"
```

Or update `.env`:
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/snaprecommend_prod
```

#### Run with PostgreSQL

```bash
# Initialize database schema
python -c "from api.db import init_db; init_db()"

# Run data pipeline
python pipeline/simulate_users.py
python pipeline/embed_items.py
# ... rest of pipeline

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Docker Compose with PostgreSQL

```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://snapuser:snappass@postgres:5432/snaprecommend
PHASE_MODE=phase2
USE_RANKER=true
EOF

# Start services
docker-compose up -d

# Initialize database
docker-compose exec api python -c "from api.db import init_db; init_db()"

# Run pipeline inside container
docker-compose exec api python pipeline/simulate_users.py
docker-compose exec api python pipeline/embed_items.py
# ... etc
```

### Connection String Formats

| Database | URL Format |
|----------|-----------|
| SQLite (dev) | `sqlite:///data/snaprecommend.db` |
| PostgreSQL | `postgresql://user:password@host:port/dbname` |
| PostgreSQL (psycopg3) | `postgresql+psycopg://user:password@host:port/dbname` |

## 10. Deployment

### Production Checklist

- [ ] Set `PHASE_MODE = "phase2"`, `USE_RANKER = True` in config
- [ ] Update `.env` with `DATABASE_URL` pointing to PostgreSQL
- [ ] Enable CLIP via `pip install git+https://github.com/openai/CLIP.git`
- [ ] Set up FAISS GPU if available
- [ ] Run health check: `curl http://{host}:8000/health`
- [ ] Test API endpoints before load testing
- [ ] Configure PostgreSQL backups and connection pooling (PgBouncer)

### Scaling Considerations

- **Horizontal**: Run multiple API instances behind load balancer (e.g., Nginx, HAProxy)
- **Caching**: Move from in-memory to Redis for distributed cache across multiple API instances
- **Database**: PostgreSQL with connection pooling for multi-node consistency
- **Search**: FAISS GPU for faster retrieval at scale
- **Monitoring**: Set up Prometheus + Grafana for metrics collection and dashboards

## Notes

- If CLIP is unavailable, the code uses deterministic embeddings to keep the pipeline runnable.
- If FAISS is unavailable, a cosine-similarity numpy retriever is used.
- This keeps end-to-end flow executable on most machines while preserving architecture.
