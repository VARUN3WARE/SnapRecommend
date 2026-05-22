Retrieval
---------
- Implements retrieval index (FAISS) and caching layer (in-memory + Redis backend).
- Cache TTLs: embedding=24h, retrieval=1h. Use Redis in production by setting `CACHE_TYPE=redis` and providing Redis host/port.
- Index files: `EMBEDDINGS_PATH` and `FAISS_INDEX_PATH` (see `config.py`).
