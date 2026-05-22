API
---
- Purpose: FastAPI application serving recommendation endpoints.
- Key endpoints: `/recommend/image`, `/recommend/text`, `/recommend/hybrid`, `/product/{id}`, `/interaction`, `/metrics/summary`, `/cache/stats`.
- Run locally: `uvicorn api.main:app --reload --port 8000`.
- Notes: Logging and Prometheus metrics are wired via environment variables `LOG_FORMAT` and `LOG_LEVEL`.
