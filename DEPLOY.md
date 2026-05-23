Deployment Guide (short)

1) Build and run with Docker Compose

```bash
# build images
docker build -t snaprecommend:latest .
# start stack (api, postgres, redis)
docker-compose up -d --build
# view logs
docker-compose logs -f api
```

2) Environment variables

- `DATABASE_URL` — e.g. `postgresql://snapuser:snappass@postgres:5432/snaprecommend`
- `CACHE_TYPE` — `redis` or `memory`
- `REDIS_HOST`, `REDIS_PORT` — when using Redis
- `LOG_LEVEL`, `LOG_FORMAT`

3) Initialize DB / run migrations

```bash
# from host (or inside api container)
python -c "from api.db import init_db; init_db()"
# or run your migration tool (Alembic) if configured:
# alembic upgrade head
```

4) Run smoke tests

```bash
curl http://localhost:8000/health
curl http://localhost:8000/cache/stats
curl http://localhost:8000/metrics
```

5) Notes

- If Docker is not available locally, run the app with virtualenv using `devops/RUNBOOK.md` steps.
- Re-enable CI by restoring `.github/workflows/ci.yml` when ready.
