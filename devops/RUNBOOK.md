Runbook (Development & Smoke Tests)

Quick dev run (sqlite, in-memory cache):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from api.db import init_db; init_db()"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Makefile shortcuts:

```bash
make setup
make init-db
make run
make test
make smoke
```

Check health and metrics:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics  # Prometheus metrics (if enabled)
```

Manual Docker (if you have Docker installed):

```bash
docker build -t snaprecommend:latest .
docker-compose up -d
# Initialize DB inside container
docker-compose exec api python -c "from api.db import init_db; init_db()"
# View logs
docker-compose logs -f api
```

Redis/Postgres staging via docker-compose (example `.env` values):

```text
DATABASE_URL=postgresql://snapuser:snappass@postgres:5432/snaprecommend
CACHE_TYPE=redis
REDIS_HOST=redis
REDIS_PORT=6379
```

Smoke test endpoints (after stack is up):

```bash
curl http://localhost:8000/health
curl http://localhost:8000/cache/stats
curl http://localhost:8000/metrics
```

Notes / Troubleshooting:
- If `/metrics` returns 500, check `prometheus_client` availability and logs.
- If index not ready, run pipeline scripts: `python pipeline/embed_items.py && python pipeline/build_index.py`.
- If Docker reports permission denied on `/var/run/docker.sock`, add your user to the `docker` group or run the command with `sudo`.
If you see "permission denied" when Docker tries to access `/var/run/docker.sock`, run:

```bash
# Add your user to the docker group (preferred)
sudo usermod -aG docker $USER
# Either log out and log back in, or run the following to pick up the new group immediately in this shell:
newgrp docker

# Or run commands with sudo (less recommended):
#sudo docker-compose -f docker-compose.staging.yml up -d --build
```

If you still see permission errors, ensure `/var/run/docker.sock` is owned by group `docker`:

```bash
ls -l /var/run/docker.sock
sudo chown root:docker /var/run/docker.sock
sudo chmod 660 /var/run/docker.sock
```
- To re-enable CI/CD: add the workflow file back into `.github/workflows/ci.yml` and push.
