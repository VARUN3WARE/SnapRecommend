DevOps
------
- Contains Docker and docker-compose files to run the stack locally.
- For now: do not enable CI/CD automatically. Build and run manually:

```bash
docker build -t snaprecommend:latest .
docker-compose up -d
```

- Initialize DB: `docker-compose exec api python -c "from api.db import init_db; init_db()"`
