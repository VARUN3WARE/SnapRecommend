.PHONY: setup test run lint format init-db smoke docker-build docker-up docker-down docker-logs docker-up-staging docker-down-staging

PYTHON ?= python
VENV_ACTIVATE = . .venv/bin/activate

setup:
	$(PYTHON) -m venv .venv
	$(VENV_ACTIVATE) && pip install -r requirements.txt

test:
	$(VENV_ACTIVATE) && $(PYTHON) -m pytest -q

run:
	$(VENV_ACTIVATE) && uvicorn api.main:app --host 0.0.0.0 --port 8000

init-db:
	$(VENV_ACTIVATE) && $(PYTHON) -c "from api.db import init_db; init_db()"

smoke:
	./devops/smoke_test.sh

docker-build:
	docker build -t snaprecommend:latest .

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

docker-up-staging:
	docker-compose -f docker-compose.staging.yml up -d --build

docker-down-staging:
	docker-compose -f docker-compose.staging.yml down
