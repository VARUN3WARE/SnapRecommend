#!/usr/bin/env bash
set -euo pipefail
BASE_URL=${1:-http://localhost:8000}

echo "Checking ${BASE_URL}/health"
if ! curl -sSf ${BASE_URL}/health >/dev/null; then
  echo "Health check failed" >&2
  exit 2
fi

echo "Checking cache stats"
if ! curl -sSf ${BASE_URL}/cache/stats >/dev/null; then
  echo "Cache stats check failed" >&2
  exit 3
fi

echo "Checking metrics endpoint"
if ! curl -sSf ${BASE_URL}/metrics >/dev/null; then
  echo "Metrics endpoint check failed" >&2
  exit 4
fi

echo "All smoke checks passed"
