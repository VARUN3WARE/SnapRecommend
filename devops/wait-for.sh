#!/usr/bin/env bash
# wait-for.sh: wait for services to be available on host:port
# Usage: wait-for.sh host:port [host:port ...] -- cmd args...
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 host:port [host:port ...] -- command"
  exit 2
fi

TIMEOUT=${WAIT_FOR_TIMEOUT:-60}

# Collect hosts until --
HOSTS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --) shift; break;;
    *) HOSTS+=("$1") ; shift;;
  esac
done

if [ ${#HOSTS[@]} -eq 0 ]; then
  echo "No hosts provided"
  exit 2
fi

CMD=("$@")

for hostport in "${HOSTS[@]}"; do
  host="${hostport%%:*}"
  port="${hostport##*:}"
  echo "Waiting for ${host}:${port} (timeout ${TIMEOUT}s) ..."
  start=$(date +%s)
  while :; do
    if (echo > /dev/tcp/${host}/${port}) >/dev/null 2>&1; then
      echo "${host}:${port} is available"
      break
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge ${TIMEOUT} ]; then
      echo "Timeout waiting for ${host}:${port}" >&2
      exit 3
    fi
    sleep 1
  done
done

# Execute the command
exec "${CMD[@]}"
