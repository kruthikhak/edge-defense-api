#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Edge Defense API — local dev server
# For production (Railway/Docker), use:  python -m app.main
# ---------------------------------------------------------------------------
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "🚀 Starting Edge Defense API on ${HOST}:${PORT}"
uvicorn app.main:app \
  --reload \
  --host "${HOST}" \
  --port "${PORT}"
