#!/usr/bin/env bash
set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [[ -z "$PING_URL" ]]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

echo "[1/5] Pinging /reset"
code=$(curl -s -o /tmp/student-planner-reset.out -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{}' \
  "$PING_URL/reset" || true)
if [[ "$code" != "200" ]]; then
  echo "FAILED: /reset returned HTTP $code"
  exit 1
fi

echo "[2/5] Docker build"
docker build "$REPO_DIR"

echo "[3/5] openenv validate"
if command -v openenv >/dev/null 2>&1; then
  (cd "$REPO_DIR" && openenv validate)
else
  echo "WARNING: openenv CLI not installed, skipping openenv validate"
fi

echo "[4/5] Pytest"
(cd "$REPO_DIR" && pytest -q)

echo "[5/5] inference.py exists"
if [[ ! -f "$REPO_DIR/inference.py" ]]; then
  echo "FAILED: inference.py missing in repo root"
  exit 1
fi

echo "All checks completed."
