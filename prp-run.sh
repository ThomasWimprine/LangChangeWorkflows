#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$ROOT/.venv}"
PY="$VENV_DIR/bin/python"

# Ensure virtualenv exists (prefer uv sync to honor uv.lock)
if [[ ! -x "$PY" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --frozen
  else
    echo "Missing venv at $PY. Create one with: python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
  fi
fi

# Default project root to the caller's CWD unless explicitly provided
export PRP_PROJECT_ROOT="${PRP_PROJECT_ROOT:-$PWD}"

# Prefer uv to ensure deps are in sync with uv.lock; fall back to python
if command -v uv >/dev/null 2>&1; then
  UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    uv run --frozen --python "$PY" -- "$ROOT/headless_operation.py" "$@"
else
  exec "$PY" "$ROOT/headless_operation.py" "$@"
fi
