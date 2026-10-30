#!/usr/bin/env bash
# Build the CuePilot desktop web UI (apps/desktop/web) into apps/desktop/web/dist.
#
# The desktop sidecar (apps/desktop/server/__init__.py) only serves the "/"
# route when web/dist exists; dist is NOT committed to the repo, so a fresh
# checkout must run this script before launching the desktop app, otherwise
# the app starts with no UI (API routes still work).
#
# Usage:  scripts/build_web.sh
# Requires: node + npm. On failure, exit non-zero and print what is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/../apps/desktop/web"

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is not installed (required to build apps/desktop/web)" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is not installed (required to build apps/desktop/web)" >&2
  exit 1
fi
if [ ! -f "$WEB_DIR/package.json" ]; then
  echo "error: web app not found at $WEB_DIR" >&2
  exit 1
fi

cd "$WEB_DIR"
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi
npm run build

if [ ! -f "$WEB_DIR/dist/index.html" ]; then
  echo "error: build finished but dist/index.html is missing" >&2
  exit 1
fi
echo "built: $WEB_DIR/dist/index.html"
