#!/usr/bin/env bash
# Run openenv validation against the running server
set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Validating OpenEnv environment at $BASE_URL ..."
openenv validate --url "$BASE_URL"
echo "Validation passed."
