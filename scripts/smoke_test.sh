#!/bin/bash
# Toxicity Moderation + Safe Rewrite Smoke Test

set -e

API_URL="http://localhost:8000"

echo "Checking health..."
curl -s "${API_URL}/health" | grep -q '"status":"ok"' || (echo "Health check failed" && exit 1)
echo "Health OK"

echo "Testing moderation..."
MODERATE_RESPONSE=$(curl -s -X POST "${API_URL}/moderate" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message."}')
echo "Moderation response: ${MODERATE_RESPONSE}"

echo "Testing rewrite safe (note: requires vLLM to be running)..."
REWRITE_RESPONSE=$(curl -s -X POST "${API_URL}/rewrite_safe" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message.", "n": 1}')
echo "Rewrite response: ${REWRITE_RESPONSE}"

echo "Smoke test passed!"
