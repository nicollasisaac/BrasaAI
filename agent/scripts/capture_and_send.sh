#!/usr/bin/env bash
# capture_and_send.sh
# Requires: curl, jq
# 1) Grab the live screenshot (base64) from the local host bridge
# 2) Pipe to OmniParser VM /api/parse without writing files
set -euo pipefail

WINDOWS_HOST_URL="${WINDOWS_HOST_URL:-http://127.0.0.1:8006}"
OMNIPARSER_URL="${OMNIPARSER_URL:-http://127.0.0.1:7860/api/parse}"

B64_IMG=$(curl -s "${WINDOWS_HOST_URL}/screenshot" | jq -r '.image_base64')

jq -n --arg img "$B64_IMG" '{
  image_base64: $img,
  box_threshold: 0.05,
  iou_threshold: 0.10,
  use_paddleocr: true,
  imgsz: 1280
}' | curl -s -X POST -H "Content-Type: application/json" -d @- "${OMNIPARSER_URL}"
echo
