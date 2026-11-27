#!/usr/bin/env bash
# Move Qwen3.5-4B's BBQ after the random-direction controls: stop its run once bb_status_quo is written,
# let rand5.sh (waiting on steer_4b.sh) go, then run Qwen3.5-4B BBQ alone.
set -u
cd "$(dirname "$0")/.."
until [ -f results/qwen3.5-4b/bb_status_quo.parquet ]; do sleep 30; done
echo "=== $(date +%H:%M) status quo written; stopping qwen3.5-4b before BBQ"
pkill -f "dbias.run --models qwen3.5-4b"
sleep 120
while pgrep -f "logs/steer_4b.sh|logs/rand5.sh" > /dev/null; do sleep 60; done
echo "=== $(date +%H:%M) qwen3.5-4b bbq"
uv run python -m dbias.run --models qwen3.5-4b --tasks bbq --task-limit bbq=6600 > logs/qwen3.5-4b.bbq6600.log 2>&1 || echo "!!! failed"
tr '\r' '\n' < logs/qwen3.5-4b.bbq6600.log | grep -E " rows |⚠|Error" | sed 's/^/    /'
echo "=== $(date +%H:%M) all done"
