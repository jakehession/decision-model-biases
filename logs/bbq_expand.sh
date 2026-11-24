#!/usr/bin/env bash
# BBQ at ~300 items per category × condition (6600 items, 3 seeds), one model per invocation, sequential.
set -u
cd "$(dirname "$0")/.."
for m in bosun-v3.1-0.6b qwen3-0.6b vega-u-0.8b qwen3.5-0.8b bosun-v3.1 qwen3-1.7b; do
  echo "=== $(date +%H:%M) $m"
  uv run python -m dbias.run --models "$m" --tasks bbq --task-limit bbq=6600 > "logs/$m.bbq6600.log" 2>&1 || echo "!!! $m failed"
  tr '\r' '\n' < "logs/$m.bbq6600.log" | grep -E " rows |⚠" | sed 's/^/    /'
done
echo "=== $(date +%H:%M) bbq expansion done"
