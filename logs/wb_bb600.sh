#!/usr/bin/env bash
# WinoBias + BiasBuster first read: 600 items per task (whole groups), 3 seeds, one model per invocation.
set -u
cd "$(dirname "$0")/.."
T=winobias,bb_framing,bb_group_attribution,bb_status_quo
for m in bosun-v3.1-0.6b qwen3-0.6b vega-u-0.8b qwen3.5-0.8b bosun-v3.1 qwen3-1.7b; do
  echo "=== $(date +%H:%M) $m"
  uv run python -m dbias.run --models "$m" --tasks "$T" --limit 600 > "logs/$m.wbbb600.log" 2>&1 || echo "!!! $m failed"
  tr '\r' '\n' < "logs/$m.wbbb600.log" | grep -E " rows |⚠|Error" | sed 's/^/    /'
done
echo "=== $(date +%H:%M) winobias/biasbuster first read done"
