#!/usr/bin/env bash
# Counterbalanced option order (each option in each slot once) for the three order-sensitive bases → results_cb/.
set -u
cd "$(dirname "$0")/.."
T=discrim_eval,winobias,bb_framing,bb_group_attribution,bb_status_quo,bbq
for m in qwen3-0.6b qwen3.5-0.8b qwen3-1.7b; do
  echo "=== $(date +%H:%M) $m"
  uv run python -m dbias.run --models "$m" --tasks "$T" --limit 600 --task-limit bbq=6600 \
    --seeds 1000,1001,1002,1003 --out results_cb > "logs/$m.cb.log" 2>&1 || echo "!!! $m failed"
  tr '\r' '\n' < "logs/$m.cb.log" | grep -E " rows |⚠|Error" | sed 's/^/    /'
done
echo "=== $(date +%H:%M) counterbalanced reruns done"
