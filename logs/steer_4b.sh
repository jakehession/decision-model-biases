#!/usr/bin/env bash
# (1) BiasBuster-gender steering on Qwen3.5-0.8B / Vega-0.8B, gender vs random direction; (2) Vega-4B + Qwen3.5-4B on every task.
set -u
cd "$(dirname "$0")/.."
run() {  # name, log tag, extra args...
  local m=$1 tag=$2; shift 2
  echo "=== $(date +%H:%M) $m"
  uv run python -m dbias.run --models "$m" "$@" > "logs/$m.$tag.log" 2>&1 || echo "!!! $m failed"
  tr '\r' '\n' < "logs/$m.$tag.log" | grep -E " rows |⚠|Error" | sed 's/^/    /'
}
for d in bbg rand; do
  for a in -6 -3 3 6; do
    run "vega-u-0.8b-$d@a=$a" steer --tasks bb_group_attribution --limit 600 --seeds 0
    run "qwen3.5-0.8b-$d@a=$a" steer --tasks bb_group_attribution --limit 600 --seeds 1000,1001
  done
done
T=discrim_eval,winobias,bb_framing,bb_group_attribution,bb_status_quo,bbq
run vega-u-4b all4b --tasks "$T" --limit 600 --task-limit bbq=6600
run qwen3.5-4b all4b --tasks "${T#discrim_eval,}" --limit 600 --task-limit bbq=6600
echo "=== $(date +%H:%M) steering + 4B runs done"
