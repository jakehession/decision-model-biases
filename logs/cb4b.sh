#!/usr/bin/env bash
# After rand5.sh: counterbalanced option order for the order-sensitive 4B pair + Vega-0.8B (→ results_cb/), then Qwen3.5-4B BBQ.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "logs/rand5.sh" > /dev/null; do sleep 60; done
run() {  # model, out dir, tag, extra args...
  local m=$1 out=$2 tag=$3; shift 3
  echo "=== $(date +%H:%M) $m"
  uv run python -m dbias.run --models "$m" --out "$out" "$@" > "logs/$m.$tag.log" 2>&1 || echo "!!! $m failed"
  tr '\r' '\n' < "logs/$m.$tag.log" | grep -E " rows |⚠|Error" | sed 's/^/    /'
}
T=winobias,bb_framing,bb_group_attribution,bb_status_quo
for m in vega-u-4b qwen3.5-4b vega-u-0.8b; do
  run "$m" results_cb cb --tasks "$T" --limit 600 --seeds 1000,1001,1002,1003
done
run qwen3.5-4b results bbq6600 --tasks bbq --task-limit bbq=6600
echo "=== $(date +%H:%M) all done"
