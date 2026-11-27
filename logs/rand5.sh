#!/usr/bin/env bash
# Five more random-direction controls for the Vega-0.8B gender steer (α ±3, ±6). Waits for steer_4b.sh to finish.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "logs/steer_4b.sh" > /dev/null; do sleep 60; done
for k in 1 2 3 4 5; do
  for a in -6 -3 3 6; do
    m="vega-u-0.8b-rand$k@a=$a"
    echo "=== $(date +%H:%M) $m"
    uv run python -m dbias.run --models "$m" --tasks bb_group_attribution --limit 600 --seeds 0 > "logs/$m.steer.log" 2>&1 || echo "!!! $m failed"
    tr '\r' '\n' < "logs/$m.steer.log" | grep -E " rows |⚠|Error" | sed 's/^/    /'
  done
done
echo "=== $(date +%H:%M) random controls done"
