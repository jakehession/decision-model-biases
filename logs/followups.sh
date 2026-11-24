#!/usr/bin/env bash
# Follow-up queue: one model per invocation, strictly sequential (M2 Pro can't hold two models).
# Conditions of ONE model (name@s=…, name@a=…) share a single load, so they go in one invocation.
set -u
cd "$(dirname "$0")/.."
run() { local m=$1 t=$2 tag=${3:-$1}; echo "=== $(date +%H:%M) $tag: $t"; uv run python -m dbias.run --models "$m" --tasks "$t" \
  --limit 600 --task-limit discrim_eval=2700 > "logs/$tag.followup.log" 2>&1 || echo "!!! $tag failed (see logs/$tag.followup.log)"; \
  grep -E "rows|⚠" "logs/$tag.followup.log" | sed 's/^/    /'; }
sweep() { local fam=$1; shift; local out=(); for v in "$@"; do out+=("${fam/\{\}/$v}"); done; (IFS=,; echo "${out[*]}"); }
ALL=position,framing,discrim_eval,bbq
S="0 0.25 0.5 0.75 1"; A="-2 -1 0 1 2"

# 1. Vega steering (0.8B): reader read through letters and cloze, then Vega, same gender direction
# PAUSED (alpha=±2 does not move the reader gender gap; see scratch/check_steer.py) run "$(sweep 'qwen3.5-0.8b@a={}' $A),$(sweep 'qwen3.5-0.8b-cloze@a={}' $A)" discrim_eval steer-qwen3.5-0.8b
# PAUSED run "$(sweep 'vega-u-0.8b@a={}' $A)" discrim_eval steer-vega-u-0.8b
# 2. Bosun-0.6B adapter sweep: native readout and letter readout on the same weights
# DONE 2026-11-24 run "$(sweep 'bosun-v3.1-0.6b@s={}' $S),$(sweep 'bosun-v3.1-0.6b-letters@s={}' $S)" $ALL sweep-bosun-0.6b
# 3. letter-free control for every base
for m in qwen3-1.7b-cloze qwen3.5-0.8b-cloze qwen3.5-4b-cloze; do run $m $ALL; done
# 4. framing (Plan X/Y labels) + discrim-eval at 20 templates for everything already scored
for m in qwen3-0.6b qwen3-1.7b qwen3.5-0.8b qwen3.5-4b bosun-v3.1-0.6b bosun-v3.1 vega-u-0.8b; do run $m framing,discrim_eval; done
run vega-u-4b framing,discrim_eval,bbq
# 5. Bosun-1.7B adapter sweep: DROPPED 2026-11-24 (not wanted)
# run "$(sweep 'bosun-v3.1@s={}' $S),$(sweep 'bosun-v3.1-letters@s={}' $S)" $ALL sweep-bosun-1.7b
echo "=== $(date +%H:%M) queue done"
