# decision-model-biases

Do decision models (LLMs fine-tuned to return a probability over a fixed candidate set) inherit,
amplify or suppress the biases of their base LLM?

## Pipeline

```
tasks.py ──► DecisionItem ──► backend.score() ──► results/<model>/<task>.parquet ──► analyze.py
             (state, instructions,   (probs over candidates,      (one row per item × seed)     (per-task bias metrics +
              candidates, variant)    presented_order)                                           decision-vs-base correlation)
```

- **DecisionItem** (`dbias/items.py`) is model-agnostic: state, instructions, candidates, `decision_type`
  (`choice | score | noul`), `group_id` (rows that are counterfactual variants of one another), and
  `variant` (the manipulated factors).
- **Backends** (`dbias/backends.py`) each return a probability vector aligned to the candidates:
  - `bosun`: native decision-slot readout (`trust_remote_code`).
  - `kev`: kev's pointer head, scored in-process with the vendored `vendor/kev` code exactly as `kev.serve` does
    (`to_record` → `admit` → `probs`, bf16 off-CPU). Kev keeps the option order it is given; yes/no is always [no, yes].
  - `mcq`: any chat causal LM (`chat: false` gives a plain "Answer:" prompt for pre-trained base models). Options are lettered and we read the next-token probability of each letter.
    `option_mass` records how much probability landed on letters; low mass means the readout is invalid.
- **Same order for every model.** All backends use Bosun's seeded shuffle (`presentation_order`), so
  base and decision model see identical option orders. `--seeds 0,1,2` averages position effects out of
  content biases, and position bias is measured from the recorded `presented_order`.
- **Plug in a model:** add an entry to `configs/models.yaml` and set `base:` to its underlying LLM's entry.
  A new decision-model family that has its own readout needs a new `Backend` subclass.

```bash
uv run python -m dbias.run --models bosun-v3.1,qwen3-1.7b --tasks position,framing,discrim_eval,bbq --limit 600
uv run python -m dbias.analyze bosun-v3.1,qwen3-1.7b
```

### Validity checks

- Bosun 1.7B / 0.6B: batched native scoring matches `model.predict()` (max |Δp| 0.018 / 0.010, bf16 padding noise),
  and adapter-off reproduces the pinned Qwen base exactly (Δp = 0.000).
- Bosun with the adapter on, read out through letters, puts ~0 mass on letters, so that condition was dropped.

## Model pairs

| decision model | base LLM (pinned revision) | backend |
| :-- | :-- | :-- |
| Hanno-Labs/bosun-v3.1-1.7b | Qwen/Qwen3-1.7B `70d244cc` | bosun / mcq (chat) |
| Hanno-Labs/bosun-v3.1-0.6b | Qwen/Qwen3-0.6B `c1899de2` | bosun / mcq (chat) |
| jaredpalmer/kev-0.6b `dece6dba` | Qwen/Qwen3-0.6B-Base `da87bfb6` | kev / mcq (plain) |

Candidates come from the DecisionBench results registry's `model_type: decision-model` submissions (open weights,
named base, ≤ ~5B). Next: kev-0.5b (Qwen2.5-0.5B), kev-4b (Qwen3.5-4B-Base), Open-Jev-2B, decider-2b, imajev-4b.

## Bias catalog

Status ✅ = implemented in `tasks.py`.

| family | bias | source | decision framing | metric |
| :-- | :-- | :-- | :-- | :-- |
| Structural | ✅ position / order | synthetic interchangeable options | choice, 2–5 options | P(slot) vs uniform |
| Structural | label / ID (A/B vs 1/2 vs names) | synthetic | choice | Δp when only labels change |
| Structural | verbosity / length | synthetic (pad one description) | choice | P(longer option) − 0.5 |
| Structural | acquiescence (yes-bias) | discrim-eval, BoolQ with polarity flip | noul | P(yes) on negated pairs |
| Structural | abstention / null preference | BBQ "unknown" option | choice | P(unknown) on disambiguated items |
| Structural | ordinal: central tendency / leniency | rubric items with known scores | score | mean predicted − gold |
| Social | ✅ stereotype in QA | BBQ (`Elfsong/BBQ`) | choice, 3 options | BBQ bias score, ambiguous and disambiguated |
| Social | ✅ demographic decision disparity | Anthropic discrim-eval (explicit) | noul | Δlogit P(yes) vs reference (white, male, 60) |
| Social | name-based disparity | discrim-eval *implicit*; resume/loan name swaps | noul / choice | same as above |
| Social | stereotype likelihood | CrowS-Pairs, StereoSet | 2-choice | P(stereotype sentence) |
| Cognitive | ✅ framing (gain/loss) | Tversky–Kahneman templates | choice | P(sure \| gain) − P(sure \| loss) |
| Cognitive | anchoring | templated estimates with high/low anchor | score | Δ expected score |
| Cognitive | decoy / attraction effect | templated A, B, decoy-of-A | choice | Δ P(A) with decoy |
| Cognitive | sunk cost, status quo, certainty | templated pairs | choice | Δp across variants |
| Cognitive | base-rate neglect, conjunction fallacy | Linda-style templates | choice | P(fallacious) |
| Cognitive | authority / sycophancy | inject "the manager prefers X" into state | choice | Δ P(X) |

Larger cognitive-bias suites worth evaluating for import: Malberg et al. 2024 (30 biases) and
BiasBuster (Echterhoff et al. 2024). Not yet checked for licence or format.

## Analysis

For each task, `analyze.py` prints the standard metric per model, then for each model with a `base:` it
reports the Spearman correlation and slope between the decision model's and base LLM's per-item bias
signal, plus the mean |signal| ratio (amplification). Low `option_mass` on an MCQ model means its numbers
are not meaningful.
