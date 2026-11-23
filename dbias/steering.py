"""Build a demographic steering direction from held-out discrim-eval templates (never the scored ones).

vector = mean over counterfactual pairs of [mean-pooled hidden(target) - mean-pooled hidden(reference)] at the
output of decoder block `layer`. Pairs differ only in the attribute; templates scored by `run` are excluded.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import typer
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backends import _device
from .interventions import decoder_layers
from .tasks import HF_CACHE, TASKS

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(..., help="mcq entry in configs/models.yaml whose weights are the reader"),
    layer: int = typer.Option(..., help="decoder block whose output is steered"),
    attr: str = "gender", target: str = "female", reference: str = "male",
    scored_limit: int = typer.Option(2700, help="discrim_eval limit used in runs; those templates are held out"),
    pairs_per_template: int = 10,
    out: Path = Path("results/steer"),
    config: Path = Path("configs/models.yaml"),
):
    spec = yaml.safe_load(config.read_text())[model]
    kw = {"revision": spec.get("revision"), "local_files_only": spec.get("local_files_only", False)}
    tok = AutoTokenizer.from_pretrained(spec["path"], **kw)
    lm = AutoModelForCausalLM.from_pretrained(spec["path"], dtype=getattr(torch, spec.get("dtype", "bfloat16")),
                                              **kw).to(_device()).eval()
    block = decoder_layers(lm)[layer]

    scored = {it.group_id.split(":")[1] for it in TASKS["discrim_eval"](scored_limit)}
    rows = [r for r in load_dataset("Anthropic/discrim-eval", "explicit", cache_dir=HF_CACHE)["train"]
            if str(r["decision_question_id"]) not in scored]
    keys = ("decision_question_id", "age", "race", "gender")
    cells = {}
    for r in rows:
        if r[attr] in (target, reference):
            k = tuple(r[x] for x in keys if x != attr)
            cells.setdefault(k, {})[r[attr]] = r["filled_template"]
    by_q: dict = {}
    for k, v in cells.items():
        if len(v) == 2:
            by_q.setdefault(k[0], []).append(v)
    rng = random.Random(0)
    pairs = [p for q in sorted(by_q) for p in rng.sample(by_q[q], min(pairs_per_template, len(by_q[q])))]

    cap = {}
    handle = block.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    def pooled(text: str) -> torch.Tensor:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, **spec.get("chat_kwargs", {}))
        with torch.inference_mode():
            lm(**{k: v.to(lm.device) for k, v in ids.items()}, use_cache=False, logits_to_keep=1)
        return cap["h"][0].float().mean(0).cpu()

    diffs = torch.stack([pooled(p[target]) - pooled(p[reference]) for p in pairs])
    handle.remove()
    vec = diffs.mean(0)
    cos = torch.nn.functional.cosine_similarity(diffs, vec[None], dim=-1)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model}-{attr}-{target}-L{layer}.pt"
    torch.save({"vector": vec, "layer": layer, "attr": attr, "target": target, "reference": reference,
                "n_pairs": len(pairs), "n_templates": len(by_q), "held_out_from": sorted(scored)}, path)
    typer.echo(f"{path}: {len(pairs)} pairs from {len(by_q)} held-out templates, |v|={vec.norm():.3f}, "
               f"mean cos(pair diff, v)={cos.mean():.3f}")


if __name__ == "__main__":
    app()
