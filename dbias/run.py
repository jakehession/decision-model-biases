"""Score tasks × models × seeds → results/<model>/<task>.parquet (one row per item × seed)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import typer
import yaml

from .backends import build_backend
from .items import ROTATE
from .tasks import TASKS

app = typer.Typer(add_completion=False)


CONDITION_KEYS = {"s": "adapter_scale", "a": "steer_alpha"}


def condition_spec(specs: dict, name: str) -> dict:
    """`bosun-v3.1@s=0.5` = config entry `bosun-v3.1` with adapter_scale 0.5; `@a=2` sets steer_alpha."""
    base, *mods = name.split("@")
    spec = dict(specs[base])
    for m in mods:
        k, v = m.split("=")
        spec[CONDITION_KEYS[k]] = float(v)
    return spec


@app.command()
def main(
    models: str = typer.Option(..., help="comma-separated names from configs/models.yaml"),
    tasks: str = typer.Option(",".join(TASKS), help="comma-separated task names"),
    limit: int = typer.Option(None, help="max items per task (sampled)"),
    task_limit: list[str] = typer.Option([], help="per-task override, e.g. discrim_eval=2700"),
    seeds: str = typer.Option("0,1,2", help="presentation-order seeds; >1 averages out position. "
                              "1000,1001,... = counterbalanced rotations (bases only)"),
    batch: int = 8,
    out: Path = Path("results"),
    config: Path = Path("configs/models.yaml"),
):
    specs = yaml.safe_load(config.read_text())
    seed_list = [int(s) for s in seeds.split(",")]
    overrides = {k: int(v) for k, v in (x.split("=") for x in task_limit)}
    task_items = {t: TASKS[t](overrides.get(t, limit)) for t in tasks.split(",")}
    for name in models.split(","):
        backend = build_backend(name, condition_spec(specs, name))
        for task, items in task_items.items():
            t0, rows = time.time(), []
            for seed in seed_list:
                for i in range(0, len(items), batch):
                    chunk = [it for it in items[i : i + batch]  # rotation k only exists for items with > k options
                             if seed < ROTATE or seed - ROTATE < len(it.candidates)]
                    if not chunk:
                        continue
                    with backend.intervention():
                        scored = backend.score(chunk, seed)
                    for it, r in zip(chunk, scored):
                        ids = [c["id"] for c in it.candidates]
                        top = max(range(len(ids)), key=r["probs"].__getitem__)
                        rows.append({
                            "model": name, "task": task, "item_id": it.item_id, "group_id": it.group_id,
                            "seed": seed, "candidate_ids": ids, "probs": r["probs"],
                            "presented_order": r["presented_order"], "option_mass": r["option_mass"],
                            "choice": ids[top], "gold": it.gold,
                            "variant": json.dumps(it.variant), "meta": json.dumps(it.meta),
                        })
            path = out / name / f"{task}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path)
            mass = pd.Series([r["option_mass"] for r in rows]).dropna()
            if len(mass) and mass.mean() < 0.5:
                typer.secho(f"  ⚠ {name}/{task}: mean option_mass {mass.mean():.2f}; the model isn't "
                            "answering with option letters, so these probabilities are not meaningful",
                            fg="yellow")
            typer.echo(f"{name:28s} {task:14s} {len(rows):6d} rows  {time.time() - t0:6.1f}s → {path}")


if __name__ == "__main__":
    app()
