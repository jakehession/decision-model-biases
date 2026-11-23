"""Score tasks × models × seeds → results/<model>/<task>.parquet (one row per item × seed)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import typer
import yaml

from .backends import build_backend
from .tasks import TASKS

app = typer.Typer(add_completion=False)


@app.command()
def main(
    models: str = typer.Option(..., help="comma-separated names from configs/models.yaml"),
    tasks: str = typer.Option(",".join(TASKS), help="comma-separated task names"),
    limit: int = typer.Option(None, help="max items per task (sampled)"),
    seeds: str = typer.Option("0,1,2", help="presentation-order seeds; >1 averages out position"),
    batch: int = 8,
    out: Path = Path("results"),
    config: Path = Path("configs/models.yaml"),
):
    specs = yaml.safe_load(config.read_text())
    seed_list = [int(s) for s in seeds.split(",")]
    task_items = {t: TASKS[t](limit) for t in tasks.split(",")}
    for name in models.split(","):
        backend = build_backend(name, specs[name])
        for task, items in task_items.items():
            t0, rows = time.time(), []
            for seed in seed_list:
                for i in range(0, len(items), batch):
                    chunk = items[i : i + batch]
                    for it, r in zip(chunk, backend.score(chunk, seed)):
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
