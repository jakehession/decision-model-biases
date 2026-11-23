"""Bias along an intervention sweep: one row per condition (adapter scale or steering alpha) per task.

ρ_ref = per-item bias signal vs a reference condition (the base); ρ_end = vs the sweep's last condition.
Headline: position = mean |P(first) - uniform|, framing = framing effect, discrim_eval = female and
non-binary logit shift vs male, bbq = mean ambiguous-item bias score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr

from .analyze import METRICS

app = typer.Typer(add_completion=False)


def headline(task: str, summary: pd.DataFrame, signal: pd.Series) -> dict:
    if task == "position":
        return {"mean_abs_first_dev": signal.abs().mean()}
    if task == "framing":
        return {"framing_effect": float(summary.framing_effect.iloc[0])}
    if task == "discrim_eval":
        return {"female_vs_male": summary.loc[("gender", "female"), "mean"],
                "nonbinary_vs_male": summary.loc[("gender", "non-binary"), "mean"],
                "p_yes": summary.loc[("overall", "P(yes)"), "mean"]}
    if task == "bbq":
        return {"ambig_bias": summary.loc["ambig"].bias_score.mean(), "ambig_p_unknown": summary.loc["ambig"].p_unknown.mean()}
    return {}


def _rho(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join="inner")
    return spearmanr(a, b).statistic if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan


@app.command()
def main(
    family: str = typer.Option(..., help="condition name with {} for the value, e.g. 'bosun-v3.1@s={}'"),
    values: str = typer.Option(..., help="comma-separated values, in sweep order"),
    ref: str = typer.Option(None, help="reference condition, e.g. the base model"),
    tasks: str = ",".join(METRICS),
    results: Path = Path("results"),
):
    names = [family.format(v) for v in values.split(",")] + ([ref] if ref else [])
    pd.set_option("display.width", 200)
    for task in tasks.split(","):
        sig, rows = {}, []
        for n in names:
            p = results / n / f"{task}.parquet"
            if p.exists():
                summary, sig[n] = METRICS[task](pd.read_parquet(p))
                rows.append({"condition": n, **headline(task, summary, sig[n]), "mean_abs_signal": sig[n].abs().mean()})
        if not rows:
            continue
        df = pd.DataFrame(rows).set_index("condition")
        end = [n for n in names if n in sig and n != ref][-1]
        df["rho_ref"] = [_rho(sig[n], sig[ref]) if ref in sig else np.nan for n in df.index]
        df["rho_end"] = [_rho(sig[n], sig[end]) for n in df.index]
        print(f"\n{'═' * 16} {task} {'═' * 16}\n{df.round(3)}")


if __name__ == "__main__":
    app()
