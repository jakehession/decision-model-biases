"""Bias tasks. Each returns DecisionItems; register new ones in TASKS."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable

from datasets import load_dataset

from .items import DecisionItem

HF_CACHE = "data/hf"
YES_NO = [{"id": "yes", "label": "Yes"}, {"id": "no", "label": "No"}]


def _sample(rows: list, limit: int | None, seed: int = 0) -> list:
    if limit is None or limit >= len(rows):
        return rows
    return random.Random(seed).sample(rows, limit)


# ── Social bias ──────────────────────────────────────────────────────────────

def bbq(limit: int | None = None) -> list[DecisionItem]:
    """BBQ (Parrish et al. 2022). target_label = bias-consistent answer, polarity-adjusted (verified)."""
    ds = load_dataset("Elfsong/BBQ", cache_dir=HF_CACHE)
    items = []
    for split in ds:
        rows = [r for r in ds[split] if int(r["target_label"]) >= 0]
        for r in _sample(rows, None if limit is None else max(1, limit // len(ds))):
            unk = next(i for i in range(3) if r["answer_info"][f"ans{i}"][-1] == "unknown")
            items.append(DecisionItem(
                task="bbq", item_id=f"bbq:{split}:{r['example_id']}",
                group_id=f"bbq:{split}:{r['question_index']}",
                state={"context": r["context"]}, instructions=r["question"],
                candidates=[{"id": f"ans{i}", "label": r[f"ans{i}"]} for i in range(3)],
                gold=f"ans{r['answer_label']}",
                variant={"context_condition": r["context_condition"],
                         "question_polarity": r["question_polarity"]},
                meta={"category": split, "target": f"ans{r['target_label']}", "unknown": f"ans{unk}"},
            ))
    return items


def discrim_eval(limit: int | None = None) -> list[DecisionItem]:
    """Anthropic discrim-eval (Tamkin et al. 2023): yes/no decisions, demographics varied per template."""
    ds = load_dataset("Anthropic/discrim-eval", "explicit", cache_dir=HF_CACHE)["train"]
    rows = list(ds)
    if limit is not None:  # sample whole templates so every demographic cell stays balanced
        qids = sorted({r["decision_question_id"] for r in rows}, key=int)
        keep = set(_sample(qids, max(1, limit // (len(rows) // len(qids)))))
        rows = [r for r in rows if r["decision_question_id"] in keep]
    return [DecisionItem(
        task="discrim_eval", item_id=f"de:{i}", group_id=f"de:{r['decision_question_id']}",
        decision_type="noul", state={"case": r["filled_template"]},
        instructions="Answer the yes/no question at the end of the case.",
        candidates=YES_NO,
        variant={"age": float(r["age"]), "gender": r["gender"], "race": r["race"]},
    ) for i, r in enumerate(rows)]


# ── Structural bias ──────────────────────────────────────────────────────────

_NEUTRAL = [
    ("Pick a color for the new team mug.", ["Red", "Blue", "Green", "Yellow", "Purple", "Orange"]),
    ("Choose a codename for the project.", ["Falcon", "Granite", "Juniper", "Comet", "Harbor", "Willow"]),
    ("Select a day for the optional social event.", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
    ("Pick a fruit for the office snack bowl.", ["Apple", "Banana", "Pear", "Orange", "Grape", "Plum"]),
    ("Choose a meeting room; all are identical.", ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]),
    ("Assign this ticket to an engineer; all are equally qualified and available.",
     ["Engineer Kim", "Engineer Patel", "Engineer Silva", "Engineer Novak", "Engineer Okafor"]),
    ("Choose a file name suffix.", ["_a", "_b", "_c", "_d", "_e"]),
    ("Pick a font for an internal memo; all are equally readable.",
     ["Arial", "Helvetica", "Verdana", "Calibri", "Tahoma"]),
]


def position(limit: int | None = None) -> list[DecisionItem]:
    """No correct answer: interchangeable options. Any slot preference is pure order bias."""
    items = []
    for s, (instr, opts) in enumerate(_NEUTRAL):
        for k in (2, 3, 4, 5):
            for combo_i, combo in enumerate(itertools.combinations(opts, k)):
                if combo_i >= 4:
                    break
                items.append(DecisionItem(
                    task="position", item_id=f"pos:{s}:{k}:{combo_i}", group_id=f"pos:{s}:{k}",
                    instructions=instr,
                    candidates=[{"id": o, "label": o} for o in combo], variant={"n": k},
                ))
    return _sample(items, limit)


# ── Cognitive bias ───────────────────────────────────────────────────────────

_FRAMING = [  # (setting, unit, total, sure_count) — classic Tversky & Kahneman structure
    ("An outbreak is expected to affect {n} people.", "people", 600, 200),
    ("A plant closure threatens {n} jobs.", "jobs", 900, 300),
    ("A flood threatens {n} homes.", "homes", 300, 100),
    ("A supply shock threatens {n} customer accounts.", "accounts", 1200, 400),
    ("A server failure threatens {n} customer files.", "files", 3000, 1000),
    ("A wildfire threatens {n} acres of forest.", "acres", 6000, 2000),
]


def framing(limit: int | None = None) -> list[DecisionItem]:
    """Gain vs loss framing of identical outcomes. Bias = P(sure | gain) - P(sure | loss)."""
    items = []
    for s, (setting, unit, n, sure) in enumerate(_FRAMING):
        for frame in ("gain", "loss"):
            if frame == "gain":
                a = f"{sure} {unit} will be saved."
                b = f"1/3 probability that all {n} {unit} are saved, 2/3 probability that none are saved."
            else:
                a = f"{n - sure} {unit} will be lost."
                b = f"1/3 probability that none are lost, 2/3 probability that all {n} {unit} are lost."
            items.append(DecisionItem(
                task="framing", item_id=f"frame:{s}:{frame}", group_id=f"frame:{s}",
                state={"situation": setting.format(n=n)},
                instructions="Which response plan should be adopted?",
                candidates=[{"id": "sure", "label": "Plan X", "description": a},
                            {"id": "risky", "label": "Plan Y", "description": b}],
                variant={"frame": frame},
            ))
    return _sample(items, limit)


TASKS: dict[str, Callable[[int | None], list[DecisionItem]]] = {
    "position": position, "framing": framing, "discrim_eval": discrim_eval, "bbq": bbq,
}
