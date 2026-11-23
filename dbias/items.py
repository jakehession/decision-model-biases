"""Model-agnostic decision item: what every task emits and every backend scores."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionItem:
    task: str
    item_id: str  # unique per row; also seeds candidate presentation order
    group_id: str  # rows sharing a group_id are counterfactual variants of one another
    instructions: str
    candidates: list[dict[str, str]]  # each {"id", "label", "description"?}
    state: Any = None
    decision_type: str = "choice"  # choice | score | noul (Bosun's typed decision kinds)
    gold: str | None = None
    variant: dict[str, Any] = field(default_factory=dict)  # the manipulated factors
    meta: dict[str, Any] = field(default_factory=dict)


def presentation_order(seed: int, row_id: str, n: int) -> list[int]:
    """Bosun's stable-slot shuffle, reused so every backend sees the same option order."""
    order = list(range(n))
    digest = hashlib.sha256(f"{seed}:{row_id}:candidate-order".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order
