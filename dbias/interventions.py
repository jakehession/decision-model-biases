"""Causal interventions on a decision model's base: LoRA adapter scaling (Bosun) and residual steering (any reader)."""

from __future__ import annotations

import contextlib
from typing import Any

import torch


def decoder_layers(model: Any) -> torch.nn.ModuleList:
    """The longest `*layers` ModuleList: the decoder blocks (same rule vegaml uses, so indices agree)."""
    best = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and (best is None or len(mod) > len(best)):
            best = mod
    if best is None:
        raise RuntimeError("no decoder layer list found")
    return best


@contextlib.contextmanager
def lora_scale(peft_model: Any, scale: float, adapter: str = "default"):
    """Multiply every LoRA delta by `scale` (0 = base weights, 1 = as trained); restored on exit."""
    from peft.tuners.lora import LoraLayer
    layers = [m for m in peft_model.modules() if isinstance(m, LoraLayer) and adapter in m.scaling]
    assert layers, "no LoRA layers found"
    for m in layers:
        m.set_scale(adapter, scale)
    try:
        yield
    finally:
        for m in layers:
            m.set_scale(adapter, 1.0)


@contextlib.contextmanager
def steer(model: Any, layer: int, vector: torch.Tensor, alpha: float):
    """Add alpha * vector to the residual stream at the output of decoder block `layer`, every position."""
    if alpha == 0:
        yield
        return
    block = decoder_layers(model)[layer]

    def hook(_mod, _inp, out):
        if isinstance(out, (tuple, list)):
            h = out[0]
            return (h + alpha * vector.to(h.device, h.dtype), *out[1:])
        return out + alpha * vector.to(out.device, out.dtype)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
