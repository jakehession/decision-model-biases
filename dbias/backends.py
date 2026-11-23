"""Backends turn DecisionItems into candidate probabilities. Add a class here to plug in a new model family."""

from __future__ import annotations

import contextlib
import string
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interventions import lora_scale, steer
from .items import DecisionItem, presentation_order

LETTERS = string.ascii_uppercase
MCQ_SYSTEM = "You are making a decision. Reply with only the letter of the option you choose."


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return str(state)


class Backend:
    """score() returns, per item: probs aligned to item.candidates, presented_order, diagnostics."""

    name: str

    def score(self, items: list[DecisionItem], seed: int) -> list[dict[str, Any]]:
        raise NotImplementedError


def load_bosun(path: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(path).expanduser()),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    return model.to(_device()).eval()


class BosunBackend(Backend):
    """Native typed-decision readout over Bosun's stable decision-slot tokens."""

    def __init__(self, name: str, model: Any):
        self.name, self.model, self.tok = name, model, model.tokenizer
        self.mod = sys.modules[type(model).__module__]  # remote-code module: prompt compiler

    def score(self, items, seed):
        prompts, orders = [], []
        for it in items:
            content, order, _ = self.mod.render_decision_prompt(
                state=it.state, instructions=it.instructions, candidates=it.candidates,
                decision_type=it.decision_type, decision_tokens=self.model.decision_tokens,
                seed=seed, row_id=it.item_id, prompt_schema=self.model.config.prompt_schema,
            )
            assert order == presentation_order(seed, it.item_id, len(it.candidates))
            prompts.append(self.tok.apply_chat_template(
                [{"role": "system", "content": self.mod._SYSTEM_PROMPT},
                 {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=self.model.config.decision_prompt_enable_thinking,
            ))
            orders.append(order)
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            logits = self.model.decision_logits(**enc).float().cpu()
        out = []
        for row, it, order in zip(logits, items, orders):
            slot_p = torch.softmax(row[: len(it.candidates)], -1)
            probs = [0.0] * len(order)
            for slot, ci in enumerate(order):
                probs[ci] = float(slot_p[slot])
            out.append({"probs": probs, "presented_order": order, "option_mass": None})
        return out


class MCQBackend(Backend):
    """Any chat causal LM: lettered options, read next-token probability of each letter.

    option_mass = total probability on the option letters (low → model wanted to say something else).
    """

    def __init__(self, name: str, model: Any, tok: Any, chat_kwargs: dict | None = None,
                 adapter_ctx: Any = None, chat: bool = True):
        self.name, self.model, self.tok = name, model, tok
        self.chat_kwargs = chat_kwargs or {}
        self.chat = chat  # False: plain "...Answer:" completion, for pre-trained base models
        self.adapter_ctx = adapter_ctx or contextlib.nullcontext
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(L if chat else f" {L}", add_special_tokens=False)
            assert len(ids) == 1, f"letter {L!r} is not a single token"
            self.letter_ids.append(ids[0])

    def _prompt(self, it: DecisionItem, order: list[int]) -> str:
        opts = []
        for slot, ci in enumerate(order):
            c = it.candidates[ci]
            desc = f" — {c['description']}" if c.get("description") else ""
            opts.append(f"{LETTERS[slot]}. {c['label']}{desc}")
        state = _render_state(it.state)
        body = (f"{state}\n\n" if state else "") + f"Question: {it.instructions}\n\nOptions:\n" + \
            "\n".join(opts) + "\n\nAnswer with a single letter."
        if not self.chat:
            return f"{MCQ_SYSTEM}\n\n{body}\nAnswer:"
        return self.tok.apply_chat_template(
            [{"role": "system", "content": MCQ_SYSTEM}, {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, **self.chat_kwargs,
        )

    def score(self, items, seed):
        orders = [presentation_order(seed, it.item_id, len(it.candidates)) for it in items]
        assert all(len(o) <= len(LETTERS) for o in orders), "MCQ backend supports ≤26 options"
        enc = self.tok([self._prompt(it, o) for it, o in zip(items, orders)], return_tensors="pt",
                       padding=True, add_special_tokens=self.chat).to(self.model.device)
        with torch.inference_mode(), self.adapter_ctx():
            logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), -1).cpu()
        out = []
        for row, order in zip(logp, orders):
            lp = row[self.letter_ids[: len(order)]]
            mass = float(lp.exp().sum())
            slot_p = torch.softmax(lp, -1)
            probs = [0.0] * len(order)
            for slot, ci in enumerate(order):
                probs[ci] = float(slot_p[slot])
            out.append({"probs": probs, "presented_order": order, "option_mass": mass})
        return out


CLOZE_SYSTEM = "You are making a decision. Reply with only the option you choose, exactly as written."


def _option_text(c: dict) -> str:
    return f"{c['label']} — {c['description']}" if c.get("description") else c["label"]


class ClozeBackend(Backend):
    """Letter-free control for MCQBackend: options listed as bullets, scored by log P(option text | prompt).

    Removes letter-token bias from the base readout. option_mass = total P over the exact option strings.
    """

    def __init__(self, name: str, model: Any, tok: Any, chat_kwargs: dict | None = None, chat: bool = True):
        self.name, self.model, self.tok, self.chat = name, model, tok, chat
        self.chat_kwargs = chat_kwargs or {}
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _prompt(self, it: DecisionItem, order: list[int]) -> str:
        opts = "\n".join(f"- {_option_text(it.candidates[ci])}" for ci in order)
        state = _render_state(it.state)
        body = (f"{state}\n\n" if state else "") + f"Question: {it.instructions}\n\nOptions:\n{opts}"
        if not self.chat:
            return f"{CLOZE_SYSTEM}\n\n{body}\nAnswer:"
        return self.tok.apply_chat_template(
            [{"role": "system", "content": CLOZE_SYSTEM}, {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, **self.chat_kwargs,
        )

    def score(self, items, seed):
        orders = [presentation_order(seed, it.item_id, len(it.candidates)) for it in items]
        seqs, conts = [], []  # one sequence per (item, candidate), prompt and option tokenised separately
        for it, order in zip(items, orders):
            p_ids = self.tok.encode(self._prompt(it, order), add_special_tokens=not self.chat)
            for c in it.candidates:
                c_ids = self.tok.encode(("" if self.chat else " ") + _option_text(c), add_special_tokens=False)
                seqs.append(p_ids + c_ids)
                conts.append(c_ids)
        width, k = max(map(len, seqs)), max(map(len, conts)) + 1
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
        with torch.inference_mode():
            logits = self.model(input_ids=ids.to(self.model.device), attention_mask=mask.to(self.model.device),
                                use_cache=False, logits_to_keep=k).logits
        logp = torch.log_softmax(logits.float(), -1).cpu()
        seq_lp = []
        for row, c_ids in zip(logp, conts):  # row[j] predicts token at position width-k+1+j
            n = len(c_ids)
            seq_lp.append(float(row[k - 1 - n : k - 1].gather(-1, torch.tensor(c_ids)[:, None]).sum()))
        out, i = [], 0
        for it, order in zip(items, orders):
            lp = torch.tensor(seq_lp[i : i + len(it.candidates)])
            i += len(it.candidates)
            out.append({"probs": torch.softmax(lp, -1).tolist(), "presented_order": order,
                        "option_mass": float(lp.exp().sum())})
        return out


KEV_SRC = Path(__file__).resolve().parent.parent / "vendor" / "kev"


class KevBackend(Backend):
    """Kev (jaredpalmer/kev): pointer head over option spans, loaded and scored exactly as kev.serve does.

    Kev keeps options in the order given, so our shared shuffle controls choice order. Noul is always [no, yes].
    """

    def __init__(self, name: str, path: str):
        if str(KEV_SRC) not in sys.path:
            sys.path.insert(0, str(KEV_SRC))
        from kev.api import SystemOneRequest, to_record
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.model import admit
        self.name, self._req, self._to_record, self._admit = name, SystemOneRequest, to_record, admit
        dev = _device()  # kev.serve's default off-CPU is bf16 (LoadOptions.dtype)
        dtype = torch.bfloat16 if dev != "cpu" else None
        self.tok, self.model = Checkpoint(str(Path(path).expanduser())).load(dev, LoadOptions(dtype=dtype))

    def score(self, items, seed):
        out = []
        for it in items:
            n = len(it.candidates)
            if it.decision_type == "noul":
                ids = [c["id"] for c in it.candidates]
                order = [ids.index("no"), ids.index("yes")]  # kev's fixed noul layout
                q = {"type": "noul", "instructions": it.instructions}
            else:
                order = presentation_order(seed, it.item_id, n)
                crit = {it.candidates[ci]["label"]: it.candidates[ci].get("description") or None for ci in order}
                assert len(crit) == n, f"{it.item_id}: kev needs unique option labels"
                q = {"type": it.decision_type, "instructions": it.instructions, "criteria": crit}
            rec, _ = self._to_record(self._req(state=it.state, questions={"q": q}))
            slot_p = self.model.probs(self._admit(self.model, self.tok, rec))[0].float()
            probs = [0.0] * n
            for slot, ci in enumerate(order):
                probs[ci] = float(slot_p[slot])
            out.append({"probs": probs, "presented_order": order, "option_mass": None})
        return out


class VegaBackend(Backend):
    """vegaml (laya 0.5 engine): a 14.3M decision engine over a frozen LLM reader.

    Core engine readout, views=1: options are read in the order given (our shared shuffle), boolean is [no, yes].
    The reader is loaded from a local snapshot so the base comparison uses the identical weights.
    """

    def __init__(self, name: str, engine_dir: str, reader: str):
        from vegaml._api import Vega
        self.name = name
        self.v = Vega(str(Path(engine_dir).expanduser()), backbone=str(Path(reader).expanduser()))

    def score(self, items, seed):
        out = []
        for it in items:
            n, ids = len(it.candidates), [c["id"] for c in it.candidates]
            if it.decision_type == "noul":
                order, keys = [ids.index("no"), ids.index("yes")], ["False", "True"]
                q = {"type": "boolean", "instructions": it.instructions}
            else:
                order = presentation_order(seed, it.item_id, n)
                keys = [it.candidates[ci]["label"] for ci in order]
                assert len(set(keys)) == n, f"{it.item_id}: vega needs unique option labels"
                q = {"type": it.decision_type, "instructions": it.instructions,
                     "criteria": {it.candidates[ci]["label"]: it.candidates[ci].get("description") or None
                                  for ci in order}}
            ans = self.v.decide(it.state, {"q": q}, views=1)["answers"]["q"]["probs"]
            probs = [0.0] * n
            for slot, ci in enumerate(order):
                probs[ci] = float(ans[keys[slot]])
            out.append({"probs": probs, "presented_order": order, "option_mass": None})
        return out


_BOSUN_CACHE: dict[str, Any] = {}
_LM_CACHE: dict[tuple, Any] = {}  # one copy of a reader per process, shared by every condition that uses it
_VEGA_CACHE: dict[tuple, Any] = {}


def _load_lm(spec: dict) -> tuple[Any, Any]:
    key = (spec["path"], spec.get("revision"), spec.get("dtype", "bfloat16"))
    if key not in _LM_CACHE:
        kw = {"revision": spec.get("revision"), "local_files_only": spec.get("local_files_only", False)}
        tok = AutoTokenizer.from_pretrained(spec["path"], **kw)
        model = AutoModelForCausalLM.from_pretrained(spec["path"], dtype=getattr(torch, key[2]), **kw)
        _LM_CACHE[key] = (model.to(_device()).eval(), tok)
    return _LM_CACHE[key]


def _build(name: str, spec: dict) -> tuple[Backend, Any, Any]:
    """-> (backend, LM the steering hook goes on, PEFT model whose adapter scales)."""
    kind = spec["backend"]
    if spec.get("loader") == "bosun" or kind == "bosun":
        path = spec["path"]
        if path not in _BOSUN_CACHE:
            _BOSUN_CACHE[path] = load_bosun(path)
        bosun = _BOSUN_CACHE[path]
        if kind == "bosun":
            return BosunBackend(name, bosun), bosun.model, bosun.model
        ctx = bosun.model.disable_adapter if spec.get("adapter") == "off" else None
        return MCQBackend(name, bosun.model, bosun.tokenizer, spec.get("chat_kwargs"), adapter_ctx=ctx), \
            bosun.model, bosun.model
    if kind == "vega":
        key = (spec["engine"], spec["reader"])
        if key not in _VEGA_CACHE:
            _VEGA_CACHE[key] = VegaBackend(name, spec["engine"], spec["reader"])
        b = VegaBackend.__new__(VegaBackend)  # same loaded engine + reader, this condition's name
        b.name, b.v = name, _VEGA_CACHE[key].v
        return b, b.v.perc.model, None
    if kind == "kev":
        return KevBackend(name, spec["path"]), None, None
    if kind in ("mcq", "cloze"):
        model, tok = _load_lm(spec)
        cls = MCQBackend if kind == "mcq" else ClozeBackend
        return cls(name, model, tok, spec.get("chat_kwargs"), chat=spec.get("chat", True)), model, None
    raise ValueError(f"unknown backend {kind!r} for {name}")


def build_backend(name: str, spec: dict) -> Backend:
    """Build from a configs/models.yaml entry. `adapter_scale` / `steer_alpha` (set by a `name@s=` / `name@a=`
    condition in run.py) attach an intervention that run.py enters around every score() call."""
    backend, lm, peft_model = _build(name, spec)
    ctxs = []
    if spec.get("adapter_scale") is not None:
        assert peft_model is not None, f"{name}: adapter_scale needs a LoRA (bosun) model"
        ctxs.append(lambda: lora_scale(peft_model, float(spec["adapter_scale"])))
    if spec.get("steer_alpha"):
        assert lm is not None and "steer" in spec, f"{name}: steer_alpha needs a `steer:` entry in the config"
        st = torch.load(Path(spec["steer"]).expanduser(), weights_only=True)
        ctxs.append(lambda: steer(lm, st["layer"], st["vector"], float(spec["steer_alpha"])))

    @contextlib.contextmanager
    def intervention():
        with contextlib.ExitStack() as stack:
            for c in ctxs:
                stack.enter_context(c())
            yield

    backend.intervention = intervention
    return backend
