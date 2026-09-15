#!/usr/bin/env python3
"""
kv_eviction_extended.py
=======================

Extends the budget-constrained KV-cache eviction study (Qwen3-8B, NF4, WikiText-103,
multi-anchor paired design) with three additional policies:

  * normkv      - L2-norm key eviction (Devoto et al., EMNLP 2024, "A Simple and
                  Effective L2 Norm-Based Strategy for KV Cache Compression").
                  Per KV head, keep the tokens with the LOWEST ||k||_2.
  * pyramidkv   - PyramidKV (Cai et al., 2024). SnapKV-style observation-window
                  scoring, but with a pyramidal per-layer budget (large in lower
                  layers, small in upper layers) whose mean equals the global budget.
  * tova        - TOVA (Oren et al., 2024, "Transformers are Multi-State RNNs").
                  Head-averaged attention of the current (last) query; the same token
                  set is kept for every KV head in a layer.
                    - `tova`        one-shot at the end of prefill (kvpress-style)
                    - `tova_stream` faithful sequential TOVA: fill to budget, then
                                    evict after every chunk of `--tova-chunk` tokens
                                    (chunk=1 is exact TOVA; slow)

The reference conditions are re-run inside this script so every comparison is
paired on identical anchors:

  * full         - no eviction
  * snapkv       - canonical SnapKV (observation window + pooled scores + window kept)
  * sink_window  - attention sinks + recency window (StreamingLLM-style)
  * recency      - pure sliding window, no sinks
  * random       - unstructured control (uniform random subset per KV head)

Models: written against Qwen3 but architecture-agnostic (QK-norm is applied only when the
attention module has q_norm), so Llama-family models work too. ALWAYS run --selftest on a
new model before a long job. Each model gets its own tokenised-corpus cache, because
tokenisers differ and a shared cache would feed one model's token ids to another.

Evaluation protocol (per anchor):
  1. Prefill `L` context tokens (logits_to_keep=1, so no L x vocab logit tensor).
  2. Compress the context KV cache to `round(budget * L)` tokens per head per layer
     (PyramidKV: mean over layers equals that number).
  3. Score the next `--eval-tokens` tokens with teacher forcing; PPL over them.
  Cached keys keep their ORIGINAL RoPE rotation; continuation tokens get their true
  absolute positions (L, L+1, ...). No position re-indexing is applied.

"Capture once, compress many": for all one-shot policies, compression at layer l is a
function of layer l's full-prefill keys and window queries only (layer l+1's input does
not depend on layer l's eviction, because eviction happens after layer l's attention
has been computed with the full cache). So we run ONE full prefill per anchor, capture
per-layer window queries, and derive every (policy, budget) cache by gathering from the
full cache. This is exactly equivalent to in-forward compression (kvpress /
KVCache-Factory behaviour) and ~20x cheaper. `tova_stream` is sequential by nature and
gets its own prefill.

Statistics: paired differences Delta PPL = PPL(method) - PPL(comparator) across anchors;
mean, SD, 95% CI with exact t_crit(df=n-1) (2.093 for n=20), paired t-test, Cohen's d_z.
Holm correction is applied across context lengths (m = number of lengths) ONLY at the
primary budget (`--primary-budget`, default 0.05); all other budgets are flagged
exploratory and receive no significance marker.

Usage
-----
  # Default run: both lengths, all policies, standard budget sweep
  python kv_eviction_extended.py --out-dir results_extended

  # Reuse the exact anchors from your existing SnapKV runs (strongly recommended so
  # the new rows pair with your existing tables):
  python kv_eviction_extended.py --anchors-file anchors.json

  # Quick correctness check on the real model before a long run (~minutes):
  python kv_eviction_extended.py --selftest

anchors.json format: {"4096": [offset, ...], "8192": [offset, ...]} where each offset is
a token index into the tokenized, "\n\n"-joined WikiText-103 split.

Runs are resumable: rows already present in <out-dir>/raw_L{L}.csv are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

ONE_SHOT_METHODS = ("snapkv", "sink_window", "recency", "pyramidkv", "normkv", "tova", "random")
ALL_METHODS = ("full",) + ONE_SHOT_METHODS + ("tova_stream",)
DEFAULT_METHODS = ("full", "snapkv", "sink_window", "pyramidkv", "normkv", "tova")


@dataclass
class PolicyConfig:
    window: int = 32            # SnapKV / PyramidKV observation window (also kept)
    kernel: int = 7             # pooling kernel for SnapKV / PyramidKV
    pooling: str = "avgpool"    # avgpool | maxpool
    pyramid_beta: float = 20.0  # PyramidKV: top-layer budget = (B - w) / beta
    sinks: int = 4              # sink_window: number of sink tokens
    norm_window: int = 0        # normkv: optional protected recent window (0 = pure KNorm)
    norm_direction: str = "low" # normkv: keep lowest (Devoto et al.) or highest ||k||
    norm_skip_layers: int = 0   # normkv: leave the first N layers uncompressed (0 = off)
    tova_chunk: int = 32        # tova_stream: tokens per eviction step (1 = exact TOVA)


# --------------------------------------------------------------------------------------
# Transformers-version-agnostic cache helpers
# --------------------------------------------------------------------------------------

def _cache_from_kwargs(kwargs):
    c = kwargs.get("past_key_values", None)
    if c is None:
        c = kwargs.get("past_key_value", None)  # transformers < 4.56
    return c


def layer_kv(cache, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):  # transformers >= 4.56
        lyr = cache.layers[i]
        return lyr.keys, lyr.values
    return cache.key_cache[i], cache.value_cache[i]


def set_layer_kv(cache, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[i].keys = k
        cache.layers[i].values = v
    else:
        cache.key_cache[i] = k
        cache.value_cache[i] = v


def layer_len(cache, i: int) -> int:
    if cache is None:
        return 0
    try:
        k, _ = layer_kv(cache, i)
    except (IndexError, AttributeError, KeyError):
        return 0
    if k is None or k.numel() == 0:
        return 0
    return int(k.shape[-2])


def build_cache(kv: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv):
        cache.update(k, v, i)
    return cache


# --------------------------------------------------------------------------------------
# Attention reconstruction (window queries) for Qwen3-style attention modules
# --------------------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def window_queries(module, hidden_states: torch.Tensor, position_embeddings, w: int) -> torch.Tensor:
    """Post-norm, post-RoPE queries of the last `w` positions: (b, H, w, head_dim)."""
    b, q_len, _ = hidden_states.shape
    w = min(w, q_len)
    hd = module.head_dim
    q = module.q_proj(hidden_states[:, -w:].contiguous()).view(b, w, -1, hd)
    if hasattr(module, "q_norm"):  # Qwen3 QK-norm (applied before RoPE)
        q = module.q_norm(q)
    q = q.transpose(1, 2)
    cos, sin = position_embeddings
    cos = cos[:, -w:].unsqueeze(1).to(q.dtype)
    sin = sin[:, -w:].unsqueeze(1).to(q.dtype)
    return q * cos + _rotate_half(q) * sin


def window_attention(q_win: torch.Tensor, keys: torch.Tensor, scaling: float) -> torch.Tensor:
    """Softmax attention of window queries over ALL cached keys, causal inside the window.

    q_win: (b, H, w, d); keys: (b, H_kv, T, d). The w queries are the last w tokens of
    the cache. Returns float32 probabilities (b, H, w, T).
    """
    b, H, w, d = q_win.shape
    T = keys.shape[2]
    groups = H // keys.shape[1]
    k = keys.repeat_interleave(groups, dim=1)           # matches HF repeat_kv ordering
    scores = torch.matmul(q_win.float(), k.float().transpose(-1, -2)) * scaling
    q_pos = torch.arange(T - w, T, device=keys.device).view(w, 1)
    k_pos = torch.arange(T, device=keys.device).view(1, T)
    scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    return torch.softmax(scores, dim=-1)


# --------------------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------------------

def pyramid_budgets(n_layers: int, budget: int, window: int, beta: float) -> List[int]:
    """PyramidKV per-layer budgets (KVCache-Factory schedule), mean exactly == budget.

    prefix budget per layer decreases linearly from max_num (layer 0) to min_num (top):
        min_num = (B - w) / beta,  max_num = 2 (B - w) - min_num
    Rounded with largest-remainder so sum == n_layers * (B - w) exactly, then + w window.
    """
    p = budget - window
    if p <= 0:
        raise ValueError(f"PyramidKV budget {budget} must exceed window {window}")
    min_num = p / beta
    max_num = 2 * p - min_num
    raw = np.array([max_num - i * (max_num - min_num) / max(n_layers - 1, 1) for i in range(n_layers)])
    target = n_layers * p
    floor = np.floor(raw).astype(int)
    rem = int(target - floor.sum())
    order = np.argsort(-(raw - floor), kind="stable")
    floor[order[:rem]] += 1
    return [int(x) + window for x in floor]


# --------------------------------------------------------------------------------------
# Selection policies: return sorted indices (b, H_kv, k) into the T cached positions
# --------------------------------------------------------------------------------------

def _topk_sorted(scores: torch.Tensor, k: int) -> torch.Tensor:
    idx = scores.topk(k, dim=-1).indices
    return idx.sort(dim=-1).values


def _pool(x: torch.Tensor, kernel: int, pooling: str) -> torch.Tensor:
    if kernel <= 1:
        return x
    if pooling == "avgpool":
        return F.avg_pool1d(x, kernel_size=kernel, padding=kernel // 2, stride=1)
    if pooling == "maxpool":
        return F.max_pool1d(x, kernel_size=kernel, padding=kernel // 2, stride=1)
    raise ValueError(pooling)


def select_snap(q_win, keys, scaling, budget: int, cfg: PolicyConfig) -> torch.Tensor:
    """SnapKV / PyramidKV selection: pooled window attention on the prefix + keep window."""
    b, H_kv, T, _ = keys.shape
    w = min(cfg.window, q_win.shape[2])
    attn = window_attention(q_win[:, :, -w:], keys, scaling)          # (b, H, w, T)
    s = attn[..., : T - w].sum(dim=2)                                 # (b, H, T-w)
    s = _pool(s, cfg.kernel, cfg.pooling)
    H = s.shape[1]
    s = s.view(b, H_kv, H // H_kv, T - w).mean(dim=2)                 # GQA: mean over group
    prefix_idx = _topk_sorted(s, budget - w)
    win_idx = torch.arange(T - w, T, device=keys.device).expand(b, H_kv, w)
    return torch.cat([prefix_idx, win_idx], dim=-1)


def select_sink_window(keys, budget: int, cfg: PolicyConfig) -> torch.Tensor:
    b, H_kv, T, _ = keys.shape
    s = min(cfg.sinks, budget)
    idx = torch.cat([torch.arange(0, s), torch.arange(T - (budget - s), T)]).to(keys.device)
    return idx.expand(b, H_kv, budget)


def select_normkv(keys, budget: int, cfg: PolicyConfig) -> torch.Tensor:
    # RoPE is a rotation and QK-norm precedes it, so ||k|| is identical pre/post RoPE.
    # direction "low": Devoto et al. -- low ||k|| correlates with high attention, so keep it.
    # direction "high": the opposite convention; use it to check the sign of the effect.
    sign = -1.0 if cfg.norm_direction == "low" else 1.0
    scores = sign * keys.float().norm(dim=-1)
    if cfg.norm_window > 0:
        scores[..., -cfg.norm_window:] = float("inf")
    return _topk_sorted(scores, budget)


def select_random(keys, budget: int, gen) -> torch.Tensor:
    """Unstructured control: uniform random subset per KV head."""
    b, H_kv, T, _ = keys.shape
    scores = torch.rand(b, H_kv, T, device=keys.device, generator=gen)
    return _topk_sorted(scores, budget)


def select_tova(q_win, keys, scaling, budget: int) -> torch.Tensor:
    b, H_kv, T, _ = keys.shape
    attn = window_attention(q_win[:, :, -1:], keys, scaling)          # (b, H, 1, T)
    s = attn[:, :, 0, :].mean(dim=1)                                  # head-averaged (b, T)
    s[:, -1] = float("inf")                                           # never evict current token
    idx = _topk_sorted(s, budget)                                     # (b, k)
    return idx.unsqueeze(1).expand(b, H_kv, budget)


def gather_kv(keys, values, idx):
    g = idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    return keys.gather(2, g), values.gather(2, g)


# --------------------------------------------------------------------------------------
# Hooks: window-query capture, streaming TOVA eviction, per-layer causal masks
# --------------------------------------------------------------------------------------

class KVController:
    """Installs forward hooks on every self-attention module.

    mode = None       : hooks inert
    mode = "capture"  : store post-RoPE window queries per layer (for one-shot policies)
    mode = "tova"     : after each layer's attention, evict to `stream_budget` via TOVA
    override_mask     : replace attention_mask with a per-layer mask sized to that
                        layer's cache (needed when layers hold different #tokens, e.g.
                        PyramidKV). All cached tokens are visible; new tokens are causal.
    """

    def __init__(self, model, cfg: PolicyConfig, capture_w: int):
        self.model = model
        self.cfg = cfg
        self.capture_w = capture_w
        self.layers = model.model.layers
        self.mode: Optional[str] = None
        self.override_mask = False
        self.stream_budget = 0
        self.q_win: Dict[int, torch.Tensor] = {}
        self._handles = []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            self._handles.append(attn.register_forward_pre_hook(partial(self._pre, i), with_kwargs=True))
            self._handles.append(attn.register_forward_hook(partial(self._post, i), with_kwargs=True))

    def remove(self):
        for h in self._handles:
            h.remove()

    # ---- pre-hook: per-layer additive mask ----
    def _pre(self, i, module, args, kwargs):
        if not self.override_mask:
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        cache = _cache_from_kwargs(kwargs)
        past = layer_len(cache, i)
        q = hs.shape[1]
        mask = torch.zeros(1, 1, q, past + q, dtype=hs.dtype, device=hs.device)
        if q > 1:
            tri = torch.triu(torch.ones(q, q, dtype=torch.bool, device=hs.device), diagonal=1)
            mask[..., past:].masked_fill_(tri, torch.finfo(hs.dtype).min)
        kwargs["attention_mask"] = mask
        return args, kwargs

    # ---- post-hook: capture or evict ----
    def _post(self, i, module, args, kwargs, output):
        if self.mode is None:
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pos = kwargs["position_embeddings"]
        if self.mode == "capture":
            self.q_win[i] = window_queries(module, hs, pos, self.capture_w).detach()
            return None
        if self.mode == "tova":
            cache = _cache_from_kwargs(kwargs)
            keys, values = layer_kv(cache, i)
            if keys.shape[2] <= self.stream_budget:
                return None
            q = window_queries(module, hs, pos, 1)
            idx = select_tova(q, keys, module.scaling, self.stream_budget)
            k2, v2 = gather_kv(keys, values, idx)
            set_layer_kv(cache, i, k2.contiguous(), v2.contiguous())
        return None


# --------------------------------------------------------------------------------------
# Model / data
# --------------------------------------------------------------------------------------

def load_model(model_id: str, quant: bool, double_quant: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    if quant:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=double_quant,
        )
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not quant and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tok, model


def corpus_cache_path(out_dir: str, model_id: str, split: str) -> str:
    """Cache path keyed by MODEL, not just split.

    Different tokenizers produce different token streams for the same text, so a shared
    cache would silently feed one model's ids to another -- and anchor offsets would
    point at different passages. Keyed this way, each model gets its own tokenisation.
    """
    tag = model_id.rstrip("/").replace("/", "__").replace(" ", "_")
    return os.path.join(out_dir, f"wikitext103_{split}_{tag}_ids.pt")


def load_corpus_ids(tok, split: str, cache_path: Optional[str]) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path)
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)
    if cache_path:
        torch.save(ids, cache_path)
    return ids


def sample_anchors(n_tokens: int, L: int, T: int, n: int, seed: int) -> List[int]:
    """Context-start offsets; each length gets its own draw (different text per length)."""
    rng = np.random.default_rng(seed + L)
    hi = n_tokens - L - T - 1
    return sorted(int(x) for x in rng.choice(hi, size=n, replace=False))


def sample_anchors_nested(n_tokens: int, max_L: int, T: int, n: int, seed: int) -> List[int]:
    """CONTINUATION-start offsets shared by every context length.

    Anchor `c` means: score tokens [c, c+T) given context [c-L, c). The scored tokens are
    then IDENTICAL across context lengths, so a 4k-vs-8k difference reflects how much
    context the policy had, not which passage it landed on. Use this whenever a claim is
    of the form "replicated at both context lengths".
    """
    rng = np.random.default_rng(seed)
    lo, hi = max_L, n_tokens - T - 1
    return sorted(int(x) for x in rng.choice(np.arange(lo, hi), size=n, replace=False))


# --------------------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------------------

def _model_call(model, **kw):
    try:
        return model(**kw)
    except TypeError:
        if "logits_to_keep" in kw:  # transformers < 4.45
            kw["num_logits_to_keep"] = kw.pop("logits_to_keep")
        return model(**kw)


@torch.inference_mode()
def score_continuation(model, ctrl: KVController, cache, first_logits, target, L: int,
                       use_mask_override: bool = True) -> float:
    """Mean NLL of `target` given a (possibly compressed) context cache.

    use_mask_override=False lets transformers build its own causal mask. That is only
    valid when every layer holds the same number of tokens (HF sizes the mask from layer
    0), but it gives an independent reference for checking our per-layer mask.
    """
    dev = first_logits.device
    nll = F.cross_entropy(first_logits.float(), target[:, :1].to(dev).view(-1), reduction="sum")
    if target.shape[1] > 1:
        inp = target[:, :-1].to(dev)
        q = inp.shape[1]
        past0 = layer_len(cache, 0)
        ctrl.override_mask = use_mask_override
        try:
            out = _model_call(
                model,
                input_ids=inp,
                past_key_values=cache,
                position_ids=torch.arange(L, L + q, device=dev).unsqueeze(0),
                cache_position=torch.arange(past0, past0 + q, device=dev),
                use_cache=True,
            )
        finally:
            ctrl.override_mask = False
        logits = out.logits[0].float()
        nll = nll + F.cross_entropy(logits, target[0, 1:].to(dev), reduction="sum")
    return (nll / target.shape[1]).item()


@torch.inference_mode()
def full_prefill_capture(model, ctrl: KVController, ctx: torch.Tensor):
    from transformers import DynamicCache
    ctrl.q_win = {}
    ctrl.mode = "capture"
    try:
        out = _model_call(model, input_ids=ctx, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
    finally:
        ctrl.mode = None
    cache = out.past_key_values
    n = len(ctrl.layers)
    kv = [layer_kv(cache, i) for i in range(n)]
    return kv, dict(ctrl.q_win), out.logits[:, -1]


def compress_one_shot(method: str, kv, q_win, modules, budget: int, cfg: PolicyConfig,
                      force_select: bool = False, seed: int = 0):
    n = len(kv)
    if method == "pyramidkv":
        budgets = pyramid_budgets(n, budget, cfg.window, cfg.pyramid_beta)
    else:
        budgets = [budget] * n
    new_kv = []
    for i, (k, v) in enumerate(kv):
        T = k.shape[2]
        Bi = budgets[i]
        if method == "normkv" and i < cfg.norm_skip_layers:
            Bi = T
        if force_select:  # selftest: run the selection/gather path even with no eviction
            Bi = min(Bi, T)
        elif Bi >= T:
            new_kv.append((k.clone(), v.clone()))
            continue
        scaling = modules[i].scaling
        if method in ("snapkv", "pyramidkv"):
            idx = select_snap(q_win[i], k, scaling, Bi, cfg)
        elif method == "sink_window":
            idx = select_sink_window(k, Bi, cfg)
        elif method == "recency":       # pure sliding window, no attention sinks
            idx = select_sink_window(k, Bi, replace(cfg, sinks=0))
        elif method == "normkv":
            idx = select_normkv(k, Bi, cfg)
        elif method == "tova":
            idx = select_tova(q_win[i], k, scaling, Bi)
        elif method == "random":
            gen = torch.Generator(device=k.device)
            gen.manual_seed(seed * 100003 + i)
            idx = select_random(k, Bi, gen)
        else:
            raise ValueError(method)
        k2, v2 = gather_kv(k, v, idx)
        new_kv.append((k2.contiguous(), v2.contiguous()))
    return new_kv


@torch.inference_mode()
def tova_stream_prefill(model, ctrl: KVController, ctx: torch.Tensor, budget: int, chunk: int):
    """Sequential TOVA: prefill `budget` tokens, then evict after every `chunk` tokens."""
    from transformers import DynamicCache
    L = ctx.shape[1]
    dev = ctx.device
    cache = DynamicCache()
    out = _model_call(model, input_ids=ctx[:, :budget], past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    ctrl.mode, ctrl.stream_budget, ctrl.override_mask = "tova", budget, True
    try:
        for s in range(budget, L, chunk):
            e = min(s + chunk, L)
            past0 = layer_len(cache, 0)
            out = _model_call(
                model,
                input_ids=ctx[:, s:e],
                past_key_values=cache,
                position_ids=torch.arange(s, e, device=dev).unsqueeze(0),
                cache_position=torch.arange(past0, past0 + (e - s), device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
    finally:
        ctrl.mode, ctrl.override_mask = None, False
    return cache, out.logits[:, -1]


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def holm(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, (m - rank) * pvals[j])
        adj[j] = min(running, 1.0)
    return adj.tolist()


def summarize(rows: List[dict], comparators: Sequence[str], primary_budget: float) -> List[dict]:
    from scipy import stats
    by = {}
    for r in rows:
        by[(int(r["L"]), r["method"], round(float(r["budget"]), 6), int(r["anchor_idx"]))] = float(r["ppl"])
    Ls = sorted({k[0] for k in by})
    conds = sorted({(k[1], k[2]) for k in by if k[1] != "full"})
    out = []
    for L in Ls:
        for method, b in conds:
            for comp in comparators:
                if comp == method:
                    continue
                comp_b = 1.0 if comp == "full" else b
                anchors = sorted(a for (LL, m, bb, a) in by if LL == L and m == method and bb == b)
                pairs = [(by[(L, method, b, a)], by[(L, comp, comp_b, a)])
                         for a in anchors if (L, comp, comp_b, a) in by]
                if len(pairs) < 2:
                    continue
                d = np.array([x - y for x, y in pairs])
                n = len(d)
                mean, sd = d.mean(), d.std(ddof=1)
                se = sd / math.sqrt(n)
                tcrit = stats.t.ppf(0.975, n - 1)
                tstat = mean / se if se > 0 else float("nan")
                p = 2 * stats.t.sf(abs(tstat), n - 1) if se > 0 else float("nan")
                out.append(dict(
                    L=L, method=method, budget=b, comparator=comp, n=n,
                    ppl_method=float(np.mean([x for x, _ in pairs])),
                    ppl_comparator=float(np.mean([y for _, y in pairs])),
                    delta_mean=mean, delta_sd=sd, ci_lo=mean - tcrit * se, ci_hi=mean + tcrit * se,
                    t_crit=tcrit, t=tstat, p=p, d_z=(mean / sd if sd > 0 else float("nan")),
                    primary=abs(b - primary_budget) < 1e-9, p_holm="", sig="",
                ))
    # Holm across context lengths at the primary budget, per (method, comparator) family
    fams = {}
    for r in out:
        if r["primary"]:
            fams.setdefault((r["method"], r["comparator"]), []).append(r)
    for fam in fams.values():
        adj = holm([r["p"] for r in fam])
        for r, a in zip(fam, adj):
            r["p_holm"] = a
            r["sig"] = "***" if a < 0.001 else "**" if a < 0.01 else "*" if a < 0.05 else "ns"
    return out


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------

RAW_FIELDS = ["L", "method", "budget", "anchor_idx", "anchor_offset", "nll", "ppl",
              "kept_mean", "kept_min", "kept_max", "seconds"]


def read_raw(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_raw(path: str, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def write_summary(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def run(args):
    torch.manual_seed(args.seed)
    cfg = PolicyConfig(window=args.window, kernel=args.kernel, pooling=args.pooling,
                       pyramid_beta=args.pyramid_beta, sinks=args.sinks, norm_window=args.norm_window,
                       norm_skip_layers=args.norm_skip_layers, tova_chunk=args.tova_chunk,
                       norm_direction=args.norm_direction)
    os.makedirs(args.out_dir, exist_ok=True)
    tok, model = load_model(args.model, quant=not args.no_quant, double_quant=args.double_quant)
    dev = next(model.parameters()).device
    modules = [layer.self_attn for layer in model.model.layers]
    ctrl = KVController(model, cfg, capture_w=max(cfg.window, 1))

    ids = load_corpus_ids(tok, args.split, corpus_cache_path(args.out_dir, args.model, args.split))
    n_layers, cfg_m = len(model.model.layers), model.config
    print(f"[model] {args.model}: {n_layers} layers, {cfg_m.num_attention_heads} q-heads / "
          f"{getattr(cfg_m, 'num_key_value_heads', cfg_m.num_attention_heads)} kv-heads, "
          f"QK-norm={hasattr(modules[0], 'q_norm')}, vocab={cfg_m.vocab_size:,}")
    print(f"[data] {ids.numel():,} tokens in WikiText-103/{args.split} "
          f"(tokenised by {args.model})")

    if args.anchors_file:
        with open(args.anchors_file) as f:
            anchors_all = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}
    elif args.nested_anchors:
        shared = sample_anchors_nested(ids.numel(), max(args.context_lengths), args.eval_tokens,
                                       args.n_anchors, args.seed)
        anchors_all = {L: shared for L in args.context_lengths}
    else:
        anchors_all = {L: sample_anchors(ids.numel(), L, args.eval_tokens, args.n_anchors, args.seed)
                       for L in args.context_lengths}
    with open(os.path.join(args.out_dir, "anchors_used.json"), "w") as f:
        json.dump({str(k): v for k, v in anchors_all.items()}, f, indent=1)

    methods = list(args.methods)
    for L in args.context_lengths:
        anchors = anchors_all[L][: args.n_anchors]
        raw_path = os.path.join(args.out_dir, f"raw_L{L}.csv")
        prior = read_raw(raw_path)
        # Resume keys on anchor_idx, so rows written under a DIFFERENT anchor draw would be
        # silently accepted as done. Refuse rather than mix two anchor sets in one table.
        for r in prior:
            ai_, off_ = int(r["anchor_idx"]), int(r["anchor_offset"])
            if ai_ < len(anchors) and anchors[ai_] != off_:
                raise SystemExit(
                    f"\n{raw_path} was written with a different anchor set "
                    f"(anchor {ai_} was offset {off_}, now {anchors[ai_]}).\n"
                    f"Resuming would mix two anchor draws into one table and break pairing.\n"
                    f"Use a fresh --out-dir for this configuration.")
        done = {(r["method"], round(float(r["budget"]), 6), int(r["anchor_idx"])) for r in prior}
        budgets_tok = {b: int(round(b * L)) for b in args.budgets}
        for b, B in budgets_tok.items():
            if ("snapkv" in methods or "pyramidkv" in methods) and B <= cfg.window:
                raise ValueError(f"budget {b} at L={L} gives {B} tokens <= window {cfg.window}")

        for ai, off in enumerate(anchors):
            # nested: `off` is where the SCORED tokens start, context is the L before it
            c0 = off - L if args.nested_anchors else off
            ctx = ids[c0: c0 + L].unsqueeze(0).to(dev)
            tgt = ids[c0 + L: c0 + L + args.eval_tokens].unsqueeze(0)

            todo_one_shot = [(m, b) for m in methods if m in ONE_SHOT_METHODS for b in args.budgets
                             if (m, round(b, 6), ai) not in done]
            need_full = "full" in methods and ("full", 1.0, ai) not in done
            if todo_one_shot or need_full:
                t0 = time.time()
                kv, q_win, first_logits = full_prefill_capture(model, ctrl, ctx)
                t_prefill = time.time() - t0
                for m, b in todo_one_shot:
                    t1 = time.time()
                    new_kv = compress_one_shot(m, kv, q_win, modules, budgets_tok[b], cfg,
                                               seed=args.seed * 1000 + ai)
                    kept = [k.shape[2] for k, _ in new_kv]
                    cache = build_cache(new_kv)
                    del new_kv
                    nll = score_continuation(model, ctrl, cache, first_logits, tgt, L)
                    del cache
                    _log(raw_path, L, m, b, ai, off, nll, kept, t_prefill + time.time() - t1)
                if need_full:  # last: reuses (and mutates) the full cache
                    t1 = time.time()
                    cache = build_cache(kv)
                    kept = [k.shape[2] for k, _ in kv]
                    nll = score_continuation(model, ctrl, cache, first_logits, tgt, L)
                    _log(raw_path, L, "full", 1.0, ai, off, nll, kept, t_prefill + time.time() - t1)
                    del cache
                del kv, q_win
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if "tova_stream" in methods:
                for b in args.budgets:
                    if ("tova_stream", round(b, 6), ai) in done:
                        continue
                    t0 = time.time()
                    cache, fl = tova_stream_prefill(model, ctrl, ctx, budgets_tok[b], cfg.tova_chunk)
                    kept = [layer_len(cache, i) for i in range(len(modules))]
                    nll = score_continuation(model, ctrl, cache, fl, tgt, L)
                    _log(raw_path, L, "tova_stream", b, ai, off, nll, kept, time.time() - t0)
                    del cache
            print(f"[L={L}] anchor {ai + 1}/{len(anchors)} done")

    # ---- summary over everything on disk ----
    all_rows = []
    for L in args.context_lengths:
        all_rows += read_raw(os.path.join(args.out_dir, f"raw_L{L}.csv"))
    comps = [c for c in ("full", "snapkv", "sink_window") if c in {r["method"] for r in all_rows}]
    summ = summarize(all_rows, comps, args.primary_budget)
    write_summary(os.path.join(args.out_dir, "summary_paired.csv"), summ)
    _print_table(all_rows)
    print(f"\n[done] raw rows -> {args.out_dir}/raw_L*.csv ; paired stats -> {args.out_dir}/summary_paired.csv")


def _log(path, L, m, b, ai, off, nll, kept, secs):
    row = dict(L=L, method=m, budget=b, anchor_idx=ai, anchor_offset=off, nll=f"{nll:.6f}",
               ppl=f"{math.exp(nll):.6f}", kept_mean=f"{np.mean(kept):.2f}", kept_min=min(kept),
               kept_max=max(kept), seconds=f"{secs:.2f}")
    append_raw(path, row)
    print(f"  L={L} {m:<12} b={b:<5} anchor={ai:>2} ppl={math.exp(nll):8.3f} kept={np.mean(kept):7.1f}")


def _print_table(rows):
    agg = {}
    for r in rows:
        agg.setdefault((int(r["L"]), r["method"], float(r["budget"])), []).append(float(r["ppl"]))
    print("\nMean PPL (descriptive)")
    for L in sorted({k[0] for k in agg}):
        print(f"  L = {L}")
        for (LL, m, b), v in sorted(agg.items()):
            if LL == L:
                print(f"    {m:<12} budget={b:<5}  PPL={np.mean(v):8.3f}  (n={len(v)})")


# --------------------------------------------------------------------------------------
# Self-test (run on the real model before a long job)
# --------------------------------------------------------------------------------------

@torch.inference_mode()
def selftest(args):
    """Checks that do not depend on eviction quality:
      1. Reconstructed window attention == the model's own attention probabilities.
      2. Every one-shot policy at budget == L reproduces full-cache NLL.
      3. A heterogeneous per-layer cache (PyramidKV) gives the same NLL whether the
         continuation is scored in one chunk (per-layer masks) or token by token.
      4. tova_stream with budget >= L reproduces full-cache NLL.
    """
    cfg = PolicyConfig(window=args.window, kernel=args.kernel, pooling=args.pooling,
                       pyramid_beta=args.pyramid_beta, sinks=args.sinks, tova_chunk=args.tova_chunk)
    tok, model = load_model(args.model, quant=not args.no_quant, double_quant=args.double_quant)
    dev = next(model.parameters()).device
    modules = [layer.self_attn for layer in model.model.layers]
    ctrl = KVController(model, cfg, capture_w=cfg.window)
    L, T = args.selftest_len, 32
    g = torch.Generator().manual_seed(0)
    vocab = min(model.config.vocab_size, 30000)
    ids = torch.randint(100, vocab, (1, L + T), generator=g)
    if not args.selftest_random_tokens:
        try:
            ids = load_corpus_ids(tok, args.split, None)[1000: 1000 + L + T].unsqueeze(0)
        except Exception as e:  # offline etc.
            print(f"[selftest] corpus unavailable ({e}); using random tokens")
    ctx, tgt = ids[:, :L].to(dev), ids[:, L:]
    ok = True

    # 1. attention reconstruction vs eager attention
    kv, q_win, fl = full_prefill_capture(model, ctrl, ctx)
    impl = model.config._attn_implementation

    def _set_impl(name):
        if hasattr(model, "set_attn_implementation"):  # transformers >= 4.56
            model.set_attn_implementation(name)
        else:
            model.config._attn_implementation = name

    try:
        _set_impl("eager")
        ref = model(input_ids=ctx, output_attentions=True, use_cache=False, logits_to_keep=1).attentions
    finally:
        _set_impl(impl)
    worst = 0.0
    mean_diff = 0.0
    n_layers = len(modules)
    for i in range(n_layers):
        mine = window_attention(q_win[i], kv[i][0], modules[i].scaling)
        theirs = ref[i][:, :, -cfg.window:, :].float()
        d = (mine - theirs).abs()
        worst = max(worst, d.max().item())
        mean_diff += d.mean().item() / n_layers
    # NF4 quantizes q_proj/k_proj; the max over all (layer, head, wq, k) probabilities
    # can be several percent on a boundary case even when mean is ~1e-4. What matters is
    # that the derived token selection matches -- check #2 tests that end-to-end.
    tol_max = 1e-1 if not args.no_quant else 5e-3
    tol_mean = 1e-3 if not args.no_quant else 1e-5
    print(f"[selftest] 1. window-attention max|diff| = {worst:.2e} (tol {tol_max:.0e}), "
          f"mean|diff| = {mean_diff:.2e} (tol {tol_mean:.0e})")
    ok &= worst < tol_max and mean_diff < tol_mean

    # 2. budget == L equals full (selection + sort + gather path forced to run)
    full = score_continuation(model, ctrl, build_cache(kv), fl, tgt, L)
    for m in ONE_SHOT_METHODS:
        if m == "pyramidkv":
            continue  # pyramid budgets are never uniform; covered by check 3
        new_kv = compress_one_shot(m, kv, q_win, modules, L, cfg, force_select=True)
        nll = score_continuation(model, ctrl, build_cache(new_kv), fl, tgt, L)
        d = abs(nll - full)
        print(f"[selftest] 2. {m:<12} budget=L  |dNLL| = {d:.2e}")
        ok &= d < 1e-3

    # 3. Is the per-layer causal mask correct, and how much is just arithmetic?
    #
    # Three measurements, because a single chunked-vs-stepwise number cannot separate
    # "the mask is wrong" from "bf16 sums in a different order".
    #
    #   3a  uniform cache, chunked: OUR mask vs TRANSFORMERS' own mask. Same kernel,
    #       same shapes, same arithmetic -- the only difference is who built the mask.
    #       This is a correctness test and must be ~0. It is valid only on a uniform
    #       cache, because HF sizes its mask from layer 0 and would mis-shape a
    #       heterogeneous one (which is the reason our override exists at all).
    #   3b  uniform cache, chunked (our mask) vs stepwise with NO mask at all (q=1, where
    #       HF skips masking entirely and each layer attends over its own cache). This is
    #       the model's numerical floor for chunked vs stepwise.
    #   3c  heterogeneous (PyramidKV) cache, same comparison as 3b. Our mask is the same
    #       construction with a per-layer `past`, so if 3a is clean, any excess here over
    #       3b is arithmetic -- driven by the small top layers, which push attention into
    #       a more extreme regime than a uniform cache of the same mean size.
    B = max(cfg.window + 8, L // 4)

    def stepwise_nll(sel_kv):
        """Reference path: one token at a time, transformers' own masking (none at q=1)."""
        cache = build_cache(sel_kv)
        nll = F.cross_entropy(fl.float(), tgt[:, :1].to(dev).view(-1), reduction="sum")
        for j in range(T - 1):
            past0 = layer_len(cache, 0)
            out = model(input_ids=tgt[:, j: j + 1].to(dev), past_key_values=cache,
                        use_cache=True,
                        position_ids=torch.tensor([[L + j]], device=dev),
                        cache_position=torch.tensor([past0], device=dev))
            nll = nll + F.cross_entropy(out.logits[0].float(),
                                        tgt[0, j + 1: j + 2].to(dev), reduction="sum")
        return (nll / T).item()

    uni = compress_one_shot("snapkv", kv, q_win, modules, B, cfg)
    pk = compress_one_shot("pyramidkv", kv, q_win, modules, B, cfg)
    lens = [k.shape[2] for k, _ in pk]

    ours_uni = score_continuation(model, ctrl, build_cache(uni), fl, tgt, L)
    hf_uni = score_continuation(model, ctrl, build_cache(uni), fl, tgt, L,
                                use_mask_override=False)
    d_mask = abs(ours_uni - hf_uni)
    d_uni = abs(ours_uni - stepwise_nll(uni))
    d_pyr = abs(score_continuation(model, ctrl, build_cache(pk), fl, tgt, L) - stepwise_nll(pk))

    limit = max(3 * d_uni, 2e-3)
    print(f"[selftest] 3a. our per-layer mask vs transformers' own mask (uniform cache, "
          f"identical arithmetic): |dNLL| = {d_mask:.2e}  <- correctness")
    print(f"[selftest] 3b. chunked vs stepwise, uniform cache:     |dNLL| = {d_uni:.2e}  "
          f"<- numerical floor")
    print(f"[selftest] 3c. chunked vs stepwise, pyramid cache:     |dNLL| = {d_pyr:.2e}  "
          f"({d_pyr / d_uni if d_uni else float('inf'):.1f}x floor, limit {limit:.2e})")
    print(f"[selftest]     pyramid layer lens {min(lens)}..{max(lens)} "
          f"(mean {np.mean(lens):.1f}, target {B}); "
          f"chunked-path PPL impact ~{100 * (math.exp(d_pyr) - 1):.2f}%")
    ok &= d_mask < 1e-4                      # the mask must be right
    ok &= d_pyr < limit                      # the rest is allowed to be arithmetic
    ok &= abs(np.mean(lens) - B) < 1e-6

    # 4. tova_stream with no eviction
    cache, fl2 = tova_stream_prefill(model, ctrl, ctx, L, cfg.tova_chunk)
    d = abs(score_continuation(model, ctrl, cache, fl2, tgt, L) - full)
    print(f"[selftest] 4. tova_stream budget=L |dNLL| = {d:.2e}")
    ok &= d < 1e-3

    # 4b. tova_stream prefill-in-chunks path at a real budget runs and keeps B tokens
    cache, fl2 = tova_stream_prefill(model, ctrl, ctx, B, cfg.tova_chunk)
    lens = {layer_len(cache, i) for i in range(len(modules))}
    s = score_continuation(model, ctrl, cache, fl2, tgt, L)
    print(f"[selftest] 4b. tova_stream budget={B}: layer lens {lens}, NLL {s:.3f} (full {full:.3f})")
    ok &= lens == {B}

    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1



# --------------------------------------------------------------------------------------
# Diagnostic: does the NormKV premise hold on this architecture?
# --------------------------------------------------------------------------------------

def _rank(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(dim=-1).argsort(dim=-1).float()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rank correlation along the last dim."""
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(-1, keepdim=True)
    rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1) + 1e-12)


@torch.inference_mode()
def diagnose(args):
    """Does the NormKV premise hold on this architecture, and if not, why does it fail?

    Reports four things per layer, averaged over anchors:
      rho     Spearman(||k||, attention received). Devoto et al. predict clearly NEGATIVE.
      CV      spread of ||k|| across tokens. Qwen3 RMSNorms keys per head before RoPE,
              which compresses this; where CV collapses, rho has nothing to work with.
      recall  share of the observation window's prefix attention landing on kept tokens.
      ctx_err relative L2 error of the context vector rebuilt from the kept subset.

    recall and ctx_err answer different questions. An UNBIASED subset (random) can have
    mediocre recall yet small ctx_err, because what it drops resembles what it keeps. A
    subset chosen along a direction correlated with value content can have HIGHER recall
    and still land further from the true context vector -- error that renormalisation
    does not fix and budget does not shrink.
    """
    cfg = PolicyConfig(window=args.window, kernel=args.kernel, pooling=args.pooling,
                       pyramid_beta=args.pyramid_beta, sinks=args.sinks,
                       norm_direction=args.norm_direction)
    tok, model = load_model(args.model, quant=not args.no_quant, double_quant=args.double_quant)
    dev = next(model.parameters()).device
    modules = [layer.self_attn for layer in model.model.layers]
    ctrl = KVController(model, cfg, capture_w=cfg.window)
    ids = load_corpus_ids(tok, args.split, corpus_cache_path(args.out_dir, args.model, args.split))
    L = args.context_lengths[0]
    n = args.n_anchors
    anchors = sample_anchors(ids.numel(), L, args.eval_tokens, n, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    budgets = list(args.diag_budgets)

    # stats[layer][key] -> list over anchors.  Named keys, so adding a metric later
    # cannot silently desynchronise the readers below.
    stats: Dict[int, Dict[str, List[float]]] = {}

    def put(layer: int, key: str, value: float):
        stats.setdefault(layer, {}).setdefault(key, []).append(value)

    gen = torch.Generator(device=dev)
    for ai, off in enumerate(anchors):
        ctx = ids[off: off + L].unsqueeze(0).to(dev)
        kv, q_win, _ = full_prefill_capture(model, ctrl, ctx)
        for i in range(len(modules)):
            k, v_all = kv[i]
            T, w = k.shape[2], q_win[i].shape[2]
            P = T - w
            attn = window_attention(q_win[i], k, modules[i].scaling)
            H, H_kv = attn.shape[1], k.shape[1]
            # GQA: query head h reads kv head h // group, so view(H_kv, group) is the
            # correct regrouping (same convention as repeat_kv and select_snap).
            mass = attn[..., :P].sum(dim=2).view(1, H_kv, H // H_kv, P).mean(dim=2)
            knorm = k[:, :, :P, :].float().norm(dim=-1)
            put(i, "rho", _spearman(knorm, mass).mean().item())
            put(i, "cv", (knorm.std(dim=-1) / knorm.mean(dim=-1)).mean().item())

            total = mass.sum(dim=-1).clamp_min(1e-9)
            vals = v_all[:, :, :P, :].float()
            o_full = ((mass / total.unsqueeze(-1)).unsqueeze(-1) * vals).sum(dim=-2)
            o_ref = o_full.norm(dim=-1).clamp_min(1e-9)

            def ctx_err(sel):
                pm = mass.gather(-1, sel)
                pm = pm / pm.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                vv = vals.gather(-2, sel.unsqueeze(-1).expand(-1, -1, -1, vals.shape[-1]))
                o = (pm.unsqueeze(-1) * vv).sum(dim=-2)
                return ((o - o_full).norm(dim=-1) / o_ref).mean().item()

            for f in budgets:
                B = max(1, int(round(f * P)))
                low = knorm.topk(B, dim=-1, largest=False).indices     # NormKV
                orc = mass.topk(B, dim=-1).indices                     # oracle
                gen.manual_seed(args.seed * 100003 + ai * 97 + i)      # reproducible control
                rnd = torch.rand(mass.shape, device=dev, generator=gen).topk(B, dim=-1).indices
                R = min(128, P)
                put(i, f"recall_norm@{f}", (mass.gather(-1, low).sum(-1) / total).mean().item())
                put(i, f"recall_orc@{f}", (mass.gather(-1, orc).sum(-1) / total).mean().item())
                put(i, f"recall_rnd@{f}", (mass.gather(-1, rnd).sum(-1) / total).mean().item())
                put(i, f"ret_norm@{f}", ((low >= P - R).sum(-1).float() / R).mean().item())
                put(i, f"ret_orc@{f}", ((orc >= P - R).sum(-1).float() / R).mean().item())
                put(i, f"pos_norm@{f}", (low.float().mean(-1) / P).mean().item())
                put(i, f"err_norm@{f}", ctx_err(low))
                put(i, f"err_rnd@{f}", ctx_err(rnd))
                put(i, f"err_orc@{f}", ctx_err(orc))
        del kv, q_win
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[diagnose] anchor {ai + 1}/{n} done")

    layers = sorted(stats)
    per_layer = {i: {k: float(np.mean(v)) for k, v in stats[i].items()} for i in layers}
    agg = lambda key: float(np.mean([per_layer[i][key] for i in layers]))
    # anchor-to-anchor spread of the layer-averaged rho, so the headline has an error bar
    rho_by_anchor = [float(np.mean([stats[i]["rho"][a] for i in layers])) for a in range(n)]

    keys = ["rho", "cv"] + [f"{m}@{f}" for f in budgets for m in
                            ("recall_norm", "recall_rnd", "recall_orc", "ret_norm", "ret_orc",
                             "pos_norm", "err_norm", "err_rnd", "err_orc")]
    path = os.path.join(args.out_dir, "knorm_diagnostic.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["layer"] + keys)
        for i in layers:
            wr.writerow([i] + [f"{per_layer[i][k]:.4f}" for k in keys])

    print(f"\n{'layer':>6} {'rho(||k||, attn)':>18} {'CV(||k||)':>12}")
    for i in layers:
        print(f"{i:>6} {per_layer[i]['rho']:>18.4f} {per_layer[i]['cv']:>12.4f}")
    sd = float(np.std(rho_by_anchor, ddof=1)) if n > 1 else float("nan")
    print(f"\nmean rho = {agg('rho'):+.4f} (SD across anchors {sd:.4f}, n={n})   "
          f"mean CV = {agg('cv'):.4f}   L={L}")
    print("Devoto et al.'s premise predicts a clearly NEGATIVE rho.")

    print("\nAttention-mass recall @ budget (share of prefix attention on kept tokens):")
    print(f"  {'budget':>8} {'NormKV':>9} {'random':>9} {'oracle':>9}   verdict")
    for f in budgets:
        rn, rr, ro = agg(f"recall_norm@{f}"), agg(f"recall_rnd@{f}"), agg(f"recall_orc@{f}")
        lift = rn / rr if rr > 0 else float("nan")
        v = "no better than chance" if lift < 1.15 else f"{lift:.2f}x chance, oracle {ro / rr:.1f}x"
        print(f"  {f:>8.2f} {rn:>9.3f} {rr:>9.3f} {ro:>9.3f}   {v}")

    print("\nPositional signature (share of last 128 prefix tokens kept; random keeps the")
    print("budget fraction by construction, so below that = deleting recency):")
    print(f"  {'budget':>8} {'NormKV':>9} {'random':>9} {'oracle':>9} {'meanpos':>9}")
    for f in budgets:
        print(f"  {f:>8.2f} {agg(f'ret_norm@{f}'):>9.3f} {f:>9.3f} "
              f"{agg(f'ret_orc@{f}'):>9.3f} {agg(f'pos_norm@{f}'):>9.3f}")
    print("  meanpos: mean normalised position of kept tokens (0.5 = positionally neutral)")

    print("\nContext-vector error (relative L2 distance from the uncompressed context")
    print("vector) -- what recall cannot tell you:")
    print(f"  {'budget':>8} {'NormKV':>9} {'random':>9} {'oracle':>9}   verdict")
    for f in budgets:
        en, er, eo = agg(f"err_norm@{f}"), agg(f"err_rnd@{f}"), agg(f"err_orc@{f}")
        rn, rr = agg(f"recall_norm@{f}"), agg(f"recall_rnd@{f}")
        if en > er and rn > rr:
            v = "HIGHER recall, HIGHER error => biased subset"
        elif en > er:
            v = "worse than chance"
        else:
            v = f"{er / en:.2f}x better than chance"
        print(f"  {f:>8.2f} {en:>9.3f} {er:>9.3f} {eo:>9.3f}   {v}")

    print(f"\n[diagnose] per-layer values -> {path}")
    return 0


# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--no-quant", action="store_true", help="load bf16 instead of NF4 (won't fit 16GB for 8B)")
    p.add_argument("--double-quant", action="store_true", help="bnb_4bit_use_double_quant=True")
    p.add_argument("--split", default="test", choices=["test", "validation", "train"])
    p.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--n-anchors", type=int, default=20)
    p.add_argument("--anchors-file", default=None, help="JSON {L: [offsets]} to reuse existing anchors")
    p.add_argument("--eval-tokens", type=int, default=256, help="continuation tokens scored per anchor")
    p.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.50])
    p.add_argument("--primary-budget", type=float, default=0.05)
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(ALL_METHODS))
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--pooling", default="avgpool", choices=["avgpool", "maxpool"])
    p.add_argument("--pyramid-beta", type=float, default=20.0)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--norm-window", type=int, default=0)
    p.add_argument("--norm-skip-layers", type=int, default=0)
    p.add_argument("--norm-direction", default="low", choices=["low", "high"],
                   help="keep lowest ||k|| (Devoto et al.) or highest")
    p.add_argument("--nested-anchors", action="store_true",
                   help="score the SAME continuations at every context length (recommended for "
                        "cross-length claims); anchors are continuation starts, not context starts")
    p.add_argument("--diag-budgets", type=float, nargs="+", default=[0.05, 0.5],
                   help="budgets at which to report attention-mass recall in --diagnose")
    p.add_argument("--diagnose", action="store_true",
                   help="report per-layer ||k|| spread and its rank-correlation with received "
                        "attention -- tests the premise NormKV relies on")
    p.add_argument("--tova-chunk", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results_extended")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--selftest-len", type=int, default=512)
    p.add_argument("--selftest-random-tokens", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.selftest:
        sys.exit(selftest(a))
    if a.diagnose:
        sys.exit(diagnose(a))
    sys.exit(run(a) or 0)