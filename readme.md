# When Attention Scores Earn Their Keep — and When They Don't: A Paired Evaluation of KV-Cache Eviction Across Two Architectures

Code and data for the paper. Everything reported there comes from one script,
`kv_eviction.py`, run once per model.
---

## Contents

```
kv_eviction.py     the whole harness: policies, runner, self-test, diagnostics
build_tables.py             builds all 11 tables from the result CSVs
results_qwen3_512/          Qwen3-8B outputs (anchors, raw rows, paired summary)
results_llama31_512/        Llama-3.1-8B outputs
context_scaling.py          context scaling using perplexity
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers bitsandbytes datasets numpy scipy
```

## Quick start

Run the self-test first. It is fast and it will catch a broken environment.

```bash
python kv_eviction.py --model Qwen/Qwen3-8B --selftest
```

Then the production run:

```bash
python kv_eviction.py \
  --model Qwen/Qwen3-8B \
  --context-lengths 4096 8192 \
  --n-anchors 20 \
  --eval-tokens 512 \
  --budgets 0.05 0.10 0.20 0.50 \
  --primary-budget 0.05 \
  --methods full snapkv pyramidkv tova normkv sink_window recency random \
  --seed 0 \
  --out-dir results_qwen3_512
```

The second model is the same command with `--model meta-llama/Llama-3.1-8B` and
`--out-dir results_llama31_512`. Nothing else changes.

A run is resumable. If it stops, re-running the same command picks up where it left off by
reading the existing `raw_L{L}.csv`.

---

## What the policies do

All eight run through the same compression-and-reposition path and differ only in which
cache entries they keep. Budgets match exactly; where a policy keeps an observation or
recent window, that window counts against its budget.

| `--methods` name | What it keeps |
|---|---|
| `full` | nothing evicted; the reference arm for every paired comparison |
| `snapkv` | top-scoring prefix entries by observation-window attention, pooled, plus the window itself (Li et al., 2024) |
| `pyramidkv` | same scoring, but a descending per-layer budget instead of a uniform one (Cai et al., 2024) |
| `tova` | scores from one query only — the final prompt token — averaged across heads, one token set per layer (Oren et al., 2024) |
| `normkv` | the entries whose key vectors have the smallest L2 norm; no attention computed (Devoto et al., 2024) |
| `sink_window` | the first 4 tokens plus the most recent B-4 (Xiao et al., 2023) |
| `recency` | the most recent B tokens; isolates what the sink tokens are worth |
| `random` | a uniform random prefix subset at the same budget; the floor |
| `tova_stream` | streaming TOVA, evicting during prefill rather than once at the end |

`tova_stream` is not in the paper. It exists to confirm that one-shot compression at the
end of prefill and step-by-step eviction agree, and it is much slower.

### Policy knobs

Defaults match each method's publication.

```
--window 32           observation window size (SnapKV, PyramidKV)
--kernel 7            pooling kernel width
--pooling avgpool     avgpool | maxpool
--sinks 4             sink tokens for sink_window
--pyramid-beta 20.0   PyramidKV's top/bottom layer ratio parameter
--norm-direction low  keep low key norm (as published) or high
--norm-window 0       optionally force-keep the last N positions in normkv
--tova-chunk 32       chunk size for tova_stream only
```

## Outputs

**`anchors_used.json`** — `{context_length: [token offsets]}`.

**`raw_L{L}.csv`** — one row per anchor, budget and policy:

```
L, method, budget, anchor_idx, anchor_offset, nll, ppl,
kept_mean, kept_min, kept_max, seconds
```

`kept_min`/`kept_max` differ from `kept_mean` only for `pyramidkv`; for every other policy
all three are equal, which is a cheap way to confirm budgets really did match.

**`summary_paired.csv`** — paired statistics, one row per (context length, method, budget,
comparator):

```
L, method, budget, comparator, n, ppl_method, ppl_comparator,
delta_mean, delta_sd, ci_lo, ci_hi, t_crit, t, p, d_z, primary, p_holm, sig
```

Confidence intervals use the t critical value at df = n-1 (2.093 at n = 20), not the
large-sample 1.96, which would make intervals about 7% too narrow at this sample size.
`d_z` is Cohen's d for paired data: `delta_mean / delta_sd`.

`primary` marks rows at `--primary-budget`. Holm correction is applied **across context
lengths within each (method, comparator) family**, with m = 2 — never pooled across models.
Each model is its own family. Non-primary rows are exploratory and carry no `p_holm`.

**`knorm_diagnostic.csv`** (with `--diagnose`) — per layer: Spearman rho between key norm and
received attention, coefficient of variation of ||k||, attention-mass recall for
norm/random/oracle selection, retained-position statistics, and context-vector
reconstruction error. This is the evidence behind Section 5.6.

---

## Reproducing the paper's tables

```bash
python3 build_tables.py --qwen3 results_qwen3_512 --llama results_llama31_512
```

Emits TSV (for pasting into Word) and LaTeX booktabs. It refuses input whose pairing is
broken — a policy missing an anchor that others have — rather than silently averaging over
a different set.

---

## Hardware and quantization

One consumer 16 GB GPU under WSL2. Weights load in 4-bit NF4 through bitsandbytes with bf16
compute.

The script sets `device_map={"": 0}` and asserts that no parameter lands on `meta` or CPU.
Silent CPU offload was an early failure here: everything runs, results look fine, and the
latency and memory numbers are meaningless. Prefill uses `logits_to_keep=1` so no large
logit tensor is materialised.

`--no-quant` loads bf16 instead. An 8B model will not fit in 16 GB that way.

**Expect absolute perplexities to differ** across bitsandbytes versions and GPUs. Paired
differences within a model are stable, because both arms share the same weights and the same
anchors, and those are what every claim rests on. We say so in the paper rather than
presenting NF4 numbers as if they were full precision.

---

## Diagnostics

```bash
python kv_eviction.py --model Qwen/Qwen3-8B --diagnose \
  --n-anchors 20 --diag-budgets 0.05 0.5 --out-dir results_qwen3_512
```

This is what produced the norm-eviction analysis. It measures, per layer: the Spearman
correlation between ||k|| and attention actually received, the spread of ||k||, how much
attention mass each selection rule recovers against an oracle, where the retained positions
sit in the sequence, and the error in the reconstructed context vector.

Headline numbers from the paper: rho = -0.342 on Llama-3.1-8B (negative in all 32 layers) and
-0.081 on Qwen3-8B (wrong sign in 11 of 36 layers). Qwen3 normalizes queries and keys per
head before the rotary embedding, which narrows the spread of ||k||; Llama-3.1 does not.
Since the rotary embedding is a rotation, key norms are identical before and after it, so it
does not matter where you measure them.

`--n-anchors` is honoured here. An earlier version silently capped it at 4, which is the kind
of bug that makes a diagnostic look noisy for no reason.

---

<!-- ## Citation

```
[TODO: paste the BibTeX entry once the paper has a venue and page numbers]
``` -->
## License

This project uses third-party datasets and pretrained model weights that are subject to their respective licenses:

* **WikiText-103** — distributed under the **Creative Commons Attribution-ShareAlike (CC BY-SA)** license, consistent with its Wikipedia-derived provenance.
* **Qwen3** — model weights are subject to the applicable **Qwen3 license and terms of use**.
* **Llama-3.1** — model weights are subject to the applicable **Llama 3.1 Community License and terms of use**.

These third-party licenses apply to the respective datasets and model weights and are **not superseded by the MIT License for this repository**. Users are responsible for reviewing and complying with the applicable terms before using or redistributing third-party materials or materials derived from them.
