#!/usr/bin/env python3
"""
Build every table and matrix reported in the paper from the harness output.

Inputs (per model): raw_L{L}.csv, summary_paired.csv, anchors_used.json
Outputs: one .tsv per table (paste into Word) and one .tex per table (booktabs).

Nothing here recomputes statistics that summary_paired.csv already carries; the
only things computed from raw are descriptive (medians, ranges, retirement
fractions) that the summary file does not store. Where both exist, a
consistency check compares them and refuses to emit on disagreement.
"""

import argparse, csv, glob, json, math, os, sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "tables")

# Where each model's harness output lives. Point these at the --out-dir you
# actually used; the defaults are the run directories from the paper.
# Override on the command line:
#   python3 build_tables.py --qwen3 results_qwen3_512 --llama results_llama31_512
RESULT_DIRS = {
    "qwen3": "results_qwen3_512",
    "llama": "results_llama31_512",
}

# Directory names tried, in order, when the configured one is absent. Saves
# retyping the flags when the folders sit next to this script under their
# usual names.
FALLBACK_DIRS = {
    "qwen3": ["results_qwen3_512", "results_qwen3", "data/qwen3"],
    "llama": ["results_llama31_512", "results_llama", "results_llama31",
              "data/llama"],
}

MODELS = [("qwen3", "Qwen3-8B"), ("llama", "Llama-3.1-8B")]
DATA_DIR = {}          # model -> resolved results directory, filled in by main()
LENGTHS = [4096, 8192]
BUDGETS = [0.05, 0.10, 0.20, 0.50]

# display order and labels
POLICY_LABEL = [
    ("snapkv", "SnapKV (attention scoring)"),
    ("pyramidkv", "PyramidKV"),
    ("tova", "TOVA"),
    ("normkv", "L2-norm"),
    ("sink_window", "sink+window"),
    ("recency", "recent-only"),
    ("random", "random"),
]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def resolve_dir(model, explicit=False):
    """Find this model's results directory, and say clearly which one was used.

    Silently falling back to a stale directory is the failure that matters
    here: the tables would build, look right, and describe the wrong run. So a
    path given on the command line is used as given or not at all -- the
    fallback list applies only when no path was specified.
    """
    if explicit:
        cands = [RESULT_DIRS[model]]
    else:
        cands = [RESULT_DIRS[model]] + [d for d in FALLBACK_DIRS[model]
                                        if d != RESULT_DIRS[model]]
    tried = []
    for c in cands:
        p = c if os.path.isabs(c) else os.path.join(ROOT, c)
        tried.append(p)
        if os.path.isdir(p) and os.path.exists(
                os.path.join(p, "summary_paired.csv")):
            return p
    sys.exit(
        f"\nNo results directory found for {model}. Looked for a folder "
        f"containing summary_paired.csv at:\n"
        + "\n".join(f"  {t}" for t in tried)
        + f"\n\nPass the right path:  python3 build_tables.py --{model} "
          f"<path/to/your/out-dir>\n")


def load_raw(model):
    rows = []
    d = DATA_DIR[model]
    for L in LENGTHS:
        p = os.path.join(d, f"raw_L{L}.csv")
        if not os.path.exists(p):
            sys.exit(f"{p} is missing. The run for L={L} did not complete, or "
                     f"--context-lengths differed from {LENGTHS}.")
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "L": int(r["L"]),
                    "method": r["method"],
                    "budget": float(r["budget"]),
                    "anchor_idx": int(r["anchor_idx"]),
                    "anchor_offset": int(r["anchor_offset"]),
                    "ppl": float(r["ppl"]),
                    "kept_mean": float(r["kept_mean"]),
                    "kept_min": int(r["kept_min"]),
                    "kept_max": int(r["kept_max"]),
                })
    return rows


def load_summary(model):
    p = os.path.join(DATA_DIR[model], "summary_paired.csv")
    out = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["L"]), r["method"], float(r["budget"]), r["comparator"])
            out[key] = {
                "n": int(r["n"]),
                "ppl_method": float(r["ppl_method"]),
                "ppl_comparator": float(r["ppl_comparator"]),
                "delta_mean": float(r["delta_mean"]),
                "delta_sd": float(r["delta_sd"]),
                "ci_lo": float(r["ci_lo"]),
                "ci_hi": float(r["ci_hi"]),
                "t": float(r["t"]),
                "p": float(r["p"]),
                "d_z": float(r["d_z"]),
                "primary": r["primary"] == "True",
                "p_holm": float(r["p_holm"]) if r["p_holm"] else None,
                "sig": r["sig"],
            }
    return out


def index_raw(rows):
    """(L, method, budget) -> {anchor_idx: ppl}"""
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["L"], r["method"], r["budget"])][r["anchor_idx"]] = r["ppl"]
    return idx


# ----------------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------------

def check_pairing(idx, model):
    """Every (L, method, budget) cell must cover the same anchor set."""
    anchors = {}
    for (L, m, b), d in idx.items():
        anchors.setdefault(L, set(d.keys()))
        if set(d.keys()) != anchors[L]:
            sys.exit(f"[{model}] L={L} {m}@{b} covers a different anchor set "
                     f"than the rest of L={L}. Refusing to build tables from "
                     f"quietly-broken pairing.")
    for L, a in anchors.items():
        if len(a) != 20:
            sys.exit(f"[{model}] L={L} has {len(a)} anchors, expected 20.")
    return anchors


def check_anchors_file(rows, model):
    """The anchor offsets in raw_L*.csv must match anchors_used.json.

    This is the guard that catches the wrong results folder. Two runs of the
    same model produce files with identical names and identical column
    structure; only the offsets differ. Without this check, building tables
    from a stale directory succeeds quietly.
    """
    p = os.path.join(DATA_DIR[model], "anchors_used.json")
    if not os.path.exists(p):
        print(f"  [{model}] warning: no anchors_used.json in "
              f"{DATA_DIR[model]}; skipping the anchor-set check.")
        return None
    with open(p) as f:
        declared = {int(k): list(map(int, v)) for k, v in json.load(f).items()}
    seen = defaultdict(dict)
    for r in rows:
        seen[r["L"]][r["anchor_idx"]] = r["anchor_offset"]
    for L, offsets in declared.items():
        got = [seen[L][i] for i in sorted(seen[L])]
        if got != offsets:
            sys.exit(
                f"\n[{model}] L={L}: the anchor offsets in raw_L{L}.csv do not "
                f"match anchors_used.json in the same folder.\n"
                f"  json: {offsets[:4]} ...\n"
                f"  csv : {got[:4]} ...\n"
                f"This folder mixes output from two different runs. Rebuild "
                f"it rather than publishing tables from it.\n")
    return declared


def check_budgets(rows, model):
    """kept_mean must equal the nominal budget for every policy, so that
    PyramidKV really is compared at the same average memory footprint."""
    for r in rows:
        if r["method"] == "full":
            continue
        want = round(r["budget"] * r["L"])
        if abs(r["kept_mean"] - want) > 1.0:
            sys.exit(f"[{model}] {r['method']}@{r['budget']} L={r['L']} kept "
                     f"{r['kept_mean']} slots, nominal {want}.")


def check_summary_vs_raw(idx, summ, model, tol=1e-6):
    """Recompute a paired delta from raw and compare to the stored one."""
    checked = 0
    for (L, m, b, comp), s in summ.items():
        a = idx.get((L, m, b))
        c = idx.get((L, comp, 1.0 if comp == "full" else b))
        if not a or not c:
            continue
        keys = sorted(set(a) & set(c))
        d = np.array([a[k] - c[k] for k in keys])
        if abs(d.mean() - s["delta_mean"]) > tol * max(1.0, abs(s["delta_mean"])) * 1e3:
            sys.exit(f"[{model}] stored delta for {m} vs {comp} @{b} L={L} "
                     f"({s['delta_mean']:.6f}) disagrees with raw "
                     f"({d.mean():.6f}).")
        checked += 1
    return checked


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:,.{nd}f}"


def fmt_signed(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+,.{nd}f}"


def fmt_p(p):
    if p is None or math.isnan(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def stars(s):
    return {"***": "***", "**": "**", "*": "*", "ns": "ns", "": ""}.get(s, s)


def tex_escape(s):
    """Escape a cell for LaTeX. '%' matters most here: an unescaped one
    comments out the rest of the row, which silently eats the line ending and
    corrupts the table rather than failing loudly."""
    out = []
    for ch in s:
        if ch in "%&#$":
            out.append("\\" + ch)
        elif ch == "_":
            out.append("\\_")
        else:
            out.append(ch)
    return "".join(out)


class Table:
    def __init__(self, name, caption, header, align=None):
        self.name = name
        self.caption = caption
        self.header = header
        self.rows = []
        self.rules = []          # row indices after which to draw a midrule
        self.align = align or ("l" + "r" * (len(header) - 1))

    def add(self, *cells):
        self.rows.append([str(c) for c in cells])

    def rule(self):
        self.rules.append(len(self.rows))

    def write(self):
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, self.name + ".tsv"), "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(self.header)
            w.writerows(self.rows)
        with open(os.path.join(OUT, self.name + ".tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\small\n")
            f.write(f"\\caption{{{self.caption}}}\n")
            f.write(f"\\label{{tab:{self.name}}}\n")
            f.write(f"\\begin{{tabular}}{{{self.align}}}\n\\toprule\n")
            f.write(" & ".join(tex_escape(h) for h in self.header)
                    + " \\\\\n\\midrule\n")
            for i, r in enumerate(self.rows):
                if i in self.rules:
                    f.write("\\midrule\n")
                f.write(" & ".join(tex_escape(c) for c in r) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        return self


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def table_deltas(model, label, idx, summ, n):
    """Table 1 / 2: perplexity delta vs full cache, every policy and budget."""
    t = Table(
        f"table{n}_deltas_{model}",
        f"Perplexity change relative to the full cache on {label}. "
        f"Mean paired difference over 20 anchors with a 95\\% CI "
        f"($t_{{crit}}=2.093$, df $=19$). Lower is better; the full-cache "
        f"reference is 0 by construction. Medians are given for the random "
        f"control because its distribution is heavy-tailed at middle budgets.",
        ["Policy", "Budget", "L=4096 delta PPL", "L=4096 95% CI",
         "L=8192 delta PPL", "L=8192 95% CI"],
    )
    for pol, plabel in POLICY_LABEL:
        for j, b in enumerate(BUDGETS):
            cells = [plabel if j == 0 else "", f"{b:.2f}"]
            for L in LENGTHS:
                s = summ.get((L, pol, b, "full"))
                if s is None:
                    cells += ["--", "--"]
                    continue
                if pol == "random":
                    a = idx[(L, pol, b)]
                    c = idx[(L, "full", 1.0)]
                    med = float(np.median([a[k] - c[k] for k in sorted(a)]))
                    cells.append(f"{fmt_signed(s['delta_mean'], 2)} "
                                 f"(med {fmt_signed(med, 2)})")
                else:
                    cells.append(fmt_signed(s["delta_mean"], 3))
                cells.append(f"[{fmt_signed(s['ci_lo'], 3)}, "
                             f"{fmt_signed(s['ci_hi'], 3)}]")
            t.add(*cells)
        t.rule()
    return t.write()


def table_primary(summ_by_model, n):
    """Table 3: the pre-specified confirmatory endpoint."""
    t = Table(
        f"table{n}_primary_endpoint",
        "Pre-specified primary endpoint: sink+window minus attention scoring "
        "at a 5\\% budget. Positive values favour attention scoring. Paired "
        "$t$ test, df $=19$; Holm correction applied across the two context "
        "lengths within each model ($m=2$), never pooled across models.",
        ["Model", "Context", "mean d", "95% CI", "t", "d_z", "p", "p (Holm)",
         "Result"],
    )
    for model, label in MODELS:
        summ = summ_by_model[model]
        for L in LENGTHS:
            s = summ[(L, "sink_window", 0.05, "snapkv")]
            passed = s["p_holm"] is not None and s["p_holm"] < 0.05
            t.add(label, f"{L:,}", fmt_signed(s["delta_mean"]),
                  f"[{fmt_signed(s['ci_lo'])}, {fmt_signed(s['ci_hi'])}]",
                  fmt(s["t"], 2), fmt(s["d_z"], 2), fmt_p(s["p"]),
                  fmt_p(s["p_holm"]), "pass" if passed else "fail")
        t.rule()
    return t.write()


def table_budget_dependence(summ_by_model, n):
    """Table 4: exploratory sweep of the same contrast across all budgets."""
    t = Table(
        f"table{n}_budget_dependence",
        "Sink+window minus attention scoring at every budget. Positive values "
        "favour attention scoring. These cells are exploratory; only the 5\\% "
        "row is the pre-specified endpoint, and no per-cell significance is "
        "claimed for the others.",
        ["Budget", "Qwen3 L=4096", "Qwen3 L=8192",
         "Llama L=4096", "Llama L=8192"],
    )
    for b in BUDGETS:
        cells = [f"{b:.2f}"]
        for model, _ in MODELS:
            for L in LENGTHS:
                s = summ_by_model[model][(L, "sink_window", b, "snapkv")]
                cells.append(f"{fmt_signed(s['delta_mean'])} "
                             f"(d_z {fmt(s['d_z'], 2)})")
        t.add(*cells)
    return t.write()


def table_vs_snapkv(summ_by_model, n):
    """Table 5: each additional published policy against SnapKV selection."""
    t = Table(
        f"table{n}_policies_vs_snapkv",
        "Each additional published policy minus attention scoring, at matched "
        "budgets. Positive values mean the policy is worse than SnapKV-style "
        "selection. Paired $t$ test, df $=19$.",
        ["Policy", "Model", "Context", "Budget", "delta PPL", "95% CI",
         "d_z", "p"],
    )
    for pol, plabel in [("pyramidkv", "PyramidKV"), ("tova", "TOVA"),
                        ("normkv", "L2-norm")]:
        first = True
        for model, label in MODELS:
            for L in LENGTHS:
                for b in [0.05, 0.50]:
                    s = summ_by_model[model][(L, pol, b, "snapkv")]
                    t.add(plabel if first else "", label if b == 0.05 and L == LENGTHS[0] else "",
                          f"{L:,}" if b == 0.05 else "", f"{b:.2f}",
                          fmt_signed(s["delta_mean"]),
                          f"[{fmt_signed(s['ci_lo'])}, {fmt_signed(s['ci_hi'])}]",
                          fmt(s["d_z"], 2), fmt_p(s["p"]))
                    first = False
        t.rule()
    return t.write()


def table_reference(idx_by_model, n):
    """Matrix A: the full-cache reference and the anchor spread that motivates
    the paired design."""
    t = Table(
        f"table{n}_full_cache_reference",
        "Full-cache reference. Anchor-to-anchor spread in perplexity is an "
        "order of magnitude larger than the policy differences under test, "
        "which is what the paired design removes.",
        ["Model", "Context", "Full-cache PPL (mean +/- SD)", "Anchor range",
         "Smallest paired SD observed"],
    )
    for model, label in MODELS:
        idx = idx_by_model[model]
        for L in LENGTHS:
            f_ = idx[(L, "full", 1.0)]
            v = np.array([f_[k] for k in sorted(f_)])
            # smallest paired SD across all policy-vs-full contrasts at this L
            sds = []
            for pol, _ in POLICY_LABEL:
                for b in BUDGETS:
                    a = idx.get((L, pol, b))
                    if not a:
                        continue
                    d = np.array([a[k] - f_[k] for k in sorted(a)])
                    sds.append(d.std(ddof=1))
            t.add(label, f"{L:,}",
                  f"{fmt(v.mean())} +/- {fmt(v.std(ddof=1))}",
                  f"{fmt(v.min(), 1)} - {fmt(v.max(), 1)}",
                  fmt(min(sds)))
        t.rule()
    return t.write()


def table_sinks(summ_by_model, n):
    """Matrix B: what four sink tokens are worth."""
    t = Table(
        f"table{n}_sink_tokens",
        "The value of four cached tokens. Sink+window and recent-only differ "
        "by exactly the four initial tokens; the column is the perplexity "
        "those four tokens are worth at a 5\\% budget.",
        ["Model", "Context", "recent-only delta", "sink+window delta",
         "Worth of 4 sink tokens", "Ratio"],
    )
    for model, label in MODELS:
        summ = summ_by_model[model]
        for L in LENGTHS:
            r = summ[(L, "recency", 0.05, "full")]["delta_mean"]
            s = summ[(L, "sink_window", 0.05, "full")]["delta_mean"]
            t.add(label, f"{L:,}", fmt_signed(r, 1), fmt_signed(s, 2),
                  fmt(r - s, 1), f"{r / s:,.0f}x")
        t.rule()
    return t.write()


def table_controls(summ_by_model, n):
    """Matrix C: structured selection against the unstructured controls."""
    t = Table(
        f"table{n}_control_ratios",
        "Structured selection against the unstructured controls at a 5\\% "
        "budget, as a ratio of excess perplexity over the full cache. "
        "Reported descriptively; these contrasts were not pre-specified for "
        "testing.",
        ["Model", "Context", "SnapKV delta", "random delta", "recent delta",
         "random / SnapKV", "recent / SnapKV"],
    )
    for model, label in MODELS:
        summ = summ_by_model[model]
        for L in LENGTHS:
            sk = summ[(L, "snapkv", 0.05, "full")]["delta_mean"]
            rd = summ[(L, "random", 0.05, "full")]["delta_mean"]
            rc = summ[(L, "recency", 0.05, "full")]["delta_mean"]
            t.add(label, f"{L:,}", fmt_signed(sk, 2), fmt_signed(rd, 1),
                  fmt_signed(rc, 1), f"{rd / sk:,.0f}x", f"{rc / sk:,.0f}x")
        t.rule()
    return t.write()


def table_convergence(summ_by_model, n):
    """Matrix D: does excess perplexity converge back toward the full cache as
    the budget grows? Norm eviction is the policy that does not.

    Reported per context length rather than averaged: the two lengths are
    independent replications everywhere else in the paper, and averaging a
    ratio across them would not match the figures quoted in the text.
    """
    t = Table(
        f"table{n}_budget_convergence",
        "Fraction of excess perplexity retired when the budget is relaxed "
        "from 5\\% to 50\\%. A policy that is merely losing information should "
        "recover most of it as the budget grows. Among the structured "
        "policies, L2-norm eviction recovers the least on both models. A "
        "value above 100\\% means the policy has caught up with the full "
        "cache and its 50\\% delta is slightly negative.",
        ["Policy", "Qwen3 L=4096", "Qwen3 L=8192",
         "Llama L=4096", "Llama L=8192"],
    )
    for pol, plabel in POLICY_LABEL:
        cells = [plabel]
        for model, _ in MODELS:
            summ = summ_by_model[model]
            for L in LENGTHS:
                lo = summ[(L, pol, 0.05, "full")]["delta_mean"]
                hi = summ[(L, pol, 0.50, "full")]["delta_mean"]
                cells.append(f"{100 * (1.0 - hi / lo):.1f}%")
        t.add(*cells)
    return t.write()


# Per-layer diagnostics come from a separate `--diagnose` run whose output
# (knorm_diagnostic.csv) is not part of the main sweep. The values below are
# transcribed from those runs; they are not derivable from raw_L*.csv, so they
# are entered here rather than computed, and the source run is named in the
# caption.
KNORM_DIAG = {
    # model: (rho, n_layers, n_wrong_sign, rho_range, rho_squared, median_cv)
    "qwen3": (-0.081, 36, 11, "-0.29 to +0.14", 0.021, 0.054),
    "llama": (-0.342, 32, 0, "-0.29 to -0.50", 0.119, 0.096),
}


def table_knorm(summ_by_model, n):
    """Matrix E: the premise L2-norm eviction depends on, measured."""
    t = Table(
        f"table{n}_knorm_diagnostic",
        "The assumption behind L2-norm eviction, measured per layer. Spearman "
        "$\\rho$ between $\\|k\\|$ and the attention a token actually receives, "
        "averaged over heads and anchors, from the \\texttt{--diagnose} run. "
        "The method needs this correlation to be strongly negative. "
        "CV$(\\|k\\|)$ is the median per-layer coefficient of variation of the "
        "key norm.",
        ["Model", "QK-norm", "rho", "Layers with wrong sign", "Per-layer range",
         "rho^2", "median CV(||k||)", "L2-norm delta @0.05, L=4096"],
    )
    for model, label in MODELS:
        rho, nl, wrong, rng, r2, cv = KNORM_DIAG[model]
        d = summ_by_model[model][(4096, "normkv", 0.05, "full")]["delta_mean"]
        t.add(label, "yes" if model == "qwen3" else "no", fmt_signed(rho),
              f"{wrong} of {nl}", rng, f"{100 * r2:.1f}%", fmt(cv),
              fmt_signed(d, 2))
    return t.write()


def table_absolute(summ_by_model, n):
    """Matrix E: absolute perplexity for every policy at every budget, for
    readers who want the raw numbers rather than the deltas."""
    t = Table(
        f"table{n}_absolute_ppl",
        "Absolute perplexity, mean over 20 anchors. Values are not comparable "
        "across models: the two tokenizers segment the corpus differently, so "
        "anchors were drawn independently per model.",
        ["Policy", "Budget", "Qwen3 L=4096", "Qwen3 L=8192",
         "Llama L=4096", "Llama L=8192"],
    )
    t.add("full (reference)", "1.00",
          *[fmt(summ_by_model[m][(L, "snapkv", 0.05, "full")]["ppl_comparator"])
            for m, _ in MODELS for L in LENGTHS])
    t.rule()
    for pol, plabel in POLICY_LABEL:
        for j, b in enumerate(BUDGETS):
            cells = [plabel if j == 0 else "", f"{b:.2f}"]
            for model, _ in MODELS:
                for L in LENGTHS:
                    s = summ_by_model[model][(L, pol, b, "full")]
                    cells.append(fmt(s["ppl_method"], 2))
            t.add(*cells)
        t.rule()
    return t.write()


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build the paper's tables from the harness output.")
    ap.add_argument("--qwen3", help="Qwen3-8B results directory (the --out-dir "
                                    "you passed to kv_eviction_extended.py)")
    ap.add_argument("--llama", help="Llama-3.1-8B results directory")
    ap.add_argument("--out", default=None, help="where to write the tables")
    args = ap.parse_args()

    explicit = {"qwen3": bool(args.qwen3), "llama": bool(args.llama)}
    if args.qwen3:
        RESULT_DIRS["qwen3"] = args.qwen3
    if args.llama:
        RESULT_DIRS["llama"] = args.llama

    global DATA_DIR, OUT
    if args.out:
        OUT = args.out
    DATA_DIR = {m: resolve_dir(m, explicit[m]) for m, _ in MODELS}
    for m, label in MODELS:
        print(f"[{m}] reading {DATA_DIR[m]}")

    raw, idx_by_model, summ_by_model = {}, {}, {}
    for model, label in MODELS:
        raw[model] = load_raw(model)
        idx_by_model[model] = index_raw(raw[model])
        summ_by_model[model] = load_summary(model)
        check_pairing(idx_by_model[model], model)
        check_anchors_file(raw[model], model)
        check_budgets(raw[model], model)
        nch = check_summary_vs_raw(idx_by_model[model], summ_by_model[model], model)
        print(f"[{model}] {len(raw[model])} raw rows, "
              f"{len(summ_by_model[model])} summary rows, "
              f"{nch} paired deltas re-derived from raw and matched.")

    built = [
        table_deltas("qwen3", "Qwen3-8B", idx_by_model["qwen3"], summ_by_model["qwen3"], 1),
        table_deltas("llama", "Llama-3.1-8B", idx_by_model["llama"], summ_by_model["llama"], 2),
        table_primary(summ_by_model, 3),
        table_budget_dependence(summ_by_model, 4),
        table_vs_snapkv(summ_by_model, 5),
        table_reference(idx_by_model, 6),
        table_sinks(summ_by_model, 7),
        table_controls(summ_by_model, 8),
        table_convergence(summ_by_model, 9),
        table_knorm(summ_by_model, 10),
        table_absolute(summ_by_model, 11),
    ]
    print(f"\nWrote {len(built)} tables to {OUT}/ (.tsv and .tex each):")
    for t in built:
        print(f"  {t.name}")


if __name__ == "__main__":
    main()