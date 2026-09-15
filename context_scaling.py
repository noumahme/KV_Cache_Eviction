"""
Context-scaling diagnostic.

Question: does your evaluation text actually contain long-range dependencies?

Method: fix a 512-token continuation at one absolute position. Vary ONLY how many
preceding tokens the model sees. Full cache throughout -- no compression anywhere.
This measures a property of the CORPUS, not of any compression method.

Reading the output:

  PPL keeps dropping out to 16k
      -> real long-range structure. The corpus is valid, and any method that
         discards distant context should lose. Trust your comparisons.

  PPL plateaus after 1-2k
      -> no usable long-range dependency. Distant context is noise. Recency
         policies win trivially and the corpus CANNOT evaluate KV compression.

  PPL gets WORSE with more context
      -> distant context is actively misleading, which is what concatenating
         unrelated articles produces. Same conclusion, more emphatic.

Usage:
    python context_scaling.py                     # wikitext, the suspect corpus
    python context_scaling.py --text-file doc.txt # a single coherent document
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def ppl_at_context(model, ids, anchor, ctx_len, eval_len, device):
    """PPL of ids[anchor : anchor+eval_len], conditioned on the ctx_len tokens before it."""
    start = anchor - ctx_len
    assert start >= 0, f"need {ctx_len} tokens before anchor {anchor}"

    ctx = ids[:, start:anchor].to(device)
    tgt = ids[:, anchor:anchor + eval_len].to(device)

    try:
        out = model(ctx, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(ctx, use_cache=True)

    logits = model(tgt, past_key_values=out.past_key_values, use_cache=True).logits
    shifted = torch.cat([out.logits[:, -1:, :], logits[:, :-1, :]], dim=1)
    nll = F.cross_entropy(shifted.float().reshape(-1, shifted.shape[-1]), tgt.reshape(-1))

    del out, logits, shifted
    torch.cuda.empty_cache()
    return nll.exp().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--text-file", default=None)
    p.add_argument("--eval-len", type=int, default=512)
    p.add_argument("--max-ctx", type=int, default=16384)
    p.add_argument("--n-docs", type=int, default=3,
                   help="repeat at several anchors; one sample is not a measurement")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    ).eval()

    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read()
        label = args.text_file
    else:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if len(t) > 200)
        label = "wikitext-103 (concatenated articles)"

    ids = tok(text, return_tensors="pt").input_ids
    print(f"corpus : {label}")
    print(f"tokens : {ids.shape[1]}\n")

    ctx_lens = [c for c in (8192, 16384) if c <= args.max_ctx] #512, 1024, 2048, 4096, 
    span = args.max_ctx + args.eval_len

    anchors = []
    for i in range(args.n_docs):
        a = args.max_ctx + i * span
        if a + args.eval_len <= ids.shape[1]:
            anchors.append(a)
    assert anchors, "text too short for even one anchor"

    print(f"{'ctx':>7} | " + " | ".join(f"{'anchor '+str(i):>12}" for i in range(len(anchors))) + " |     mean")
    print("-" * (10 + 15 * len(anchors) + 10))

    results = {}
    for ctx_len in ctx_lens:
        row = [ppl_at_context(model, ids, a, ctx_len, args.eval_len, model.device)
               for a in anchors]
        results[ctx_len] = sum(row) / len(row)
        print(f"{ctx_len:>7} | " + " | ".join(f"{v:>12.4f}" for v in row) +
              f" | {results[ctx_len]:>8.4f}")

    # ---- verdict -----------------------------------------------------------
    short, long = results[ctx_lens[0]], results[ctx_lens[-1]]
    gain = short - long
    late_gain = results[ctx_lens[len(ctx_lens) // 2]] - long

    print(f"\n{ctx_lens[0]} -> {ctx_lens[-1]} tokens : {gain:+.4f} PPL")
    print(f"second half of that range  : {late_gain:+.4f} PPL")

    if late_gain < 0.05:
        print("\nVERDICT: no usable long-range dependency past the midpoint.")
        print("Distant context is noise here. Recency policies win by default and")
        print("this corpus CANNOT evaluate KV cache compression. Switch to coherent")
        print("long documents -- PG-19 books, full arXiv papers, or a single long")
        print("article -- before running any more comparisons.")
    else:
        print("\nVERDICT: real long-range structure present. Corpus is usable.")


if __name__ == "__main__":
    main()