# Quantization versus deletion, at equal bytes

Two ways to spend a byte budget on a MoE's experts: keep all of them and
store them at fewer bits, or delete some and store the survivors at more
bits. POE can build either, and until now they had never been run head to
head at the same file size.

The short version: **in the calibration domain they are within noise of each
other; outside it, deletion is catastrophic and quantization is not.** A
REAP recipe validated only on the domain it was calibrated for will look
free, and is not.

## Setup

One source, one imatrix, one base ftype, five checkpoints.

- Source: `Qwen3.6-35B-A3B-Q8_0.gguf` (34.4 GiB), the same source as the
  mixed-precision arms in [mixed-precision.md](mixed-precision.md).
- Imatrix: POE's own, 8192 coding-calibration tokens
  (`poe profile --metric imatrix`), passed to `llama-quantize --imatrix`.
- Every arm: base ftype `q4_k_m` for non-expert tensors; the routed expert
  slabs forced to one type with `poe quantplan --types <T> --tensor-types`.
- Pruned arms: `poe plan --method reap` + `poe apply`, keep count chosen so
  the output lands just under the control's size.
- Metric: KL divergence against the Q8_0 source, 100 chunks × 512 tokens,
  on held-out C/C++ (`code-novel.txt`, the calibration domain) and on
  wikitext-2 (outside it).

| Arm | experts | slab type | file size | vs control |
|---|---|---|---|---|
| **uniform Q3_K** *(control)* | 256 | Q3_K | 15.504 GB | — |
| Q4_K + REAP 23.8% | 195 | Q4_K | 15.445 GB | −0.38% |
| Q5_K + REAP 37.5% | 160 | Q5_K | 15.473 GB | −0.20% |
| Q6_K + REAP 47.7% | 134 | Q6_K | 15.455 GB | −0.32% |
| *(ceiling)* Q4_K, no pruning | 256 | Q4_K | 19.783 GB | +27.6% |

Every pruned arm is *smaller* than the control, so none of them can win on
having more bytes to spend.

## Result

| Arm | KLD code | KLD general | top-1 code | top-1 general |
|---|---|---|---|---|
| uniform Q3_K | 0.034306 ±0.00057 | **0.067458** ±0.00094 | 94.68% | **89.09%** |
| Q4_K + REAP 23.8% | **0.032663** ±0.00083 | 0.483002 ±0.00574 | **95.17%** | 73.82% |
| Q5_K + REAP 37.5% | 0.035525 ±0.00102 | 0.679604 ±0.00701 | 95.05% | 69.00% |
| Q6_K + REAP 47.7% | 0.042886 ±0.00116 | 0.880364 ±0.00824 | 94.71% | 65.11% |
| *(ceiling)* Q4_K full | 0.024926 ±0.00049 | 0.045970 ±0.00072 | 95.60% | 90.91% |

**In the calibration domain, deletion is competitive.** Pruning a quarter of
the experts and spending the savings on Q4_K lands at 0.0327 against the
control's 0.0343 — about 1.6σ apart, in pruning's favour — with higher top-1
agreement (95.17% vs 94.68%). At equal bytes, on the domain you calibrated
for, the two strategies are a coin flip.

**Outside it, they are not remotely comparable.** The same checkpoint scores
0.483 on wikitext against the control's 0.067: **7.2× the divergence**, with
top-1 agreement falling from 89.1% to 73.8%. Quantization damages a model;
deletion replaces it with a different one that happens to still be good at
the thing it was calibrated on.

**Pruning deeper never pays, in either domain.** Each step down the table
buys the survivors a wider type, and each step is worse than the one before
it — 0.0327 → 0.0355 → 0.0429 on code, 0.48 → 0.68 → 0.88 on general. There
is no crossover where deleting more and quantizing softer wins. If deletion
is used at all, it belongs at the shallow end.

**The ceiling is worth naming.** Keeping every expert at Q4_K costs 27.6%
more bytes and is the best arm on both axes — 27% less damage than the
control on code, 32% less on general. The right question at a fixed budget
is which type to use, not how many experts to drop.

## What this means for the recipe

1. At a fixed byte budget, spend it on precision, not on expert count.
2. If a deployment is genuinely single-domain — the coding-only local model
   POE exists to make — REAP at ~25% is a legitimate arm, and this is the
   measurement that says so honestly. It is not free; its cost is invisible
   from inside the domain.
3. Never validate a pruned checkpoint only on the calibration domain. The
   in-domain number is the one that cannot see the damage.

## Caveats

- The prune plans were built from a REAP profile captured on a *different*
  quantization of the same model (`--force`), relying on the banked result
  that expert ranking is quantization-invariant (REAP Spearman 0.994 between
  the Q6 and Q8 profiles of this checkpoint). The control needs no profile
  at all, so this asymmetry can only work against the pruned arms.
- KLD is measured against the Q8_0 source, not against BF16. Every arm
  inherits whatever the Q8_0 step cost, equally.
- 100 chunks × 512 tokens per arm and domain. The error bars above are
  llama-perplexity's own.
