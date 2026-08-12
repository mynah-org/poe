# Mixed precision per expert slab: what a byte budget should buy

`poe quantplan` spends a byte budget across a MoE's routed expert slabs by
giving each slab its own quantization type. This documents what that
mechanism is, what it can and cannot express, and the first measurement of
whether spending the budget *unevenly* is worth anything.

The short version: on Qwen3.6-35B-A3B, **uniform allocation beats every
non-uniform one we tried**, and the saliency signal that drove the uneven
allocation turned out to carry no usable direction. That is a negative
result, and the control is what makes it readable.

## What the format allows

A GGUF tensor carries exactly one type, and routed experts are packed one
tensor per projection (`blk.N.ffn_{gate,up,down}_exps.weight`, with the
expert dimension inside). So expert 5 at Q2_K and expert 6 at Q4_K *within
one layer* is not expressible — per-expert precision needs the slab split
into several tensors and a runtime that can consume them.

What is expressible today, with no runtime change, is one type per slab.
`llama-quantize` accepts a `--tensor-type-file` of `name=type` lines,
compiled as `std::regex` and matched against the full tensor name, so a plan
is a text file it reads directly.

## The allocator

Every slab starts on the cheapest rung of the ladder; the budget is then
spent on the upgrades with the best ratio of removed error to added bytes,
weighted by slab size and by the layer's saliency. The error scale is
measured, not modelled — it is ingot's own round-trip relative L2 per type
on a bell-shaped block, which is what a weight matrix looks like:

| Type | bits/weight | round-trip rel L2 |
|---|---|---|
| Q2_K | 2.625 | 0.2682 |
| Q3_K | 3.4375 | 0.1615 |
| Q4_K | 4.5 | 0.0724 |
| Q5_K | 5.5 | 0.0366 |
| Q6_K | 6.5625 | 0.0179 |

The ladder halves cleanly per bit, which is the sanity check that these
measure the format rather than an encoder bug. They rank types correctly;
they are not a prediction of KLD on a real model.

## The experiment

Three checkpoints from the **same** Q8_0 source, the **same** imatrix (POE's
own, 8192 calibration tokens on C/C++), the **same** base ftype for every
non-expert tensor. The only difference is the type map over the 120 routed
slabs (40 layers × 3 projections):

| Arm | Q2_K | Q3_K | Q4_K | Q5_K | slab bytes |
|---|---|---|---|---|---|
| uniform | | 120 | | | 13.841 GB |
| saliency | 62 | 22 | 24 | 12 | 13.837 GB |
| inverted *(control)* | 21 | 83 | 16 | | 13.839 GB |

Budgets match to within 0.03%, and the arm under test is the *smallest* of
the three — so any advantage it showed could not come from having more
bytes. KL divergence against the Q8_0 source, 100 chunks × 512, on
in-domain C and on wikitext-2.

## Result: uniform wins, and the signal has no direction

| Arm | KLD code | KLD general | top-1 code | top-1 general |
|---|---|---|---|---|
| **uniform Q3_K** | **0.034306** ±0.00057 | **0.067458** ±0.00094 | **94.68%** | **89.09%** |
| saliency | 0.045695 ±0.00083 | 0.093049 ±0.00137 | 94.10% | 87.07% |
| inverted *(control)* | 0.041562 ±0.00069 | 0.091705 ±0.00121 | 94.54% | 87.14% |

- **Uniform is decisively better**: 33% less damage than the saliency map on
  code, 38% less on general.
- **The control did not behave as a working signal should.** Inverting the
  ranking should hurt if the ranking is informative. Instead the inverted
  map is *better* than the saliency map on code (0.0416 vs 0.0457, about 4σ
  apart) and indistinguishable from it on general. The ranking carries no
  usable direction for this decision.

**Why the signal looked promising and was not.** Per-layer REAP totals on
this model rise monotonically with depth — normalized, 0.03–0.05 in the
early layers climbing to 1.00 at the last. That looks like a strong signal
and is largely the residual stream's norm growing with depth: REAP is
`gate × ‖expert output‖₂`, so a layer whose activations are simply larger
scores higher without being more sensitive to quantization. Ranking by it,
or against it, both lose to not ranking at all.

**Do not read this as a verdict on mixed precision itself.** What was tested
is *per-layer* allocation, the only granularity the format allows today. The
original argument for spending bits unevenly is about *per-expert*
differences — quantizing the experts a workload never routes to, while
keeping the hot ones wide — and that is untouched by this result, because
no arm here could express it.

**Practical consequence.** Until a per-expert mechanism exists, pick one
type for the expert slabs and spend the effort on choosing the right total
size. `poe quantplan --types q3_k` writes that plan, and it is the arm that
won.

## What the tool learned from losing

A negative result is only worth its cost if the next attempt cannot repeat
it by accident, so the failure mode is now something `poe quantplan`
measures about its own input rather than something a GPU run discovers
three hours later.

**The depth check.** Every plan records `depth_rho`, the Spearman
correlation of its per-layer scores against the layer index, and warns when
it is at or beyond ±0.9:

```
depth     Spearman +0.987 against the layer index   (see the warning below)
warning: layer scores are near-monotone in depth (Spearman +0.99): both
         allocations that lost to uniform had this shape — measure against
         uniform, and carry --invert as a control
```

A ranking that is essentially the layer index is not necessarily wrong, but
it is the exact shape that has already failed twice — once for routing
budgets, once for precision — and it should not be spent GPU time on
without a control.

**A second ranking, from the imatrix rather than the profile.** REAP
saliency measures an expert's contribution to the layer *output*, which is
the right question for deletion. Requantization damages the product of a
weight matrix with its *inputs*, and an imatrix is literally the quantity a
weighted encoder fits against, so `poe quantplan --imatrix` ranks layers by
it instead:

- `--imatrix-stat energy` (default) — mean squared input activation per
  element. It carries the activation scale, which on a residual network
  means it is expected to be a depth ramp; the depth check will say so.
- `--imatrix-stat concentration` — one minus the participation ratio of the
  per-column energy. Scale-free by construction, so it cannot be an
  activation-norm ramp in disguise: it measures how far a layer's input
  energy is concentrated in a few columns, which is what makes a block hard
  to fit at two or three bits.

Neither is a result. They are candidate rankings, and the only thing that
promotes one is beating uniform at matched bytes.

**Comparing two bit maps is one command.** `poe diff a.poequant b.poequant`
reports how many slabs got the same type, how the bytes moved, the type
histogram side by side, and the rank correlation of the two layer scores —
which answers "did swapping the ranking actually change anything" before a
quantize run does:

```
type map
  slabs with the same type   3 / 12
  wider in A                 6
  wider in B                 3
  A - B                      +288 B

layer scores   Spearman +0.000   (depth +1.000 vs +0.000)
```
