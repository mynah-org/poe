# Per-expert precision: measuring M9b before building it

M9's thesis is per expert: **quantize hard the experts a workload never
routes to, keep the hot ones wide.** No GGUF can store that — a tensor
carries exactly one type and routed experts are packed one tensor per
projection — so reaching it means splitting each slab in two, patching
`build_moe_ffn` to run two `mul_mat_id` passes with a router-id remap, and
merging the results on the hot path. That is weeks of work on the most
performance-sensitive code in the runtime.

[M9a](mixed-precision.md) tested the only granularity the format allows
today, one type per slab, and lost to uniform. That says nothing about the
per-expert question, because no arm there could express it.

This is how the question gets answered first, for the cost of two quantize
runs.

## The emulation

`poe requant` pushes every expert's weights through a quantizer and back,
but only the **cold** ones go through the hard type; the whole slab is then
stored at the **carrier** type:

```sh
# the arm under test: the coldest 145 of 256 experts crushed to Q2_K
poe requant model-q8.gguf -o aimed.gguf \
    --carrier q4_k --degrade-type q2_k --degrade-frac 0.56640625 \
    --profile coding.poeprofile

# the matched control: every expert through Q3_K, the same average bits
poe requant model-q8.gguf -o uniform.gguf \
    --carrier q4_k --degrade-type q3_k --degrade-frac 1

# the control that decides: the same 145 experts, chosen from the other end
poe requant model-q8.gguf -o inverted.gguf \
    --carrier q4_k --degrade-type q2_k --degrade-frac 0.56640625 \
    --profile coding.poeprofile --invert
```

The output is a plain Q4_K checkpoint that any runtime loads unpatched. Its
*size* is the carrier's; its *quality* is the mixed allocation's, because
the information the hard type destroyed is gone before the carrier ever
sees it. Every arm is byte-identical in size, so nothing has to be matched
by hand.

145/256 experts at Q2_K (2.625 bits) with the rest at Q4_K (4.5 bits)
averages **3.4380** bits per weight; uniform Q3_K is **3.4375**. The two
budgets differ by 0.01%, and the files differ by nothing at all.

## What the emulation does and does not model

Measured, not assumed (`tests/test_requant.c`), on the bell-shaped
distribution a weight matrix has:

| Path | rel L2 |
|---|---|
| Q4_K, once | 0.0704 |
| Q4_K, twice | +0.00000 |
| Q3_K stored directly | 0.1644 |
| Q3_K through the Q4_K carrier | 0.1765 (+7.3%) |
| Q2_K stored directly | 0.2649 |
| Q2_K through the Q4_K carrier | 0.2705 (+2.1%) |

Three things follow:

1. **The carrier pass is idempotent.** Once values sit on the carrier's
   grid, encoding them again costs exactly zero — so the ceiling arm
   (degrade type = carrier) is a plain Q4_K quant, not a degraded one.
2. **The emulation charges a small tax**, and it is *not equal across
   arms*: the uniform arm pays +7.3%, the aimed arm +2.1%. The comparison
   aimed-versus-uniform is therefore a practical one that slightly favours
   the aimed arm.
3. **Aimed versus inverted is the comparison that decides.** Same types,
   same number of degraded experts, same tax — the only difference is
   *which* experts were chosen. If aiming carries information, that pair
   separates; if it does not, nothing else in the table matters.

The other standing limit: ingot's encoders are unweighted least squares
where llama.cpp's are imatrix weighted. That applies equally to every arm,
and it means these numbers rank arms against each other rather than against
a released quant.

## Which experts are cold

Two signals, and they are not the same ordering:

- `--rank reap` (default) — REAP saliency, an expert's contribution to the
  layer output. The right question for *deletion*.
- `--rank counts` — how often the workload routes there at all. This is the
  M9 thesis stated literally, and it is the arm the thesis actually claims.

Both get an inverted control. On Qwen3.6 the two orderings agree on about
half of what they pick — `poe diff` on the two plans at the same prune
fraction reports **Jaccard 0.496**, with 1962 of 5800 expert slots chosen
by one and not the other — so the second arm is a genuinely different test
rather than a rerun of the first.

## Result: aiming works, and it works as specialization

Qwen3.6-35B-A3B, from the Q8_0 source, coding calibration (8192 tokens).
All six arms are **20,797,013,312 bytes** — the same file size, not a
matched one — and all emulate 3.4380 bits per weight against uniform Q3_K's
3.4375. KLD against the Q8_0 source, 100 chunks × 512, on held-out C/C++
and on wikitext-2.

| Arm | KLD code | top-1 code | KLD general | top-1 general |
|---|---|---|---|---|
| *(ceiling)* Q4_K, 4.5 bits | 0.008998 ±0.00025 | 97.31% | 0.017164 ±0.00028 | 94.31% |
| uniform, all Q3_K | 0.031487 ±0.00059 | 95.35% | **0.064341** ±0.00099 | **89.33%** |
| **aimed by REAP, 145 coldest at Q2_K** | **0.013328** ±0.00042 | **96.91%** | 0.092707 ±0.00148 | 87.49% |
| inverted *(control)* | 0.067434 ±0.00100 | 92.74% | 0.085017 ±0.00110 | 87.27% |
| aimed by frequency | 0.016521 ±0.00037 | 96.34% | 0.084363 ±0.00116 | 87.57% |
| inverted by frequency *(control)* | 0.063965 ±0.00104 | 93.05% | 0.093685 ±0.00137 | 87.43% |

**In the calibration domain, aiming wins by 2.4×.** Same bytes, same
average bits: 0.0133 against uniform's 0.0315. The gap is far larger than
the emulation's asymmetric tax (+7.3% for uniform, +2.1% for aimed), and
the clean comparison confirms it — the inverted control, which pays exactly
the same tax as the arm under test, is **5× worse** (0.0674). Unlike M9a's
per-layer ranking, this signal has a direction, and a strong one.

Put differently: at 3.438 bits the aimed allocation recovers most of the
way to a full Q4_K checkpoint (0.0133 against 0.0090) while spending 76% of
its expert bytes.

**Contribution beats frequency, and that is not what the thesis said.**
The thesis is usually stated as "the experts a workload never routes to",
which is selection frequency — but ranking by frequency scores 0.0165
against REAP saliency's 0.0133, about 6σ apart, even though both crush the
same number of experts and both beat uniform by a wide margin (1.9× and
2.4×). How much an expert *contributes when it is used* predicts
quantization sensitivity better than how often it is used at all. Both
rankings carry direction: each is ~4× better than its own inverted control.

**Outside the calibration domain it reverses.** On wikitext the aimed arm
(0.0927) is worse than uniform (0.0643) and **ties its own inverted
control** (0.0850, marginally better); the frequency arm behaves the same
way (0.0844 against its control's 0.0937). That is exactly what should
happen: the cold experts are cold *for code*. A general workload routes to
them, and finds them crushed. The ranking that has a 5× direction in the
calibration domain has none outside it.

**This is the same shape as [quantization versus deletion](quant-vs-prune.md).**
Both knobs buy a lot on the domain they were calibrated for and charge for
it everywhere else. The difference is the exchange rate: per-expert
precision at 3.438 bits gives 58% less in-domain damage than uniform for a
1.44× off-domain penalty, where deletion at matched bytes gave *no*
in-domain gain for a 7.2× penalty.

**Consequence for M9b.** The runtime work — splitting each slab into a hot
and a cold tensor, two `mul_mat_id` passes, a merge on the hot path — now
has a measured payoff to justify it, which it did not have this morning. It
should be built for the single-domain case POE exists to serve, and it must
never be shipped as a general-purpose quant.

## What the runtime side costs, measured before building it

A second `mul_mat_id` pass over every slot is the straightforward way to
consume a split checkpoint, and it doubles the expert work. On GB10, with
the expert budget doubled from 8 to 16 (the same amount of extra work, no
patch required to measure it):

| | tg128 | vs K=8 |
|---|---|---|
| K=8 | 63.63 ±0.60 t/s | — |
| K=16 | 55.30 ±0.15 t/s | **−13.1%** |

So the honest trade for the straightforward design is: same bytes, 2.4× less
in-domain damage, **−13% generation throughput**. That is a good deal for a
memory-bound target and a bad one for a throughput-bound target, and it is
the number to beat.

The variant that would cost nothing is a *fixed quota* — route the top few
within the hot set and the rest within the cold set, keeping the total at
eight expert evaluations. It buys the bytes for free but displaces some of
the router's real choices, and how often it would do that is a property of
the workload, not of the design. `poe split` reports it from the profile:
the share of routed slots that land in the hot set.
