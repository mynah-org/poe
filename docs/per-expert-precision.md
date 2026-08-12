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

Both get an inverted control.
