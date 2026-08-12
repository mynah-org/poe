# Routing experiments: how much does the expert budget K cost?

Measured results on **Qwen3.6-35B-A3B** (40 MoE blocks × 256 experts, top-8,
plus always-active shared experts) for two questions POE has to answer before
it can advise anyone on the routed-expert budget:

1. what does lowering the global top-k actually cost, in the same currency as
   pruning and quantization (**R3**);
2. does spending that budget *unevenly across layers* beat spending it
   uniformly (**R7**).

The first has a clean answer. The second is a **negative result**, which is
why it is written down: it closes a line of work that looked promising.

## Method

- **Metric**: KL divergence against a single fixed reference, **Q8_0 full**,
  via `llama-perplexity --kl-divergence` (100 chunks × 512 tokens). Every
  variant is measured against that same reference, so pruning damage,
  quantization damage and routing damage are all expressed in one unit and
  can be compared directly.
- **Two corpora, deliberately contrasted**: an in-domain C/C++ set disjoint
  from the calibration text, and wikitext-2 test as the out-of-domain one.
  A coding-specialized checkpoint should cost far more on the second, and
  that gap *is* the specialization measurement.
- **Controls first.** Q8_0 against itself establishes the noise floor. For
  R7, an inverted schedule is run alongside the real one (see below).
- **Throughput**: `llama-bench`, 3 repetitions, prompt 2048 / generation 128,
  on a single NVIDIA GB10 (CUDA, unified memory).
- The base model's K is varied with stock llama.cpp's
  `--override-kv <arch>.expert_used_count=int:N`. Per-layer schedules need
  the optional patch in [`tools/patches/`](../tools/patches/README.md).

## R3 — the fixed-K sensitivity curve

UD-Q4_K_XL (20.82 GiB), K varied globally, KLD vs Q8_0 full:

| K | KLD code | top-1 code | KLD general | top-1 general |
|---|---|---|---|---|
| 8 (stock) | 0.00725 | 97.76% | 0.01277 | 95.18% |
| 7 | 0.01323 | 96.93% | 0.02532 | 93.30% |
| 6 | 0.02839 | 95.49% | 0.05403 | 89.88% |
| 5 | 0.05584 | 93.46% | 0.11105 | 85.50% |
| 4 | 0.10976 | 90.80% | 0.22034 | 79.92% |

Composed with pruning (Q4 + REAP-25%, 16.23 GiB): K=8 → 0.01875 code /
0.47444 general; K=6 → 0.03595 / 0.50323; K=5 → 0.06207 / 0.54648.

**There is no cliff — KLD roughly doubles per expert removed**, cleanly
(×1.82, ×2.15, ×1.97, ×1.97 down the ladder). This *contradicts* the earlier
128-expert story ("K=6 free, K=4 broken"): on a 256-expert fine-grained
router whose per-layer entropy spans only 7.25–7.77 of 8 bits, no expert is
free to drop and none is catastrophic. Routing intuitions from coarse-grained
MoE do not transfer.

**Reducing K is a poor trade if what you want is memory.** K=6 costs KLD
0.0284 and saves **zero bytes**; REAP-25% costs 0.0188 *and* frees 4.6 GiB.
At iso-quality (KLD ≈ 0.03) you can have 11.63 GiB (Q4+REAP-50%) or 20.82 GiB
(Q4 at K=6) — pruning wins that objective by nearly 2×. **K is a speed knob**,
and its quality price is higher than pruning's, so it has to be justified by
throughput alone.

**Damage composes sub-additively**: REAP-25% adds 0.0115 over Q4-K8, K=6 adds
0.0211; independent addition would predict 0.0399, measured 0.0360. The two
axes interfere slightly *less* than independently — good news for combining
them.

## R7 — per-layer static K does not pay (negative)

The hypothesis: a flat-router layer needs its full top-k while a peaked one
does not, so a schedule at the same *average* K should beat uniform K.

**The patch was validated before the experiment.** A schedule of all-8
reproduces stock exactly (KLD 0.00725) and all-6 reproduces
`--override-kv expert_used_count=6` exactly (0.02839). Then, at matched
average K = 6.0:

| Config (avg K = 6.0) | KLD code | KLD general |
|---|---|---|
| **uniform K=6** | **0.02839** | **0.05403** |
| entropy-ranked schedule | 0.03023 | 0.05514 |
| inverted schedule *(control)* | 0.03570 | 0.07696 |

- **The entropy signal is real.** Inverting it costs 26% more damage on code
  and 43% more on general, so "flat layers need more experts, peaked layers
  need fewer" is the correct direction. The control earned its place: without
  it, the middle row alone could not distinguish an informative signal from
  "any non-uniform schedule moves the number".
- **But uniform still wins.** With per-layer entropy spanning half a bit
  across 40 layers, no layer has exploitable slack; deviating from uniform
  costs in both directions, just far more in one of them.
- **The schedule also loses on speed**: same tg128 as uniform K=6 (71.09 vs
  71.01) but worse prefill (2337 vs 2511 pp2048). Worse on both axes.

## Throughput anatomy

`llama-bench`, r=3. Stock `llama-bench` has no `--override-kv`, so the
per-layer patch is what made these numbers obtainable at all:

| Config | pp2048 | tg128 |
|---|---|---|
| Q4 full, K=8 | 2284.9 | 67.10 |
| Q4 full, K=6 | 2510.8 (+9.9%) | 71.01 (+5.8%) |
| Q4 full, entropy schedule | 2337.6 | 71.09 |
| Q4+REAP-25%, K=8 | 2521.0 (+10.3%) | 65.98 (−1.7%) |
| **Q4+REAP-25%, K=6** | **2770.6 (+21.2%)** | **69.34 (+3.3%)** |

Pruning reproduces the anatomy seen on the 128-expert model: **prefill faster**
(+10.3%, fewer expert bytes streamed per ubatch), generation marginally slower
(−1.7%, kernel shapes). Composed with K=6 the shipping point is 16.23 GiB with
+21.2% prefill and +3.3% generation against stock — while being 4.6 GiB
smaller.

## What this closes, and what it does not

R7 kills **static per-layer** scheduling on this model: no `.poeroute`
schedule format, no per-layer budget search. It does **not** say anything
against **token-adaptive** K, because per-layer entropy is a *mean over
tokens* and carries no information about token-level variance.

The cheap discriminator for that question is a histogram, not a kernel: the
per-token distribution of "how many experts are needed to reach probability
mass T". If that distribution is tight, adaptivity dies with the schedule; if
it is wide, adaptivity has headroom a static schedule structurally cannot
reach. One observer field answers it, and it should be answered before any
runtime work.

## R9 — that histogram, measured: adaptivity by probability mass is dead

`poe-profile --metric routing` now keeps the whole distribution of min-k per
mass threshold, and `poe routing-budget --profile` reports it. Qwen3.6, 8192
tokens, 40 layers pooled, bins 4 experts wide:

| mass | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| 80% | 140.9 | 144 | 156 | 164 | 180 |
| 90% | 180.2 | 180 | 192 | 200 | 212 |
| 95% | 206.9 | 208 | 216 | 224 | 232 |
| 99% | 239.5 | 240 | 244 | 248 | 252 |

**The top-8 the model actually runs holds 15.7% of the router's probability
mass** (per layer 9.4%–21.6%). On general text: 18.2%, and 132.4 experts for
80% of the mass.

Three things follow, and together they close the question:

1. **Probability mass is not what governs which experts matter here.** A
   model running 8 of 256 experts that carry a sixth of the mass works fine.
   So a gating policy driven by a mass threshold is not a smaller version of
   what the model does — it is a different, far more expensive thing: 80% of
   the mass costs 141 experts, 17× the current budget.
2. **Token-to-token variation is modest**: p50 → p99 is +14% at 80% mass and
   +11% at 90%. Even granting a mass-threshold policy, the headroom between
   the median token and the tail — the only thing adaptivity can recover over
   a fixed K — is a small fraction of an already impossible budget.
3. **Code is not a narrower workload than general text**, which is the
   opposite of the intuition. It needs *more* experts for the same mass
   (140.9 vs 132.4) and its top-8 holds *less* of it (15.7% vs 18.2%).

This also explains R3's shape. With mass spread this thin, no expert is
free to drop and none is catastrophic — KLD doubling per expert removed is
what a flat router looks like from the outside.

**Consequence:** the ggml-cuda work for token-adaptive K stays unbuilt. If
adaptive routing is ever revisited it must be driven by measured *output
contribution*, not by router probability — which is also what the per-expert
precision result found, where REAP saliency beat selection frequency
([per-expert precision](per-expert-precision.md)).
