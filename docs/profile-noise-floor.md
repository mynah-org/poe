# The noise floor of profile overlap (C1)

POE's case for task-specific pruning rests on one measurement: profiles from
different workloads disagree about which experts to delete. The recipe doc
quoted that disagreement as a Jaccard of ~0.52 between coding and general
prune sets.

A disagreement number means nothing without knowing how much two profiles of
the *same* workload disagree. That control had never been run. This is it.

## The control

Two **disjoint halves of one coding corpus**, profiled independently on the
same checkpoint with identical settings (8192 tokens, `--metric reap`). Same
workload, same everything — only the sampled text differs. Whatever they
disagree about is sampling variance.

| checkpoint | pair | REAP bottom-25% Jaccard | top-25% Jaccard |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | **control** — code-A vs code-B | **0.636** | 0.803 |
| | code-A vs general prose | 0.226 | 0.065 |
| Qwen3.6-35B-A3B Q8_0 | **control** — code-A vs code-B | **0.588** | 0.567 |
| | code-A vs general prose | 0.230 | 0.108 |
| | code-B vs general prose | 0.230 | 0.116 |

"General prose" is wikitext-2 test, a domain with no relationship to source
code. Both code halves land on the same cross-domain figure to three decimal
places on the target model, from independent text — the signal is stable in a
way the noise floor is not.

## What this settles, and what it retires

**Workload identity is real, and the effect is much larger than the published
number suggested.** Against a genuine domain contrast the prune sets agree on
0.23, against a noise floor of 0.59–0.64. The signal sits **~2.8x below the
floor**, on two different checkpoints, two different expert granularities
(128 and 256), and two independently sampled coding corpora. Task-specific
pruning produces a genuinely different artifact.

**The specific figure of ~0.52 does not survive, and should not be quoted
again.** Its provenance is confirmed — 0.517, the REAP bottom-set Jaccard
between the two original 30B profiles — but it sits only 0.12 below this
checkpoint's noise floor of 0.636. On its own it demonstrates nothing:
sampling variance at 8192 tokens produces disagreements of that size between
profiles of *identical* workloads.

The reason it was weak is not the metric but the contrast. The corpus behind
the original "general" profile was, in hindsight, adjacent to the coding
domain rather than a different domain. A weak contrast measured with a noisy
metric returns a number that looks like a result and is not one. Replacing
the general corpus with genuinely unrelated prose moves the same metric from
0.517 to 0.226 — the effect was always there, and the old measurement was
mostly reporting its own noise.

## Consequences worth carrying

1. **Publish overlap numbers with their floor, or not at all.** A Jaccard
   between two profiles is uninterpretable alone. `poe compare` reports a
   difference; only a same-workload control says whether that difference is
   an effect.
2. **8192 tokens is enough to decide a prune set, and not enough to make the
   prune set stable.** Convergence was measured as *self*-stability — the
   set stops moving between consecutive checks of one run — and that is a
   weaker property than reproducibility across independent samples. The same
   corpus twice gives 0.59–0.64, not 0.95. Anything that compares two
   profiles must budget for that.
3. **The floor is a property of the metric, not of the corpus language.**
   Measured again inside Chinese on unrelated text in a different script, it
   lands at 0.551 against English's 0.566 (see
   [language-experts.md](language-experts.md)). That reproducibility is what
   lets a floor measured once be used to read a different comparison.
4. **Fine granularity costs top-set reproducibility.** The 30B's top-25% set
   reproduces at 0.803 across samples; Qwen3.6's, with 256 experts instead
   of 128, at 0.567. More, smaller experts means the identity of the *most
   used* ones is less determined by the workload — while the cross-domain
   contrast stays just as sharp.
