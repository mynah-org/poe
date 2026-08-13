# Super experts — rare, enormous, and unsafe to delete

arXiv:2507.23279 reports a failure mode with an unusually sharp edge: a
handful of experts carry activation magnitudes far above their neighbours,
and deleting as few as **three** of them breaks Qwen3-30B-A3B outright. What
makes them dangerous is that the cheap safety signal points the wrong way —
they are *rarely selected*, so any frequency-driven ranking sorts them
straight to the bottom of the prune list.

POE detects them and, by default, refuses to delete them.

```sh
poe plan model.gguf --profile coding.poeprofile --prune 25%
#   super     69 activation outliers flagged (21 of them rarely selected)
#             0 rescued from this cut  — the ranking was already keeping them

poe plan model.gguf --profile coding.poeprofile --prune 25% --no-protect-super-experts
poe plan model.gguf --profile coding.poeprofile --super-z 4      # widen the net
```

## What counts as one

Per layer, over `actnorm_mean` (the mean unweighted ‖expert output‖₂, present
in any profile captured with `--metric reap`):

```
z = (actnorm[e] − median) / (1.4826 · MAD)      flagged when z ≥ 6
```

Median and MAD rather than mean and standard deviation, because the
distribution is heavy-tailed *by hypothesis*: a mean would be dragged by the
very outliers being searched for, and a standard deviation would be inflated
by them until they no longer look unusual. The 1.4826 puts MAD on the same
footing as a σ for normal data, so a threshold of 6 means roughly what it
usually means.

**Rarity is reported, not required.** An expert is separately marked when the
workload routes to it less than half the uniform share; the intersection
(huge *and* rare) is the published signature. Protection covers every
activation outlier, because rarity is what makes them easy to lose, not what
makes them matter.

**A flat layer is undecidable, not clean.** If every expert in a layer has
the same activation norm, MAD is zero and no z-score exists. Those layers are
counted and reported separately rather than folded into "no outliers here" —
the detector must not be able to claim safety it did not establish.

## How protection works

Flagged experts are lifted out of the cut and **the next candidates up the
ranking take their place**. The per-layer keep count is unchanged — `poe
apply` requires it to be uniform — and so is the byte accounting: protection
never silently shrinks the saving you asked for. If a layer ever holds more
flagged experts than the cut can spare, the least salient of them are pruned
anyway and the plan says so loudly, with a count.

The plan file records what happened, so an artifact carries its own evidence:

```json
"super_experts": { "protected": true, "flagged": 69, "rare": 21,
                   "rescued": 15, "pruned_anyway": 0 }
```

`rescued` is the only number that says whether the guard did anything: it
counts experts the ranking *would have deleted* and protection kept.

## Measured: it depends entirely on the ranking

Qwen3.6-35B-A3B Q8_0, coding profile, 40 layers × 256 experts (10 240 experts
in total). **69 activation outliers, 21 of them also rare** — 0.67% of the
model, and the paper's threshold for catastrophe is three.

| ranking | prune | super experts the cut would have deleted |
|---|---|---|
| `reap` | 25% | **0** |
| `reap` | 40% | **0** |
| `reap` | 50% | **0** |
| `gate` | 25% | 5 |
| `frequency` | 25% | 15 |

Two things follow, and they point in opposite directions:

1. **Under REAP the guard is free insurance.** REAP saliency already keeps
   every flagged expert, at every cut depth tested up to 50%. That is not an
   accident: REAP scores gate · ‖output‖ averaged over the tokens actually
   routed to an expert, so a rare-but-enormous expert scores *high*. The
   protection changes nothing, costs nothing, and the report says as much
   instead of implying a rescue that did not happen.
2. **Under the cheaper rankings it is a real rescue.** Frequency would delete
   15 of them at a 25% cut, gate 5. This is the published failure mode,
   reproduced on the target model with a number attached — and it is an
   independent reason to prefer REAP that has nothing to do with the quality
   measurements in [quant-vs-prune.md](quant-vs-prune.md).

Protection is on by default anyway. A guard that only matters for the
rankings a user might reach for when they have no REAP data is exactly the
guard that should not need to be asked for.
