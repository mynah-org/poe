# Router calibration after pruning — there is nothing to calibrate

The idea behind router calibration is that `poe apply` compacts the router's
rows in keep order, which is exact for the survivors, but leaves the routing
*distribution* shaped by experts that no longer exist. Distilling the original
model's next-token distribution into the router alone would repair that, on
unlabeled data, touching only the 80 MiB of routers rather than the 32 GiB of
experts — the cheapest possible intervention against the largest damage
pruning causes.

The premise is false on this architecture, and the reason is worth writing
down because it is a property of how the runtime applies routing weights, not
of any measurement.

## The chain

1. **`poe apply` copies surviving router rows byte for byte.** The rows are
   sliced out of the tensor and written unchanged — no rescaling, no refit.
   So a surviving expert's logit is bit-identical to the original.

2. **The runtime turns logits into weights in this order**
   (`build_moe_ffn`): `probs = softmax(logits)` over *every* expert present,
   then top-k selection over those probs, then — when `norm_w` is set —
   `weights /= sum(weights)`.

3. **Qwen3.6 sets `norm_w`.** `qwen35moe` passes `true` in that position, so
   the applied weights are renormalized to sum to one.

## The consequence

Write `Z` for the original softmax denominator and `Z'` for the pruned one,
summed over the surviving logits only. For a surviving expert *e*:

```
probs'[e] = exp(logit[e]) / Z'  =  probs[e] · (Z / Z')
```

The two differ by one positive constant, shared by every survivor. Therefore:

- **Selection is unchanged.** Top-k is invariant under a positive monotone
  rescaling, so the pruned model selects exactly the original ranking
  restricted to the experts that survive.
- **The weights are unchanged too.** The constant cancels in the
  renormalization: `probs'[sel] / Σ probs'[sel] = probs[sel] / Σ probs[sel]`.

So after pruning the router's behaviour is **exactly the conditional
distribution of the original router over the surviving set** — in both the
choice and the weights. The deleted experts leave no residue to correct.

There is consequently no training-free recalibration to ship. Any gain from
router distillation would have to come from *re-learning* routing to make
better use of the survivors — a token whose first choice is gone might be
better served by something other than its second choice — and that is
training, which the project's scope rules out by default, and which is not a
small piece of work in a zero-dependency C11 codebase.

## What this reframes

The largest damage pruning causes — off-domain top-1 agreement falling from
100% to 74% at 25% and 64% at 50%, measured in
[reap-coding-recipe.md](reap-coding-recipe.md) — is therefore **capability
loss, not miscalibration**. The model is not misrouting; it is routing
correctly to a set that no longer contains what it wanted. Calling router
calibration "the cheapest intervention against the largest damage" was wrong
on the cheap part: there is no closed-form component to it at all.

## What would revive it

One condition, and it is checkable per architecture: **`norm_w` unset**.
Without the renormalization the applied weights are the raw softmax masses of
the selected experts, which sum to whatever mass they hold — on Qwen3.6 the
top-8 holds only 15.7% of the router mass. There, deleting an expert a token
wanted *shrinks the total weight applied to its FFN output*, an effect that
grows with prune depth and admits a closed-form per-expert correction from a
profile POE already captures.

That flag is not in the checkpoint (Qwen3.6 carries no
`expert_weights_norm` key; the runtime supplies the default per architecture),
so it cannot be read off a GGUF — it has to be checked against the runtime
for each new architecture before assuming pruning is calibration-neutral
there.
