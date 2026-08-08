# .poeplan format (version 0)

A plan is a deterministic, persistent expert-removal decision: which experts
survive in each MoE layer of a specific checkpoint, plus exact byte
accounting. Plans are built by `poe plan` from one or more `.poeprofile`
files and consumed by `poe apply` (M7). JSON, human-diffable.

Reproducibility invariant: same model + same profiles + same parameters +
same POE version → byte-identical plan file.

## Envelope

| Key | Meaning |
|---|---|
| `poeplan` | format version (0) |
| `poe_version` | POE that produced the plan |
| `model_fingerprint` | `poe1:<hash>` of the source checkpoint; `apply` and `estimate` refuse a different model |
| `arch` | GGUF architecture string |
| `method` | ranking metric: `reap`, `frequency`, or `gate` |
| `prune_fraction` | requested fraction |
| `n_layers`, `n_experts`, `top_k` | source topology |
| `keep_per_layer` | uniform surviving expert count (bottom cut is clamped so at least `top_k` survive) |
| `expected` | exact on-disk accounting: `bytes_before/removed/after`, `params_before/removed` — pruned expert slices plus pruned router rows, computed from the tensor table, never estimated |
| `warnings` | conservativeness notes (small calibration, clamping, forced fingerprint mismatch, >50% outside REAP's validated range) |

## Layers

```json
{"layer": 17, "keep": [0,2,5, ...], "prune": [1,3, ...]}
```

`keep` and `prune` partition `0..n_experts-1`; both are sorted ascending.
**The remap rule for `apply`:** the new index of a kept expert is its rank
within the `keep` list — router rows and packed expert slabs are compacted
in the same order, so no separate remap table is needed.

## Selection semantics

Per layer, experts are ranked by the method's score from the profile(s);
the bottom `n_experts - keep_per_layer` are pruned. Multiple profiles are
combined by min-max normalizing each profile's per-layer scores to [0,1]
and summing with the user-given weights (`--profile p.poeprofile:0.7`).
Ties break deterministically by expert index. Uniform per-layer pruning
mirrors the REAP reference; layer-adaptive budgets are a planned
alternative (COMPEL-inspired), not a format change — only `keep_per_layer`
becomes per-layer.
