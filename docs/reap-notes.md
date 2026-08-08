# REAP: Router-weighted Expert Activation Pruning — implementation notes

Working notes for POE's REAP metric (milestone M4 of the roadmap in
vision.md), distilled from the reference implementation and paper:

- Reference repo: <https://github.com/CerebrasResearch/reap> (Apache-2.0)
- Paper: *REAP the Experts: Why Pruning Prevails for One-Shot MoE
  Compression* — Lasby, Lazarevich, Sinnadurai, Lie, Ioannou, Thangarasa,
  <https://arxiv.org/abs/2510.13999>

## 1. Core idea

One-shot (no retraining) compression of sparse MoE models by **pruning whole
experts**, not merging them. The paper's central argument: expert merging
incurs an *irreducible* error from "functional subspace collapse" — the loss
of fine-grained, input-dependent routing control — whereas pruning keeps the
router's independent control over the surviving experts. REAP is the
saliency criterion used to pick which experts to drop, derived to minimize a
layer-output reconstruction-error bound.

## 2. Saliency criterion (paper Eq. 9)

```text
S_j = (1 / |X_j|) * Σ_{x ∈ X_j}  g_j(x) · || f_j(x) ||_2
```

- `X_j = { x : j ∈ TopK(g(x)) }` — only tokens actually routed to expert `j`
  contribute.
- `g_j(x)` — the router gate value for expert `j` on token `x` (post-softmax,
  renormalized over the top-k; see below).
- `|| f_j(x) ||_2` — **L2 norm over the hidden dimension** of the expert's
  output vector for that token.
- Aggregation is a **mean over routed tokens** (not sum, not EMA). In the
  reference, per-batch means are combined across batches by an online
  tracker weighted with per-expert token counts (`expert_frequency`), so the
  result equals the global mean over `X_j`. Accumulated statistics are moved
  to CPU as they are gathered.
- **Per-layer**: saliency is computed and consumed independently in each MoE
  layer.

### Router weight normalization

- Gate values are the top-k softmax outputs.
- The reference renormalizes the **top-k gates to sum to 1** per token before
  use in the saliency (gather top-k weights, divide by their sum). This is
  controlled by `ObserverArgs.renormalize_router_weights`, default `True`,
  mirroring what the model itself does when `config.norm_topk_prob` is set.
- Ablation in the paper: without gate/logit normalization the mean accuracy
  drop worsens (2.6% vs 1.9%) — the renormalization matters. POE must
  normalize consistently with the target model's routing convention.

### Related criteria in the reference (useful as POE baseline metrics)

`prune_method` choices include: `frequency` (default), `ean_sum`
(activation-norm sum, unweighted), `ean_mean`, `weighted_ean_sum` (REAP
numerator without the mean), `reap`, `reap_l2`, `max_activations`.
These map directly onto POE's planned baseline metrics (routing frequency,
gate statistics, activation norm, REAP). The exact `reap` vs `reap_l2`
difference was not confirmed from the code excerpts reviewed — verify in the
source before mirroring it.

## 3. Expert selection given a target ratio

- **Per-layer, uniform prune ratio** — *not* a global cross-layer ranking.
  Each MoE layer drops its bottom `n` experts by saliency
  (`torch.topk(saliency, n, largest=False)`).
- Compression ratio: CLI default 0.25 (keep 75%); the paper's headline
  results use 0.50.
- Optional guards exist in the reference (`perserve_super_experts`,
  `perserve_outliers`; both default off).
- POE note: layer-adaptive allocation (COMPEL-style) is a planned
  improvement *on top of* this baseline — implement uniform per-layer first
  to match the reference, then compare.

## 4. Structural changes after pruning

- **Router rows removed**: the router weight matrix is sliced to the
  retained expert rows (`router.weight = router.weight[retained, :]`), bias
  handled analogously; `out_features` updated.
- **Experts dropped and implicitly remapped**: the retained expert modules
  are re-packed in retained-index order; because the router rows are sliced
  with the same order, new expert `i` matches router output `i` — no
  explicit remap table.
- **No weight compensation and no explicit router rescaling** in the code.
  The renormalization over surviving experts happens implicitly at
  inference (softmax/top-k now runs over the reduced logit set).
- Config expert counts (`num_experts` / `n_routed_experts` /
  `num_local_experts`) updated to the retained count.
- POE/GGUF mapping (M7): slice `blk.N.ffn_gate_inp.weight` rows, slice the
  packed `ffn_{gate,up,down}_exps` tensors along the expert dimension
  (ne[2]), update `<arch>.expert_count`, preserve everything else.

## 5. Layer-wise / memory-efficient calibration observer

Added to the reference (2026-03) to prune very large models on a single GPU:

- Statistics gathered via forward hooks registered per MoE layer; experts
  can be run one at a time.
- All accumulated state streamed to CPU with online/weighted-count trackers;
  no per-token activations retained; GC between layers.
- Enables calibration of 480B–1T-parameter models on one GPU.
- POE takes the same approach from day one: streaming accumulators,
  layer-wise observation, never retain activations (vision.md §8). Community
  reports of ~60 GB RAM peaks for 30B-class models with the older path are a
  property of the stack, not the algorithm — mmap + layer-wise observation
  is the fix.

## 6. Calibration data (reference recipe for agentic/reasoning checkpoints)

Composite set of **24,576 samples**, max sequence length **16,384**:

| Slice | Dataset | Samples |
|---|---|---|
| General coding | `theblackcat102/evol-codealpaca-v1` | 4,096 |
| Reasoning (code/math/science, equal thirds) | `open-r1/Mixture-of-Thoughts` | 12,288 |
| Single-turn tool calling | `Salesforce/xlam-function-calling-60k` | 4,096 |
| Agentic coding / multi-turn tool calls | `SWE-bench/SWE-smith-trajectories` | 4,096 |

Defaults in the reference are far smaller (one dataset, `batch_size 8`,
`model_max_length 2048`); the 24.6K × 16K recipe is what their released
checkpoints use. POE's convergence-driven calibration (milestone M5) exists
precisely to avoid hardcoding either extreme: measure ranking stability and
stop when the pruning decision is stable.

## 7. Architectures handled by the reference

`MODEL_ATTRS` maps HF architecture names to module attribute names: Qwen3-MoE
(and Qwen3-Coder), Mixtral (`w1/w2/w3`), Llama-4 (**fused** `gate_up_proj`),
DeepSeek-V2 / GLM-4-MoE (`n_routed_experts`), Ernie-4.5-MoE, gpt-oss-20b.
Paper/README additionally report runs on GLM-4.5/4.6, Kimi-K2, Kimi-Linear,
MiniMax-M2, DeepSeek-V3.2, Qwen3-Coder-480B (20B–1T params), with pruned
checkpoints released on HuggingFace.

Unverified details (check the source when implementing M4): gpt-oss
fused-tensor slicing path (the entry claims `fused: False`, which conflicts
with HF's fused layout), and shared-expert treatment (shared experts are not
routed, so they should be untouched by design).

## 8. Reported quality / safe ratios

- **50% expert pruning**: "near-lossless" on code generation and tool
  calling for Qwen3-Coder-480B and Kimi-K2; mean accuracy drop **1.9%**
  across non-agentic coding evals over five models.
- REAP consistently beats expert merging and prior criteria
  (frequency-only, activation-norm-only) on *generative* benchmarks;
  merging degrades disproportionately there.
- 25% is the conservative default; 50% is the headline and presented as
  safe for large code/agentic models with the matched calibration mix.
  No claims found above 50% — POE should warn beyond the validated range
  (safety rules in vision.md §23).

## 9. Licensing

The reference implementation is Apache-2.0. POE is MIT and implements the
algorithm independently (C11, GGUF-native); these notes document the method
for that purpose. Credit the paper in user-facing docs.
