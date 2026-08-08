# M2 router observer — llama.cpp instrumentation design

How `poe profile` will observe MoE routing per token without patching the
inference backend. Verified against ggml-org/llama.cpp master
(commit `69bf6437914`, 2026-08-08); line references below refer to that
checkout. Pin the llama.cpp commit when the observer lands and add a
startup self-check that the expected tensor names exist for layer 0 —
graph-internal names are stable in practice but not API-guaranteed.

## Mechanism: the scheduler eval callback

llama.cpp exposes `ggml_backend_sched_eval_callback` through public API
(`llama_context_params.cb_eval` / `cb_eval_user_data`,
`include/llama.h:376`). The scheduler calls it twice per graph node:

- **ask phase** (`ask == true`, before computing): return whether we want
  to observe this node. The full `ggml_tensor *` is available — name, op,
  shapes — so a cheap `strncmp(t->name, "ffn_moe_", 8)` prefix filter works.
- **collect phase** (`ask == false`, after the node's sub-range is computed
  and the backend synchronized): read the tensor. GPU-resident tensors are
  read with `ggml_backend_tensor_get()` (one small device-to-host copy).
  Returning `false` aborts the decode — our error path.

This is the exact mechanism `llama-imatrix` and `examples/eval-callback`
use in-tree (`tools/imatrix/imatrix.cpp:225`, registered as a plain
`cb_eval`). It works on every backend, including CUDA, and is unaffected by
`--cpu-moe` / `--n-cpu-moe` (those only override tensor buffer placement —
expert tensors become host-resident, which makes observation cheaper).

## Tensor names in the graph

`build_moe_ffn` (`src/llama-graph.cpp:1914`) names every intermediate as
`<name>-<layer>`. What the observer captures per MoE layer:

| Purpose | qwen3moe (softmax gating, norm_w) | gpt-oss (delayed softmax, biases) |
|---|---|---|
| Router distribution | `ffn_moe_probs-%d` (true softmax over all experts) | `ffn_moe_probs-%d` — **raw biased scores**, not probabilities (softmax happens only over the selected k); softmax on host if a full distribution is needed |
| Selected expert IDs | `ffn_moe_topk-%d` — I32 `[k, n_tokens]`, **non-contiguous** (a view of argsort): honor `t->nb[]` strides when copying | same |
| Applied gate weights | `ffn_moe_weights_norm-%d` (renormalized top-k) | `ffn_moe_weights_softmax-%d` |
| Per-expert FFN output (REAP norms) | `ffn_moe_down-%d` `[n_embd, k, n_tokens]`, pre-weighting; slot *i* pairs with `ffn_moe_topk` row *i* | `ffn_moe_down_biased-%d` (the biased output is the applied one) |

Generic rule: take the **last** `ffn_moe_weights*` tensor in program order
as the applied gates (`weights_norm` > `weights_softmax` > `weights_scaled`
> `weights`), and the last `ffn_moe_down*` as the expert output.

Name-free fallback for expert IDs: on any `GGML_OP_MUL_MAT_ID` node,
`t->src[2]` is the ids tensor — the trick imatrix uses; immune to renames.

## Fusion caveat (CUDA/Vulkan)

CUDA fuses the router softmax→top-k→gather pipeline into one `topk_moe`
kernel and never materializes the intermediates. The callback mechanism
inherently protects us: asking for a tensor ends the fused sub-range at
that node, so it gets computed and written for real.
`GGML_CUDA_DISABLE_FUSION=1` exists as belt-and-suspenders.

## Cost model

Per observed tensor the scheduler inserts a graph split + backend sync.
For a Qwen3-30B-A3B-class model (48 layers, 128 experts, k=8, n_embd 2048):

- routing-only capture (`probs` + `topk` + `weights`): ~30 KB/token of
  D2H traffic — negligible; the real cost is the ~150 syncs/token and lost
  kernel fusion. Fine for calibration runs.
- REAP capture (adds `ffn_moe_down`): ~3.1 MB/token, expect roughly
  1.2–2× slower decode. **Make expert-output capture opt-in**
  (`--metric reap`) and prefer batched prefill-style scoring for corpus
  profiling — each callback then carries all batch tokens at once, which
  amortizes the syncs dramatically.

Token attribution: tensor columns follow the ubatch order of our
`llama_batch`; with a sequential batch the mapping is identity.

## Consequences for POE's design

1. **No fork, no patches.** POE links llama.cpp (or embeds it as an
   optional backend) and installs the callback via public API. Prior art
   that patched kernels (adaptive-gate and expert-cache experiments,
   llama.cpp discussions #25136 / #24528) did so because they needed to
   *change* execution; observation alone never requires it.
2. **The `poe` core CLI stays zero-dependency.** The observer lives in an
   optional build target that requires a llama.cpp checkout
   (`make profiler LLAMA_DIR=...`); static commands never pull it in.
3. **`--override-kv <arch>.expert_used_count=int:N`** changes k at load
   time with zero code — the R-track's fixed-K experiments (R1/R3) get
   their baseline mechanism for free, though *physical* skipping still has
   to be proven with kernel counters, not assumed.
4. Streaming accumulators consume the callback directly: routing frequency
   and gate statistics (M2) come from `topk` + `weights`; REAP saliency
   (M4) adds the L2 norms of the expert-output slots. Nothing is retained
   per token.
