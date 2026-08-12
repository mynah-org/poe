# llama.cpp patches

Optional patches to llama.cpp that enable POE experiments the stock runtime
cannot express. POE never requires them: every static command works against
an unpatched build, and the patches change no file format.

## `llama.cpp-per-layer-k.patch` — per-layer expert budget (R7)

Stock llama.cpp has one global `expert_used_count`. It can be overridden at
load time (`--override-kv <arch>.expert_used_count=int:6`), but the value
applies to every block equally. This patch makes the budget **per layer**, so
a flat-router layer can keep the nominal top-k while a peaked one runs fewer
experts.

The whole change is 34 lines in `src/llama-graph.cpp`. It hooks
`build_moe_ffn`, which already receives the layer index `il`, so one
injection point covers every MoE architecture — no per-model edits.

```sh
cd llama.cpp            # generated against b1-69bf643, the same commit
                        # docs/router-observer.md was verified against
git apply /path/to/poe/tools/patches/llama.cpp-per-layer-k.patch
cmake --build build -j
```

Usage — a comma-separated budget per layer, read once from the environment:

```sh
POE_K_SCHEDULE="8,8,6,6,4,4,..." llama-server -m model.gguf
```

Rules the patch enforces:

- a missing entry, or a value `<= 0`, leaves that layer at its nominal top-k;
- values are **clamped upward-never**: a schedule can only reduce the number
  of experts a layer executes, never raise it;
- with the variable unset the build behaves exactly like stock llama.cpp.

The reduction is physical, not cosmetic: `n_expert_used` sizes the expert
selection tensor and every `mul_mat_id` that follows, so fewer expert matmuls
are scheduled. Graph shapes stay static because `il` is fixed at build time —
which is exactly why per-*layer* K is cheap while per-*token* K is not.

### Why the patch touches the aggregation loops too

Lowering the local `n_expert_used` alone is **not** sufficient, and the
failure is silent rather than loud. The expert-aggregation block at the end
of `build_moe_ffn` deliberately bounds its loops by `hparams.n_expert_used`
rather than the local count, to keep the number of add nodes small during
warmup (upstream PR 14753) — warmup can pass a *larger* local count. A
per-layer budget makes the local count *smaller*, so those loops build views
past the end of the now-shorter expert dimension and sum garbage into the
output. The patch bounds them by the minimum of the two.

The first version of this patch had exactly that bug. It was caught because
the validation runs a schedule of all-6 and requires it to reproduce the
known `--override-kv expert_used_count=6` result: the buggy build returned
KLD 2.72 where the reference is 0.028, and crashed `llama-bench`. **Always
validate a routing patch against a configuration whose answer is already
known** — the model keeps generating fluent text either way.

## `split-experts.patch` — per-expert precision (M9b), CPU-correct, CUDA open

Consumes a checkpoint from `poe split`: routed experts stored as a hot tensor
and a cold tensor at different precisions, reordered so the hot ones come
first with the router's rows permuted to match. Expert id keeps meaning "row
of the router", so `id < poe.split.hot_count` decides which tensor a slot
belongs to — no remap table.

The graph runs both halves over every slot and merges per slot, so routing
stays exactly what the router chose. Ids for each half come from clamping the
router's own selection (`clamp(id, 0, n_hot-1)` and `clamp(id - n_hot, 0,
n_cold-1)`), and the merge weight is `clamp(n_hot - id, 0, 1)` — for integer
ids that is exactly the "is this slot hot" indicator, with no extra tensors.

```sh
cd llama.cpp                     # generated against b1-69bf643
git apply /path/to/poe/tools/patches/split-experts.patch
cmake --build build -j
```

### State: correct on CPU, crashes on CUDA

**CPU is validated.** With `CUDA_VISIBLE_DEVICES=""` the split checkpoint
loads and scores a sane perplexity (1.6442 on the first code chunk), so the
format, the loading, the clamped ids and the per-slot merge are right.

**CUDA fails with an illegal memory access**, and it is localized:

```
Invalid __global__ write of size 4 bytes
  at quantize_mmq_q8_1<(mmq_q8_1_ds_layout)1, (bool)1>(const float *, const int *, ...)
  Access to 0xffffffdabafceba0 is out of bounds
```

That is the MMQ activation quantizer in its *with-ids* form, writing at a
sign-extended negative offset. Everything below was ruled out by experiment,
each one a build and a run:

| Ruled out | How |
|---|---|
| topk-moe fusion | `GGML_CUDA_DISABLE_FUSION=1` still crashes |
| CUDA graphs, batch size | `GGML_CUDA_DISABLE_GRAPHS=1`, `-b/-ub 128` |
| the cold half and the merge | `POE_SPLIT_PASSES=1` (hot half alone) crashes |
| non-contiguous ids | `ggml_cont` before the cast — no change |
| ids stride/layout | ids rebuilt as a full-width view with the stock `nb[1]` |
| **id values out of range** | `POE_ZERO_IDS=1` sends every slot to expert 0 — still crashes |
| degenerate id distribution | placeholders spread across the half instead of piled on one expert |
| MMQ versus cuBLAS | `GGML_CUDA_FORCE_CUBLAS=1` still crashes |
| **the patch breaking the normal path** | the same build scores a normal model fine: `[1]1.6840,[2]1.7739` |

One real bug was found and fixed on the way: `ggml_clamp` is in-place and
returns a view of its input (`ggml.h` says so), so the hot ids, cold ids and
mask each need their own cast of the selection — sharing one buffer made the
clamps race, which CPU graph order hides.

**What the eliminations leave.** CPU + split works, CUDA + unsplit works,
CUDA + split fails *even when every id is zero*. The only invariant across
all failing configurations is a `mul_mat_id` whose `src0->ne[2]` is 111 while
the model's `n_expert` is 256 — a shape stock llama.cpp never produces, since
its expert tensors always carry the full expert count.

Since in-range ids do not help, the ids the kernel reads are probably not the
ids the graph computes: the derived tensor's device buffer may simply never
be written. **The next step is therefore to observe, not to guess**: dump the
ids at compute time with a `ggml_backend_sched_eval_callback` (or
`GGML_SCHED_DEBUG=2`) and compare what `mm_ids_helper` receives against what
the graph built. `ggml/src/ggml-cuda/mmq.cu:183-201` is where they meet —
`expert_bounds` is sized `ne02 + 1` and the helper is handed `ids->data`
directly.

`POE_SPLIT_PASSES=1` is left in deliberately: it is what turned "the split
crashes" into "the hot half alone crashes", which is half the search space
in one run.
