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

### State: it runs, and only the large-batch MMQ path is broken

The patch works end to end on GPU for **generation**, which is what a split
checkpoint exists for:

| | size | tg128 |
|---|---|---|
| split, 128 hot at Q4_K + 128 cold at Q2_K | **15.84 GiB** | **57.06 ± 0.21 t/s** |
| the same model unsplit | 19.29 GiB | 66.51 ± 0.42 t/s |

3.45 GiB smaller for 14.2% slower generation — within noise of the 13.1%
this cost was predicted to be, from doubling the expert budget as a proxy.

Prefill at a large micro-batch still crashes, because that is the only path
that reaches MMQ. `-b 8 -ub 8` keeps `mul_mat_id` on the `mul_mat_vec_q`
kernels and prefill runs fine (`[1]1.6484,[2]1.7515,[3]1.6757,[4]1.6852`),
which is how quality can be measured while the MMQ path is unfixed.

### The MMQ failure

CPU was validated first: with `CUDA_VISIBLE_DEVICES=""` the split checkpoint
scores a sane perplexity, so the format, the loading, the clamped ids and the
per-slot merge are right.

**The large-batch CUDA path fails with an illegal memory access**, localized to:

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
| **the ids themselves** | a probe inside `mmq.cu` copies them to the host before the launch: `ne02=111 n_used=8 si1=256 min=0 max=110 out_of_range=0 first=[3 37 5 46]` — exactly right, and it still crashes |
| an unaligned expert count | 111 is odd, so the split was redone at 128/128 — same crash |

One real bug was found and fixed on the way: `ggml_clamp` is in-place and
returns a view of its input (`ggml.h` says so), so the hot ids, cold ids and
mask each need their own cast of the selection — sharing one buffer made the
clamps race, which CPU graph order hides.

**What the eliminations leave.** CPU + split works, CUDA + unsplit works,
CUDA + split fails *even when every id is zero*. The only invariant across
all failing configurations is a `mul_mat_id` whose `src0->ne[2]` is 111 while
the model's `n_expert` is 256 — a shape stock llama.cpp never produces, since
its expert tensors always carry the full expert count.

The probe settles the ids: they are correct on the device, in range, with
the stock stride. So the fault is on the **other** input — the activation
staging. `quantize_mmq_q8_1` with `has_ids` writes into `src1_q8_1`, sized

```c
ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
    ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq)
```

and scatters through the inverse map `ids_src1` that `mm_ids_helper` builds
(`dedup_bcast` is on here, because `ne11 == 1` — MoE activations are
broadcast across experts). Nothing in that expression mentions the expert
count, which is the one thing this checkpoint changes, so the next step is to
read `mm_ids_helper` and check what it writes into `ids_src1` when the ids
span a *subset* of `[0, ne02)` — every stock MoE covers the whole range,
because its tensors carry every expert the router can name.

Worth trying before that, because it is one flag: `GGML_CUDA_FORCE_MMQ=0`
(distinct from `FORCE_CUBLAS`, which was already tested) would take the
quantizer that is crashing out of the picture entirely.

`POE_SPLIT_PASSES=1` is left in deliberately: it is what turned "the split
crashes" into "the hot half alone crashes", which is half the search space
in one run.
