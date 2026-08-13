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

## `split-experts.patch` — per-expert precision (M9b), working on GPU

Consumes a checkpoint from `poe split`: routed experts stored as a hot tensor
and a cold tensor at different precisions, reordered so the hot ones come
first with the router's rows permuted to match. Expert id keeps meaning "row
of the router", so `id < poe.split.hot_count` decides which tensor a slot
belongs to — no remap table.

The graph runs both halves over every slot and merges per slot, so routing
stays exactly what the router chose. Ids for each half come from the router's
own selection, rescaled and clamped into the half's range, and the merge
weight is `clamp(n_hot - id, 0, 1)` — for integer ids that is exactly the "is
this slot hot" indicator, with no extra tensors.

```sh
cd llama.cpp                     # generated against b1-69bf643
git apply /path/to/poe/tools/patches/split-experts.patch
cmake --build build -j
```

### State: generation and prefill both run on CUDA

| | size | pp512 | tg128 |
|---|---|---|---|
| split, 128 hot at Q4_K + 128 cold at Q2_K | **15.84 GiB** | **1588.36 ± 5.91** | **58.90 ± 0.25** |
| the same model unsplit | 19.29 GiB | 2261.96 ± 38.78 | 67.65 ± 0.29 |

3.45 GiB smaller for **-12.9% generation and -29.8% prefill**. The two passes
cost prefill roughly twice what they cost generation, which is what a
compute-bound phase should do with the expert matmul work doubled.

Before the fix the only way to prefill at all was `-b 8 -ub 8`, which keeps
`mul_mat_id` off MMQ: that path measures **244.69 ± 0.56 t/s**, so the fix is
worth **6.5x** on prompt processing. It is a fix to
`ggml/src/ggml-cuda/mmid.cu`, and how it was found is the rest of this
section.

### The MMQ failure: symptom

**The large-batch CUDA path failed with an illegal memory access**, localized
to:

```
Invalid __global__ write of size 4 bytes
  at quantize_mmq_q8_1<(mmq_q8_1_ds_layout)1, (bool)1>(const float *, const int *, ...)
  Access to 0xffffffdabafceba0 is out of bounds
```

That is the MMQ activation quantizer in its *with-ids* form, writing at a
sign-extended negative offset. Generation never reached it — `mul_mat_id` at
one token per pass runs on `mul_mat_vec_q` — so `-b 8 -ub 8` was the
workaround that let quality be measured while the path was broken, at a
sixth of the prefill speed.

CPU was validated first: with `CUDA_VISIBLE_DEVICES=""` the split checkpoint
scores a sane perplexity, so the format, the loading, the ids and the
per-slot merge are right.

Everything below was ruled out by experiment, each one a build and a run:

| Ruled out | How |
|---|---|
| topk-moe fusion | `GGML_CUDA_DISABLE_FUSION=1` still crashes |
| CUDA graphs, batch size | `GGML_CUDA_DISABLE_GRAPHS=1`, `-b/-ub 128` |
| the cold half and the merge | `POE_SPLIT_PASSES=1` (hot half alone) crashes |
| non-contiguous ids | `ggml_cont` before the cast — no change |
| ids stride/layout | ids rebuilt as a full-width view with the stock `nb[1]` |
| id values out of range | `POE_ZERO_IDS=1` sends every slot to expert 0 — still crashes |
| MMQ versus cuBLAS | `GGML_CUDA_FORCE_CUBLAS=1` still crashes |
| the patch breaking the normal path | the same build scores a normal model fine |
| the ids themselves | a probe in `mmq.cu` copies them to the host before the launch: in range, stock stride, no out-of-range value |
| an unaligned expert count | 111 is odd, so the split was redone at 128/128 — same crash |

### The cause: repeated expert ids inside one token

None of those is the fault, and neither is the partial expert count. The
invariant the split breaks is one nobody writes down, because in stock
llama.cpp it cannot fail: **the top-k ids of a token are distinct experts.**

The split needs an id for every slot in *both* halves, so the slots that
belong to the other half get a stand-in id inside this half's range. Two
stand-ins — or a stand-in and a real id — can land on the same expert. On
this checkpoint that is not an edge case: **410 of 512 tokens carry a
repeated id, 1559 slots in all, up to 7 slots on one expert.**

`mm_ids_helper` compacts *per token*: `warp_reduce_any` asks whether the token
used this expert at all, so a token contributes exactly one row per expert.
With a repeated id the second slot never gets a row, and therefore never gets
its entry in the inverse map `ids_src1` — which is left **uninitialized**.
`quantize_mmq_q8_1<has_ids>` then reads that entry as a row index, and writes
at whatever it happens to hold. Hence a negative, sign-extended offset, and
hence a crash that is identical when every id is zero: all-zero ids are the
maximal case of the same collision, not a separate mystery.

The bounds already assumed the other convention: `nex_prev` counts *slots*
with a lower expert id, while `it_compact` counts *tokens*, so `expert_bounds`
and the compact row count disagree as soon as an id repeats.

Two measurements settled it, both cheap:

- the ids probe now counts repeats — `dup_tokens=410/512 dup_slots=1559
  max_mult=7`, on the very run that crashes;
- `POE_SEQ_IDS=1` feeds the hot half ids `0..n_expert_used-1` per token: same
  shape, same stride, same partial expert count, same kernels — the only
  difference is that no two slots name the same expert. It **runs to
  completion at full batch**. That is the whole hypothesis, isolated.

### The fix — and precisely what it does not fix

`mm_ids_helper` compacts per *use* instead of per token: a warp-level prefix
sum gives every matching slot of a token its own row, and the shared-memory
store is sized `n_tokens * n_expert_used` to hold the worst case. Both the
generic and the specialized branch are changed; the generic one had a second
form of the same bug, keeping only the last match of a strided loop.

**That removes the illegal memory access. It does not make repeated ids
numerically correct on CUDA**, and the distinction matters to anyone reusing
this patch. Measured with a reproducer added to ggml's own test suite
(`ggml-repeated-ids-repro.patch`, applied on top of a build that already
carries the fix above):

| ids | path | result vs the CPU reference |
|---|---|---|
| one duplicated pair of 8 slots | f16 / q8_0 / q4_0, batch 1–256 | NMSE ≈ 0.14 |
| every pair duplicated | same | NMSE ≈ 1.0 |
| any duplication | q8_0 / q4_0 at batch 1 (vector path) | correct |

The damage is **local**: one wrong output row per duplicated slot, roughly
1/8 of the rows when 1 of 8 slots repeats, and it appears on both the
broadcast and the non-broadcast activation paths. Somewhere past
`mm_ids_helper` at least one more consumer still assumes an id cannot repeat;
this patch does not find it.

**Why the split checkpoint measures correct anyway.** Its duplicated slots
are the stand-ins, which the merge multiplies by zero, and the paired
measurement above (0.19σ against the `mul_mat_vec_q` path over 100 chunks)
says the shipped artifact is unaffected. That is a measurement, not a proof:
a stand-in can in principle collide with a *real* hot id, and nothing here
establishes which of the two colliding rows comes out wrong. A design that
cannot mask its duplicates must not reuse this patch and assume correctness.

One trap on the way, worth remembering: `warp_reduce_sum<width>` **cannot** be
used to sum within a token group. On Ampere and later it calls
`__reduce_add_sync` over the whole warp and silently ignores its `width`
template — it would have mixed neighbouring tokens together, quietly.

With distinct ids the new code produces exactly the old mapping (every prefix
is zero), so stock models are unaffected — and measured: pp512 on an unsplit
Q4_K checkpoint is **2364.22 ± 26.57 t/s** against **2394.90 ± 24.84 t/s** before the fix, 0.8σ
apart.

**And the fixed path is not merely non-crashing, it is the same model.** The
same checkpoint, the same reference log-probs and the same 100 chunks, with
only the kernel path differing:

| path | KLD code | same top-1 |
|---|---|---|
| `mul_mat_vec_q`, forced with `-b 8 -ub 8` | 0.012180 ± 0.000326 | 96.992 ± 0.107 % |
| MMQ, full batch, after the fix | 0.012268 ± 0.000319 | 96.937 ± 0.108 % |

0.19σ on KLD and 0.36σ on top-1 agreement. Worth doing as a *paired* run: a
first comparison against a number measured on another day suggested a 3σ drop
in top-1 that does not exist.

One real bug was found earlier on the way: `ggml_clamp` is in-place and
returns a view of its input (`ggml.h` says so), so the hot ids, cold ids and
mask each need their own cast of the selection — sharing one buffer made the
clamps race, which CPU graph order hides.

### `ggml-repeated-ids-repro.patch` — the reproducer, for upstream

ggml's own test suite never exercises this: `init_mul_mat_id_tensors` fills
each ids row with `i % n_mats` and shuffles, so within a row the ids are
distinct by construction. The patch adds a `repeat_ids` mode to
`test_mul_mat_id` and registers cases across f16/q8_0/q4_0, batch 1/16/256,
broadcast and not.

```sh
cd llama.cpp
git apply /path/to/poe/tools/patches/ggml-repeated-ids-repro.patch
cmake --build build --target test-backend-ops -j
./build/bin/test-backend-ops -o MUL_MAT_ID
```

Reported upstream as
[ggml-org/llama.cpp#27015](https://github.com/ggml-org/llama.cpp/issues/27015),
with this reproducer inline and the compaction change offered as a partial
fix — partial because it removes the abort without closing the numerical
disagreement, so it would land the added tests still failing.

It reproduces on stock llama.cpp, where the run does not merely disagree with
the CPU reference but **aborts**:

```
MUL_MAT_ID(type_a=q8_0,type_b=f32,n_mats=16,n_used=8,b=1,m=128,n=1,k=256,repeat_ids=2):
CUDA error: an illegal memory access was encountered
ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error
```

`mmid.cu` is byte-identical between the pinned b1-69bf643 and upstream master
88 commits later, and no commit has touched it. The CPU backend is the reference the comparison is made against,
and nothing in `ggml.h` documents a uniqueness requirement on ids.

### Diagnostics left in the patch, deliberately

- `POE_SPLIT_PASSES=1` runs the hot half alone. It is what turned "the split
  crashes" into "the hot half alone crashes", halving the search space in one
  run.
- `POE_IDS_PROBE=1` copies each `mul_mat_id`'s ids to the host and reports
  range, out-of-range count and repeats. Off by default, and the `getenv` is
  cached in a static — a probe on a hot path has to cost nothing when unused.
- `POE_ZERO_IDS=1` and `POE_SEQ_IDS=1` replace the hot half's ids with all
  zeros, or with one distinct expert per slot. Numerically meaningless, but
  the pair is what separates "the ids are wrong" from "the ids are not
  distinct".
