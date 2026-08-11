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
