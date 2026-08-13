# Residency — what fits on the device you have

`poe residency` answers one question: given a VRAM budget, what stays on the
GPU, what goes to the host, and which llama.cpp flags say so.

```sh
poe residency model.gguf --vram 24G --ctx 8192
poe residency model.gguf --vram 24G --emit-flags     # just the flags, for $(...)
```

It is a static command. It reads the checkpoint, computes, and prints — no
inference, no GPU, no rewrite. The output is advice for a stock llama.cpp;
nothing here needs a patch.

## The claim this command does *not* make

Every other axis in POE trades quality for size. This one does not.

**Offloading is quality-neutral.** A routed expert slab computed on the host
returns the same numbers as one computed on the device. There is nothing for
a saliency metric to protect, and "which layers should go to the CPU" is not
a quality question at all — it is a traffic question.

That matters because the obvious design is wrong. Ranking layers by REAP
saliency and keeping the salient ones resident *sounds* right and means
nothing: the displaced layer still runs, still routes to the same experts,
still produces the same output. It just produces it more slowly. And for a
uniform MoE, every layer is displaced at the same cost — same top-k, same
expert size, same bytes touched per token.

So the default eviction order is llama.cpp's own convention (deepest blocks
first) and the report says, in as many words, that it is a convention and not
a measurement. On this architecture per-layer differentiation has already
lost twice — R7 for routing budgets, M9a for precision — and a third
per-layer heuristic does not get to ship as a default on the strength of
sounding plausible.

Two profile-driven orders are available as **experiments with their control
attached**, and are labelled `[experiment, unmeasured]` in the output:

| `--rank` | displaces first | rationale |
|---|---|---|
| `last` (default) | the deepest blocks | llama.cpp's convention |
| `first` | the shallowest blocks | the only order `--n-cpu-moe` can express |
| `workset` | layers whose routing is narrowest | a narrow working set re-reads fewer distinct expert rows on the host |
| `workset-inverted` | the widest | the control that says whether `workset` is a signal at all |

The hypothesis behind `workset` is that a host-resident slab whose workload
concentrates on few experts gets better cache reuse. It is plausible, it is
cheap to test with a paired run, and until someone runs it, it is not the
default.

## The one ordering that is not a guess

A checkpoint from [`poe split`](per-expert-precision.md) stores its routed
experts as two tensors: the hot half at a wide type, the cold half — the
experts the calibration workload routes to least — at a narrow one. Two
tensors means `-ot` can place them **separately**, which is the only way
per-expert residency is expressible without patching the runtime.

There the ordering is dominance, not heuristic: displacing the cold half is
strictly less host traffic than displacing an arbitrary layer, because the
cold half is by construction the half the workload uses least. With 111 hot
experts of 256 on Qwen3.6, 69.1% of routed slots land in the hot set, so a
host that holds only cold experts is on the critical path for less than a
third of the traffic.

`poe residency` applies that rule on top of whichever `--rank` is chosen, and
says so in the report.

### What it actually costs, measured

The flags this command prints were run, not just printed. Qwen3.6-35B-A3B
split 128 hot Q4_K / 128 cold Q2_K, on GB10, paired in one session — the only
difference between the rows is the `-ot` override:

| placement | pp512 | tg128 | expert bytes on device |
|---|---|---|---|
| everything on the device | 1591.03 ± 17.16 | **58.59 ± 0.24** | 13.4 GiB |
| cold halves on the host | **2118.39 ± 12.84** | 8.93 ± 0.09 | 8.3 GiB |

**The cost has opposite signs on the two phases**, which no simple mental
model of offloading predicts. Generation collapses 6.6x: every token waits on
the host for its cold slots, and that wait is the critical path. Prefill gets
**33% faster**: it is compute-bound, the split runs two `mul_mat_id` passes,
and handing one of them to the CPU lets the two proceed at once.

State the hardware with the number, because it is doing work here: GB10 is a
unified-memory part, so "moving a tensor to the host" costs no transfer. On a
discrete GPU across PCIe the prefill result should not be expected to survive,
and the generation result would likely be worse. This is one machine, one
checkpoint, and it is a demonstration that the advice is executable — not a
throughput recommendation.

## Exact and estimated, never mixed

```
resident floor
  non-expert weights    2.5 GiB     exact
  KV cache              640.0 MiB   ESTIMATE  (8192 ctx, f16)
  compute allowance     512.0 MiB   ESTIMATE
  headroom for experts  20.4 GiB
```

- **Exact**: every byte figure taken from the tensor table — the non-expert
  weights, each slab, the totals. `expert_bytes_gpu + expert_bytes_cpu`
  equals the checkpoint's expert bytes exactly, and a test asserts it.
- **ESTIMATE**: the KV cache is
  `ctx · Σ_caching_blocks (n_embd_k_gqa + n_embd_v_gqa) · bytes_per_element`,
  with both widths read off each block's `attn_k` / `attn_v` tensors. It
  ignores context padding, sliding windows, multi-sequence caches and
  recurrent state. `--cache-type q8_0` and `q4_0` are modelled at their real
  block sizes. The compute allowance is a flat 512 MiB (`--reserve`) standing
  in for buffers that depend on batch size, vocabulary and backend — crude on
  purpose, and never dressed up as anything else.

### Count the blocks that cache, not the blocks

The KV term is derived from the tensor table rather than from `block_count`
and `attention.head_count_kv`, and on the model POE targets that is the
difference between right and 4x wrong.

**Qwen3.6-35B-A3B is a hybrid.** Of its 40 blocks only **10 carry attention
tensors** (`attn_q`, `attn_k`, `attn_v`); the other 30 hold `ssm_*` tensors
and use linear attention, so they have no KV cache at all — llama.cpp filters
them out of the cache entirely. Sizing 40 blocks from the metadata gives
640 MiB at 8192 context. Counting the ten that actually cache, at the widths
their own tensors declare, gives **160.0 MiB** — which is byte for byte what
llama.cpp reports:

```
llama_kv_cache: size = 160.00 MiB ( 8192 cells, 10 layers, 1/1 seqs)
```

The recurrent state those 30 blocks do hold is not modelled here. It is small
and, unlike a KV cache, independent of context length — but it is real, so
the report warns about it rather than implying the number is complete.

When the checkpoint lacks the metadata an estimate needs, the fallback is
stated as a warning rather than absorbed silently:

```
warning: no attention.head_count_kv: the KV estimate assumes 8 heads
```

A budget too small for the resident floor is an error with a number attached,
not a plan that quietly offloads everything and pretends to fit.

## `--n-cpu-moe` versus `-ot`

`--n-cpu-moe N` keeps the MoE weights of the **first N layers** on the host.
It is printed only when the plan happens to be exactly that — a whole prefix,
both halves of a split included. In every other case it would be a different
placement from the one the byte accounting describes, so it is not printed at
all. `-ot` is always emitted and is always the authoritative form.

When a whole class of tensors is displaced, the pattern collapses to
`blk\.[0-9]+\.` rather than enumerating every block.
