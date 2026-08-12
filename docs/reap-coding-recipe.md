# Recipe: coding-specializing a Qwen3 MoE with REAP

Field notes from pruning Qwen3-30B-A3B (Q4_K_M, 128 experts, top-8) into
smaller coding-specialized checkpoints with POE — what we ran, what the
numbers were, and what we learned along the way. Everything here is
training-free and reproducible: same model + same calibration text + same
parameters → byte-identical output GGUF.

Hardware used for the numbers below: a single NVIDIA GB10 (CUDA, unified
memory), llama.cpp CUDA build. Nothing in the recipe depends on that; any
machine that runs the model for a few thousand calibration tokens works.

## TL;DR

```sh
# one command, dataset -> pruned model (artifacts kept alongside):
poe forge Qwen3-30B-A3B-Q4_K_M.gguf --dataset code-calib.txt \
    --method reap --prune 25% -o qwen3-coding-reap25.gguf

# or stage by stage:
poe-profile model.gguf --dataset code-calib.txt --metric reap \
            --until-stable 0.99 -o coding.poeprofile
poe plan model.gguf --profile coding.poeprofile --method reap --prune 25% \
         -o coding-reap25.poeplan
poe estimate coding-reap25.poeplan model.gguf     # exact bytes, verified
poe apply coding-reap25.poeplan model.gguf -o pruned.gguf
poe validate pruned.gguf --plan coding-reap25.poeplan

# did it cost anything? (needs llama-server)
python3 tools/coding_eval.py model.gguf pruned.gguf
```

Result at 25%: **17.3 GiB → 13.2 GiB (−23.7%)**, coding eval **10/10 vs
10/10** against the full model, generation throughput unchanged
(~92 vs ~95 t/s). The win is memory, not speed — see below.

## 1. Calibration

- **Dataset**: ~1 MB of representative source code is enough to start.
  We used C/C++ files concatenated into one text file. The profiler
  streams it; no formatting needed.
- **How many tokens**: don't guess — measure convergence. `poe-profile
  --until-stable 0.99` checks, once per batch, the Jaccard stability of
  the would-be prune set and the rank correlation of expert saliencies
  between consecutive checkpoints. On Qwen3-30B-A3B the bottom-25% prune
  set reached Jaccard ~0.88 and rank correlation ~0.975 by 8192 tokens —
  the decision barely moves after ~4k tokens. Under 8192 tokens
  `poe plan` records a warning in the plan itself.
- **Metric**: `--metric reap` captures gate weight × expert-output norm
  (REAP saliency). Plain `--metric routing` (selection frequency) is
  cheaper and, in our runs, ranks the bottom set almost identically —
  the plan diff between `reap` and `frequency` prune sets is a useful
  sanity check (`poe diff a.poeplan b.poeplan`).

## 2. Workload identity is real

Profiles from different workloads disagree exactly where you'd hope:
coding vs general-text profiles on the same checkpoint overlap only
~0.52 (Jaccard) in their bottom-25% prune sets (`poe compare`, confirmed
independently by `poe diff` on the resulting plans). That disagreement
is the whole justification for *task-specific* pruning: a generic pruned
model and a coding pruned model are different artifacts. Weighted blends
(`--profile coding:0.7 --profile general:0.3`) are supported when you
want a hedge.

## 3. Plan, then look at it

`poe plan` is pure bookkeeping — it never touches the checkpoint. The
`.poeplan` is human-readable JSON carrying the per-layer keep/prune
partition and **exact** byte accounting computed from the tensor table
(pruned expert slabs + pruned router rows; estimates are never mixed with
exact numbers). Guards worth knowing:

- fingerprint binding: a plan refuses a different model unless `--force`;
- the bottom cut is clamped so at least top-k experts survive per layer;
- pruning beyond 50% is flagged as outside REAP's validated range.

## 4. Apply

`poe apply` streams mmap → output; the model is never materialized in
RAM. Kept expert slabs are written in ascending-id order, which *is* the
remap: the router rows are compacted in the same order, so no remap
table exists anywhere. Metadata is copied byte-for-byte except
`<arch>.expert_count` (patched in place) and `poe.*` provenance keys
(version, method, source fingerprint). Applying 25% to the 17.3 GiB
Qwen3 takes about as long as copying 13 GiB.

`poe validate pruned.gguf --plan plan.poeplan` closes the loop: it
recognizes the file as the plan's pruned output via the provenance
fingerprint and re-checks expert counts and exact bytes.

## 5. Verify behavior, not just structure

Structural validity says nothing about quality. `tools/coding_eval.py`
runs 10 small coding tasks (easy → hard, Python + C) at temperature 0
through llama-server and **executes** the generated programs against
hidden test cases — win/fail, no judgment calls. The hard tier is
deliberately discriminative (e.g. LIS on 20k elements times out unless
the model writes the O(n log n) algorithm).

Qwen3-30B-A3B, coding calibration, GB10:

| Variant | Size | Coding eval | Gen t/s |
|---|---|---|---|
| full Q4_K_M | 17.3 GiB | 10/10 | 95.0 |
| REAP 25% | 13.2 GiB (76.3%) | 10/10 | 91.9 |
| REAP 37.5% | 11.1 GiB (64.4%) | 9/10 | 92.7 |
| REAP 50% | 9.1 GiB (52.6%) | 9/10 | 97.7 |

The degradation profile is exactly what you want from a guardrail: easy
and medium tiers stay perfect all the way to 50%; the first cracks
appear on the hardest reasoning-heavy task (deterministic topological
sort: at 37.5% the generated program was inefficient enough to time
out, at 50% it ignored the tie-breaking rule). 25% is the free lunch;
beyond that you are trading hard-tier reliability for gigabytes, and
the eval quantifies that trade per checkpoint.

### Verify off-domain too, or the verification is worthless

The table above is measured on the domain the model was calibrated for,
and that is the one measurement that structurally cannot see what pruning
costs. Head to head at equal bytes on Qwen3.6
([docs/quant-vs-prune.md](quant-vs-prune.md)): a REAP-23.8% checkpoint at
Q4_K scores KLD **0.0327 on code** against a full-expert Q3_K's 0.0343 —
better — and **0.483 on wikitext** against 0.067. Same file size, 7.2×
the divergence, top-1 agreement down from 89% to 74%.

So the rule is: at a fixed byte budget spend on precision first, prune
shallow if at all, and always run one off-domain check. A pruned model is
not a smaller model, it is a narrower one, and the narrowing is invisible
from inside the domain.

## The second axis: reducing active top-k

Expert *pruning* shrinks the checkpoint; reducing the *active* expert
count per token (top-k) cuts compute — they are independent axes and
they compose. Two ways to run reduced K:

```sh
# zero-cost experiment, no checkpoint change (llama.cpp KV override):
llama-server -m model.gguf --override-kv qwen3moe.expert_used_count=int:6

# baked in at apply/forge time:
poe apply plan.poeplan model.gguf --top-k 6 -o pruned-k6.gguf
```

Measured on Qwen3-30B-A3B with the same coding eval:

| Config | Eval | Gen t/s |
|---|---|---|
| full, K=8 (stock) | 10/10 | 95.0 |
| full, K=6 | 10/10 | 107.6 |
| full, K=4 | 6/10 | 126.3 |
| REAP 25% + K=6 | **10/10** | **103.3** |

K=6 was free on this eval (+13% generation speed); K=4 broke medium and
hard tiers at once (wrong logic, even a C program using `new` as an
identifier). The cliff sits between 6 and 4, and the profile predicts
it: Qwen3's router is flat (mean gate entropy 6.16 of 7 bits), so each
marginal active expert carries a non-trivial share — you can drop the
8th and 7th, not half of them. The combined **REAP 25% + K=6** point is
the current sweet spot: 23.7% smaller *and* ~9% faster than stock, still
10/10. Use `poe routing-budget` for the exact active-parameter numbers
behind any K.

**Relation to adaptive-routing papers.** Static reduced K is the crude
form of what the adaptive-K line of work (Ada-K / AdaMoE-style null
experts / DynMoE, and the mass-K policies on this project's roadmap)
actually proposes: let the *router decide per token* how many experts to
run — typically by cutting the sorted gate distribution at a cumulative
mass threshold — so easy tokens use 3 experts and hard tokens use 8. At
a matched *average* K that dominates fixed-K on the quality/compute
Pareto. It cannot be faked with metadata: it needs a change in the
backend's MoE forward (threshold selection instead of fixed top-k),
which is why it is a later milestone here. The K=6-free / K=4-broken
result above is the motivation: if the average can sit near 5–6 with
peaks at 8 for the tokens that need them, dynamic K collects the speed
without paying the cliff.

## Performance anatomy

`llama-bench`, 3 repetitions, prompt 2048 / generation 128, at three
context depths (GB10):

| tg128 t/s (pp2048 t/s) | depth 0 | depth 8k | depth 16k |
|---|---|---|---|
| full, K=8 | 95.9 (2888) | 69.5 (2285) | 54.7 (1849) |
| REAP 25%, K=8 | 92.6 (3257) | 68.2 (2502) | 53.6 (1977) |
| REAP 25%, K=6 | 103.7 (3712) | 72.3 (2737) | 56.4 (2099) |

What the numbers say:

- **Pruning makes prompt processing faster (+13%), not slower.** In a
  2048-token ubatch essentially every expert is activated by some token,
  so the whole expert weight set is streamed per ubatch — 25% fewer
  experts is 25% fewer weight bytes to read. Fewer experts also means
  more tokens per expert matmul, which GPUs like.
- **The small generation penalty is real but bounded (−2 to −3.4%), and
  it is not the router "struggling".** The router is one `[embd × E]`
  matvec plus a partial sort — *cheaper* with 96 experts than with 128.
  At batch 1 the per-token weight traffic (8 expert slabs) is identical;
  the residual delta is kernel-shape territory (96/80 experts are not
  the power-of-two shapes the kernels were tuned on), and reduced K
  flips it positive anyway.
- **Long context dilutes everything.** Generation speed is FFN (constant
  per token) + attention KV reads (linear in depth): from depth 0 to 16k
  the full model drops 95.9 → 54.7 t/s, and the spread between variants
  compresses (K=6 advantage +8% → +3%, REAP penalty −3.4% → −2%). MoE
  surgery matters most at short-to-medium context; at long context the
  KV cache is the wall, which is a different battle (GQA, KV quant).

## Memory anatomy

Allocator numbers from llama.cpp (`-v`), context 8192, same five
configurations:

| Config | model buffer | KV cache | compute |
|---|---|---|---|
| full, K=8 | 17 524 MiB | 768 MiB | 92 MiB |
| full, K=6 | 17 524 MiB | 768 MiB | 75 MiB |
| REAP 25%, K=8 | 13 327 MiB | 768 MiB | 92 MiB |
| REAP 25%, K=6 | 13 327 MiB | 768 MiB | 75 MiB |
| REAP 50%, K=8 | 9 130 MiB | 768 MiB | 92 MiB |

Three facts, cleanly separated:

- **Reducing K saves essentially no memory.** All experts stay resident
  whether 8 or 6 run per token — only the activation workspace shrinks
  (92 → 75 MiB). K is a *compute* knob.
- **REAP is the memory knob, and it is exactly linear**: −4 197 MiB at
  25%, −8 394 MiB at 50% — matching `poe plan`'s exact byte accounting
  to the mebibyte, now confirmed by a third independent party
  (llama.cpp's allocator).
- **KV cache belongs to context, not experts**: 768 MiB at 8k either
  way. On small devices the budget line is
  `model buffer + KV(context you want)` — REAP frees room for either.

## Second architecture: gpt-oss-20b (MXFP4)

The same recipe ran unchanged on gpt-oss-20b (24 MoE blocks, 32 experts
top-4, MXFP4 experts, per-expert gate/up/down *biases* and a router
bias — 8 sliceable tensors per block vs Qwen3's 4):

| Variant | Size | Coding eval | Gen t/s |
|---|---|---|---|
| full MXFP4 | 11.3 GiB | 10/10 | 84.7 |
| REAP 25% | 8.9 GiB (78.9%) | 9/10 | 83.4 |

The one failure (Dijkstra) was a "no code block" — the pruned model's
reasoning ran past the token budget rather than producing wrong code,
so treat it as a soft signal. Structural validation (`poe validate
--plan`), provenance, exact byte accounting and coherent generation all
held on the second architecture, biases included.

## 6. Lessons learned

- **Expert pruning buys memory, not tokens/second.** Top-k is untouched,
  so active compute per token is identical; measured throughput was
  within noise (±3%). The value is a 23.7% smaller checkpoint and
  residency footprint — the difference between fitting and not fitting
  on a smaller device, or more room for KV cache.
- **Exact accounting catches real bugs.** Because plan/estimate/apply all
  recompute byte-exact numbers from the tensor table independently, a
  disagreement anywhere aborts before a byte is written. This caught a
  real inconsistency during development (router bias rows counted by one
  code path and not the other).
- **Router bias matters on some architectures.** Qwen3 has none, but
  DeepSeek-style `exp_probs_b` and gpt-oss per-expert biases exist; the
  expert dimension is always the last (slowest) dimension of every
  routed tensor, which makes slicing uniform: one rule covers weights,
  biases and router rows.
- **Beware the llama-cli REPL when smoke-testing.** Recent llama.cpp
  `llama-cli` drops into a chat REPL; with stdin closed it loops forever
  printing its prompt (at ~4M writes/s, on one core, GPU idle — easily
  mistaken for a hang or a broken model). Smoke-test with
  `-st --simple-io < /dev/null`.
- **Keep artifacts.** `.poeprofile` and `.poeplan` are tiny, diffable and
  carry fingerprints; with them, any pruned GGUF can be audited
  (`poe validate --plan`) or reproduced byte-identically later.
