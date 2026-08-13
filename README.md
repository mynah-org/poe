# POE

Pruning & Optimization of Experts.

POE is a lightweight toolkit for inspecting, profiling, specializing and
compressing Mixture-of-Experts language models.

It can discover which experts matter for a workload, build task-specific
expert profiles, estimate disk/RAM/VRAM savings, and create smaller MoE
checkpoints using training-free compression methods such as REAP.

GGUF-first. CUDA-accelerated. Ingot-backed.

*Poe, the reaper of MoE.*

## Status

Early development. Currently implemented (milestone 1 of the roadmap in
[vision.md](vision.md)):

| Command | Status |
|---|---|
| `poe inspect` | ✅ static MoE structure, storage accounting, theoretical reductions |
| `poe experts` | ✅ expert tensor mapping and sizes |
| `poe routing-budget` | ✅ exact active-parameter accounting per routed-K; with `--profile`, the per-token distribution of how many experts the router's mass actually needs (and how much mass the applied top-k holds) |
| `poe-profile` (standalone) | ✅ router observer over llama.cpp: routing + gate statistics, entropy, cumulative-mass-K; `--metric reap` adds streaming REAP saliency (gate × expert-output norm); `--metric imatrix` adds per-expert activation statistics, written as llama.cpp's GGUF imatrix (`--imatrix-all` covers the dense path too: 511 entries on Qwen3.6, against the 510 the checkpoint's own metadata records); `--until-stable` stops calibration when the prune decision converges |
| `poe compare` | ✅ expert-fingerprint overlap between workload profiles: Jaccard / weighted Jaccard / Spearman / JS divergence, exclusive and cold experts, REAP prune-set agreement |
| `poe plan` | ✅ deterministic expert-removal plans from weighted profiles (`reap` / `frequency` / `gate`), exact byte accounting, conservative guards ([format](docs/poeplan.md)) |
| `poe estimate` | ✅ plan accounting + fingerprint/byte verification against the source model |
| `poe diff` | ✅ plans (prune-set agreement), profiles, bit maps (`.poequant`: type map, byte delta, rank correlation of the layer scores), and structural model diffs |
| `poe apply` | ✅ structural GGUF pruning: slices packed expert tensors along the expert dimension, compacts router rows in keep order, patches `expert_count`, preserves all other metadata byte-for-byte; streams mmap → output, then reopens and verifies exact accounting |
| `poe forge` | ✅ the whole pipeline in one command: (profile via `poe-profile` →) plan → exact estimate → apply → verify, intermediate artifacts preserved next to the output |
| `tools/coding_eval.py` | ✅ pruned-vs-full behavioral eval: 10 coding tasks (easy→hard) generated at temp 0 through llama-server, compiled/executed against hidden test cases; win/fail per task + prompt/generation t/s |
| `poe quantplan` | ✅ per-slab mixed precision: spend a byte budget across the routed expert slabs (Q2_K..Q6_K ladder, greedy by measured error per byte), exact accounting, emits llama-quantize's `--tensor-type-file`; ranks layers by REAP saliency (`--profile`) or by an imatrix statistic (`--imatrix`), and reports how far the ranking is from being a plain depth ramp |
| `poe requant` | ✅ per-expert precision emulated inside one carrier type: cold experts pushed through a hard quantizer and back, the slab stored at the carrier, so two arms at the same average bits are byte-identical in size — the M9b question measured without a runtime patch ([why](docs/mixed-precision.md)) |
| `poe split` | ✅ per-expert precision in one checkpoint: experts reordered by need, each slab stored as a hot tensor and a cold tensor at different types, router rows permuted to match so a runtime only tests `id < poe.split.hot_count` ([patch](tools/patches/README.md), [result](docs/per-expert-precision.md)) |
| `poe plan --protect-super-experts` | ✅ activation-magnitude outliers (robust z over `actnorm_mean`) are lifted out of the cut and replaced by the next candidates, so the size target and the byte accounting are unchanged; on by default ([doc](docs/super-experts.md)) |
| `poe residency` | ✅ what fits in a VRAM budget and the llama.cpp flags that place the rest on the host: exact slab bytes, labelled KV/compute estimates, and hot/cold placement on a split checkpoint ([doc](docs/residency.md)) |
| `poe validate` | ✅ structural checks (shapes vs metadata, slab divisibility, quant geometry); with `--plan`, source/pruned provenance and exact byte accounting |

Recipe + field notes: [docs/reap-coding-recipe.md](docs/reap-coding-recipe.md).
Routing-budget measurements (what lowering top-k costs, and why per-layer
schedules do not pay): [docs/routing-experiments.md](docs/routing-experiments.md).
Mixed precision per expert slab, and why uniform allocation won:
[docs/mixed-precision.md](docs/mixed-precision.md).
Quantization versus expert deletion at equal bytes — a coin flip in the
calibration domain, 7× worse outside it:
[docs/quant-vs-prune.md](docs/quant-vs-prune.md).
Per-expert precision, measured before building the runtime for it — aiming
the damage at cold experts wins 2.4× in the calibration domain:
[docs/per-expert-precision.md](docs/per-expert-precision.md).

## Build

C11, no dependencies. The [ingot](https://github.com/mynah-org/ingot)
amalgam is vendored in `third_party/ingot/`.

```sh
make          # builds ./poe
make test     # synthetic-fixture tests, no model downloads
make profiler LLAMA_DIR=~/llama.cpp   # optional: poe-profile (needs llama.cpp built with shared libs)
```

## Usage

```sh
$ poe inspect model.gguf

MoE architecture
  blocks                48   (48 MoE)
  experts/block         128
  active/token          8

Storage
  experts               25.7 GiB
  other                  4.3 GiB
  total                 30.0 GiB
```

```sh
poe experts model.gguf --layer 17     # expert tensor mapping for one block
poe routing-budget model.gguf         # ActiveParams(K) table, exact, per routed-K
poe inspect model.gguf --json         # machine-readable output
```

```sh
# materialize a plan: 17.3 GiB -> 13.2 GiB, no training, exact bytes
poe apply coding-reap25.poeplan model.gguf -o model-pruned.gguf

# or the whole pipeline in one command (profile -> plan -> apply -> verify)
poe forge model.gguf --dataset code.txt -o model-coding.gguf

# behavioral check: did pruning cost anything on real coding tasks?
python3 tools/coding_eval.py model.gguf model-coding.gguf
```

Static commands (`inspect`, `experts`, `routing-budget`, `estimate`, `plan`,
`diff`, `apply`) never run inference and never require a GPU.

## Design

- **ingot owns the container**: GGUF read/write, mmap, quantized tensor
  handling, deterministic rewrites.
- **POE owns the MoE logic**: architecture discovery, router instrumentation,
  workload profiling, expert saliency (REAP first), pruning/merging plans,
  hardware-aware planning.
- **Training-free by default**: profile → plan → apply, no gradients.
- **Three optimization axes**, independent and combinable: what exists
  (pruning/merging/quantization), what executes (adaptive routing and
  expert skipping), where it lives (GPU/CPU expert residency).
- **Reproducible artifacts**: profiles (`.poeprofile`) and plans (`.poeplan`)
  are persistent, versioned, and bound to model fingerprints.

See [vision.md](vision.md) for the full design and milestone roadmap.
Algorithm notes: [docs/reap-notes.md](docs/reap-notes.md).

## License

MIT
