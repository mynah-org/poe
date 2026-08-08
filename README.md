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
| `poe routing-budget` | ✅ exact active-parameter accounting per routed-K |
| `poe profile` / `compare` / `plan` / `estimate` / `apply` / `forge` / `diff` / `validate` | planned — see [vision.md](vision.md) |

## Build

C11, no dependencies. The [ingot](https://github.com/mynah-org/ingot)
amalgam is vendored in `third_party/ingot/`.

```sh
make          # builds ./poe
make test     # synthetic-fixture tests, no model downloads
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

Static commands (`inspect`, `experts`, `routing-budget`, `estimate`, `plan`,
`diff`) never run inference and never require a GPU.

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
