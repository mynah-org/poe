# POE — Pruning & Optimization of Experts

> **Poe, the reaper of MoE.**
>
> Lightweight CLI and library for analyzing, profiling, specializing, compressing, and forking Mixture-of-Experts language models.

Repository target: `github.com/mynah-org/poe`  
CLI executable: `poe`  
Primary checkpoint backend: **Ingot**  
Primary deployment format: **GGUF**  
Compute strategy: **CUDA-first for dynamic profiling, CPU-capable by design**  
Training philosophy: **training-free by default**

---

## 1. Vision

POE should answer practical questions about Mixture-of-Experts models:

- How many experts does this model contain?
- How many experts are active per token and per layer?
- Which experts are selected for coding, math, tool-use, reasoning, or a custom workload?
- Are the same experts consistently selected for a task class?
- Which experts are important, redundant, cold, or task-specific?
- How stable is an expert ranking as calibration samples increase?
- Which experts can be removed while respecting a quality-risk budget?
- Would pruning, merging, quantization, offload, or a hybrid strategy work best?
- How much disk, RAM, and VRAM can potentially be saved?
- Can a general-purpose MoE be forked into a smaller coding-specialized model?
- Can the optimized model be emitted directly as a valid GGUF?
- Can this be done without retraining?

The intended mental model is:

```text
MODEL
  +
WORKLOAD
  +
HARDWARE
  +
CONSTRAINTS
  ↓
ANALYZE / PROFILE
  ↓
EXPERT IMPORTANCE + REDUNDANCY MODEL
  ↓
OPTIMIZATION PLAN
  ↓
SPECIALIZED / COMPRESSED MODEL
```

POE is **not** intended to become another full inference framework.

---

## 2. Project boundaries

### Ingot

Ingot should remain responsible for generic checkpoint/container primitives:

- GGUF reading/writing
- safe tensor access
- mmap
- tensor metadata
- tensor slicing/copying
- quantized tensor handling
- deterministic checkpoint rewrites
- size accounting
- low-level CPU kernels where appropriate

POE should depend on Ingot rather than duplicate those responsibilities.

### POE

POE owns MoE-specific logic:

- MoE architecture discovery
- expert identification
- router instrumentation
- workload profiling
- task/expert fingerprints
- expert saliency
- pruning
- merging/recombination
- expert routing analysis
- VRAM/disk/RAM planning
- expert residency planning
- model specialization
- validation
- generation of reproducible transformation plans

---

## 3. Design principles

### 3.1 Training-free by default

The primary POE workflow must not require model retraining.

Modes:

```text
LEVEL 0 — static inspection
no forward pass
no training

LEVEL 1 — calibration/profile
forward-only
no gradients
no training

LEVEL 2 — structural transformation
checkpoint rewrite
no training

LEVEL 3 — optional router recalibration
small optimization step
NOT required for standard POE

LEVEL 4 — optional recovery fine-tuning
external/optional
outside initial scope
```

A user should be able to request:

```bash
poe forge model.gguf \
  --task coding \
  --target-vram 24G \
  --no-train
```

and POE must constrain its optimizer accordingly.

---

## 4. Compute strategy

### CUDA-first, not CUDA-required

Dynamic MoE profiling involves model forward passes and can be expensive.

Priority:

1. NVIDIA CUDA backend
2. CPU backend
3. optional Metal backend
4. optional ROCm backend

However:

```text
poe inspect
poe estimate
poe plan
poe apply
poe diff
```

should not require CUDA.

CUDA mainly accelerates:

```text
poe profile
poe reap/profile
activation observers
ablation profiling
benchmark/evaluation helpers
```

### Avoid making POE an inference engine

Initial preferred architecture:

```text
              POE
        ┌──────┴──────┐
        │             │
      Ingot       compute backend
        │             │
        │        CUDA / llama.cpp /
        │        minimal kernels
        │             │
        └──────┬──────┘
               │
          MoE observers
               │
        importance metrics
               │
       optimizer / planner
               │
            Ingot
               │
          output GGUF
```

Investigate whether llama.cpp can expose enough router/expert information without invasive patches.

Do **not** rewrite an entire transformer runtime merely to obtain expert statistics unless later benchmarks justify it.

---

## 5. Supported formats

### MVP

- GGUF
- models already supported by the chosen inference backend

### Later

- safetensors / Hugging Face checkpoints
- conversion via Ingot where possible
- FP8 / BF16 / FP16
- quantized GGUF types

GGUF should be a first-class format, not merely an export target.

A major differentiator should be profiling the **actual deployment artifact**, including quantized GGUF models.

---

## 6. Core concepts

### Expert

Canonical identifier:

```text
layer:<L>/expert:<E>
```

Example:

```text
layer:17/expert:42
```

### Profile

A reusable statistical observation of a model under a workload:

```text
model.poeprofile
```

Contains:

- model fingerprint/hash
- architecture
- dataset fingerprint
- token count
- routing statistics
- gate statistics
- activation statistics when available
- REAP saliency
- convergence history
- backend/device information
- timing
- optional benchmark results

### Plan

A deterministic proposed transformation:

```text
coding.poeplan
```

Contains:

- source model fingerprint
- workload/profile fingerprint
- selected algorithm
- experts retained/pruned/merged per layer
- quantization policy
- residency policy
- expected disk/RAM/VRAM
- expected risk/confidence
- transformation parameters

The expensive profiling step should be reusable by many plans.

---

## 7. CLI proposal

### Inspect

```bash
poe inspect model.gguf
```

Output:

```text
Architecture
------------
MoE layers
experts per layer
active experts/token
shared experts
router type
expert dimensions
quantization

Storage
-------
expert bytes
non-expert bytes
router bytes
metadata
total size

Theoretical reductions
----------------------
25% expert removal
50% expert removal
75% expert removal
```

No model inference.

---

### Expert listing

```bash
poe experts model.gguf
poe experts model.gguf --layer 17
```

Show expert tensor mappings and sizes.

---

### Profile a workload

```bash
poe profile model.gguf \
  --dataset coding.jsonl \
  --device cuda
```

Presets may later exist:

```bash
poe profile model.gguf --task coding
poe profile model.gguf --task math
poe profile model.gguf --task tool-use
poe profile model.gguf --task general
```

Presets must be transparent about the calibration data used.

Custom datasets are more important than presets.

---

### Profiling metric levels

```bash
--metric routing
--metric gate
--metric reap
--metric ablation
```

Approximate cost:

```text
routing     cheap
gate        cheap
REAP        moderate
ablation    expensive
```

The CLI should explicitly communicate this.

---

### Compare workloads

```bash
poe compare \
  coding.poeprofile \
  math.poeprofile \
  general.poeprofile
```

Output examples:

- expert overlap
- weighted Jaccard similarity
- rank correlation
- task-exclusive experts
- shared experts
- per-layer specialization
- entropy
- routing concentration
- stable/cold experts

---

### Plan

```bash
poe plan model.gguf \
  --profile coding.poeprofile \
  --method reap \
  --prune 40%
```

Or constraint-based:

```bash
poe plan model.gguf \
  --profile coding.poeprofile \
  --target-vram 24G \
  --max-quality-loss 2%
```

No rewriting yet.

---

### Estimate

```bash
poe estimate coding.poeplan
```

Estimate:

- resulting parameters
- GGUF disk size
- CPU RAM
- GPU VRAM
- expert working set
- optional KV cache assumptions
- expected pruning risk
- confidence

Clearly distinguish exact accounting from estimates.

---

### Apply

```bash
poe apply model.gguf coding.poeplan \
  -o model-coding.gguf
```

Responsibilities:

- validate source fingerprint
- structurally rewrite expert tensors
- update architecture metadata
- update expert counts/topology
- preserve unrelated metadata
- validate output
- optionally run smoke test

---

### Forge

High-level convenience command:

```bash
poe forge model.gguf \
  --dataset coding.jsonl \
  --target-vram 24G \
  --method auto \
  -o model-coding.gguf
```

Equivalent to:

```text
inspect
→ profile
→ plan
→ estimate
→ apply
→ validate
```

But individual stages must remain accessible.

---

### Diff

```bash
poe diff model.gguf model-coding.gguf
poe diff coding.poeplan math.poeplan
poe diff coding.poeprofile math.poeprofile
```

---

## 8. REAP — primary MVP compression algorithm

**REAP: Router-weighted Expert Activation Pruning**

Use REAP as the first serious dynamic pruning backend.

Reason:

- retraining-free
- one-shot
- based on actual routed contribution
- demonstrated on multiple modern MoE architectures
- particularly promising on generative/code/tool workloads
- conceptually straightforward enough for independent implementation
- natural match for POE's workload-aware philosophy

Conceptually accumulate:

```text
importance(e)
≈ mean over routed tokens (
    normalized_router_weight(e, token)
    ×
    norm(expert_output(e, token))
)
```

Implementation details should follow the current REAP reference/paper carefully rather than relying on this simplified expression.

Important:

- normalize top-k router weights consistently
- support layer-wise accumulation
- avoid retaining all activations
- use streaming accumulators
- support memory-efficient layer-wise observation
- store enough statistics to regenerate plans without recalibration

REAP should initially be used for **ranking**, while the structural rewrite remains a separate subsystem.

---

## 9. REAP convergence / fast calibration

This could become a major POE innovation.

Do not require users to guess a calibration sample count.

Track expert-ranking stability while profiling.

Every N tokens/windows compute:

- Spearman rank correlation
- Kendall rank correlation if cheap enough
- Jaccard overlap of bottom-X%
- Jaccard overlap of top-X%
- per-layer stability
- global stability
- confidence intervals/bootstrap where practical

Example:

```text
tokens     bottom-40% stability
--------------------------------
1,024      72.1%
2,048      86.4%
4,096      94.7%
8,192      98.3%
12,288     99.1%
```

Possible UX:

```bash
poe profile model.gguf \
  --dataset coding.jsonl \
  --metric reap \
  --until-stable 0.99
```

or:

```bash
poe profile model.gguf \
  --dataset coding.jsonl \
  --max-tokens 50000 \
  --confidence 0.99
```

Stop early when the pruning decision is stable enough.

This must be empirically validated.

---

## 10. Baseline importance metrics

Before implementing many research algorithms, support inexpensive baselines.

### Router frequency

```text
count expert selection
```

Useful diagnostic baseline.

### Mean router probability / gate weight

```text
mean selected gate contribution
```

### Activation norm

Observe expert output magnitude independently of gate weight.

### REAP

Router-weighted activation contribution.

### Ablation score

Measure change in loss/quality after suppressing an expert.

Very expensive, but useful as a ground-truth-like research metric on small models.

These should share one metric interface.

---

## 11. DERN — phase 2

**DERN: Dropping Experts, Recombining Neurons**

Properties:

- retraining-free
- first identifies redundant experts using router statistics
- then recombines neuron-level segments from removed experts into retained experts
- intended to preserve information lost by simple expert deletion

Why implement:

- gives POE a strategy fundamentally different from pure pruning
- useful benchmark against REAP
- potentially strong where expert removal loses important but distributed information

Implementation complexity is significantly higher than REAP.

Do not block MVP on DERN.

Suggested API:

```bash
poe plan model.gguf \
  --profile profile.poeprofile \
  --method dern \
  --compression 50%
```

---

## 12. REAM — phase 2/3

**REAM: Router-weighted Expert Activation Merging**

REAM adapts router-weighted expert activation ideas to expert merging.

Include because:

- useful comparison with pruning
- may preserve different capabilities
- calibration mixture appears to affect the quality Pareto frontier
- strengthens POE as an optimizer rather than a REAP wrapper

Possible method:

```bash
poe plan model.gguf \
  --method ream \
  --compression 50%
```

Keep separate from REAP even if they reuse profile statistics.

---

## 13. C-Prune — experimental

Cluster-driven expert pruning.

Useful concepts:

- intra-layer expert similarity
- expert redundancy clustering
- layer-dependent homogeneity
- task-specific pruning

Potential POE contribution:

```text
static weight similarity
+
dynamic workload saliency
```

Could allow fast pre-clustering from weights followed by workload-aware pruning.

Not MVP.

---

## 14. COMPEL — experimental

COMPEL introduces:

- layer-adaptive pruning
- Fisher-information-based expert importance
- layer importance
- expert weight compensation

Important design lesson even without immediately implementing the exact algorithm:

> POE must not assume every MoE layer should lose the same percentage of experts.

Support:

```text
global pruning budget
↓
layer-sensitive allocation
↓
per-layer expert decisions
```

The optimizer should eventually compare uniform pruning vs layer-adaptive pruning.

---

## 15. Algorithm plugin architecture

Do not hardcode REAP into the project design.

Suggested abstraction:

```text
Metric
  observe(...)
  finalize(...)
  score(layer, expert)

Transform
  build_plan(profile, constraints)
  validate(plan)
  apply(...)

Optimizer
  search(model, profile, hardware, constraints)
```

Candidate tree:

```text
src/
  metrics/
    frequency.*
    gate.*
    activation.*
    reap.*
    ablation.*

  transforms/
    prune.*
    dern.*
    ream.*
    cprune.*
    compel.*

  optimizer/
    budget.*
    pareto.*
    residency.*
    hardware.*

  backends/
    cuda.*
    cpu.*
    llamacpp.*

  formats/
    gguf_ingot.*

  models/
    qwen.*
    mixtral.*
    deepseek.*
    gptoss.*
```

Exact language/layout should follow the existing mynah-org conventions and Ingot API.

---

## 16. Task/expert fingerprints

One of POE's most interesting capabilities should be answering:

> Are the same experts used for coding?

For each workload calculate:

- expert selection frequency
- weighted routing contribution
- REAP saliency
- activation contribution
- expert rank
- expert entropy
- per-layer distribution

Compare workloads via:

- Jaccard
- weighted Jaccard
- cosine similarity
- Spearman rank correlation
- Jensen-Shannon divergence where appropriate
- mutual information between task labels and expert selection
- per-layer specialization score

Example output:

```text
Task overlap
-------------------------
coding ↔ math        81.2%
coding ↔ tool-use    88.4%
coding ↔ general     54.7%
```

Also identify:

```text
coding-critical
coding-biased
general-purpose
cold
high-frequency/low-impact
low-frequency/high-impact
redundant
```

Avoid claiming semantic expert roles without statistical evidence.

---

## 17. Workload-aware specialization

Primary use case:

```bash
poe forge model.gguf \
  --dataset coding.jsonl \
  --target-size 20G \
  -o model-code.gguf
```

POE should be able to produce different variants from one source profile family:

```text
base
 ├── coding
 ├── math
 ├── tool-use
 ├── coding+math
 └── constrained-hardware
```

Multiple workload profiles may be combined with weights:

```bash
poe plan model.gguf \
  --profile coding.poeprofile:0.7 \
  --profile tool.poeprofile:0.3 \
  --target-vram 24G
```

---

## 18. VRAM-aware expert residency

Pruning is not always the correct answer.

POE should eventually distinguish:

```text
DELETE expert
MERGE expert
QUANTIZE expert
KEEP expert on GPU
OFFLOAD expert to CPU
CACHE expert dynamically
```

For a known workload derive a hot set.

Example:

```text
GPU hot set        31 experts
CPU warm set       54 experts
cold set           43 experts

expected GPU hit   91.7%
```

Potential command:

```bash
poe residency model.gguf \
  --profile coding.poeprofile \
  --vram 24G
```

This produces a deployment plan without modifying the model.

Longer-term optimizer objective:

```text
minimize:
  disk
  VRAM
  RAM
  PCIe traffic
  latency

subject to:
  quality loss <= X
  hardware constraints
```

---

## 19. Quantization interaction

Quantization should be modeled independently from expert pruning.

Potential transformations:

```text
prune only
quantize only
prune + quantize
merge + quantize
residency + quantize
```

Important experiment:

```bash
poe compare-profile model-f16.gguf model-q4.gguf --task coding
```

Research question:

> Does quantization materially change expert routing or expert importance?

Do not assume FP16/BF16 profiles always transfer perfectly to Q4/Q5/Q8 deployment models.

---

## 20. Size / RAM / VRAM accounting

Provide exact accounting whenever possible.

Separate:

### Exact

- checkpoint bytes
- expert tensor bytes
- non-expert tensor bytes
- parameters retained
- parameters removed

### Estimated

- runtime VRAM
- runtime RAM
- KV cache
- backend workspace
- transfer bandwidth
- expected latency

Never present estimates as measured values.

Output should explain assumptions.

---

## 21. Planner / optimizer

Long-term high-level interface:

```bash
poe optimize model.gguf \
  --profile coding.poeprofile \
  --gpu-vram 24G \
  --ram 64G \
  --disk 30G \
  --max-quality-loss 2% \
  --no-train
```

Candidate strategies:

```text
REAP-20
REAP-30
REAP-40
REAP-50
DERN-*
REAM-*
Q8
Q6
Q5
Q4
REAP + Q*
DERN + Q*
CPU offload
GPU expert cache
hybrid
```

Return Pareto candidates instead of pretending there is always one objectively optimal plan.

Example:

```text
PLAN          DISK    VRAM   QUALITY-RISK
-----------------------------------------
REAP-25       31G     29G    low
REAP-40+Q8    24G     23G    low/medium
REAP-50+Q6    18G     17G    medium
```

Quality-risk predictions must initially be clearly marked heuristic until calibrated against real benchmark data.

---

## 22. Validation

A structurally valid GGUF is not enough.

Validation levels:

```text
poe validate model.gguf
```

### Structural

- metadata consistency
- expert count
- tensor dimensions
- tensor offsets
- quantization metadata
- checksums where relevant

### Runtime smoke test

- load model
- tokenize prompt
- run short generation
- ensure no invalid expert/router indices
- finite outputs

### Behavioral

Optional benchmark set:

- perplexity where useful
- code benchmark subset
- math subset
- tool-use subset
- custom evaluator hooks

---

## 23. Safety against bad pruning

POE should be conservative by default.

Warn when:

- calibration dataset is too small
- expert ranking is unstable
- task coverage is narrow
- requested pruning ratio exceeds validated range
- some layers show little redundancy
- important experts are rare but high-impact
- a profile was generated from a different quantization/model revision
- source model hash differs from the plan

Require `--force` for clearly dangerous transformations.

---

## 24. Reproducibility

Every profile/plan should contain:

```text
POE version
Ingot version
model hash
model metadata
dataset hash
sampling seed
calibration token count
algorithm version
backend
device
precision
metric options
timestamp
```

A transformation should be reproducible from:

```text
source model
+
profile
+
plan
+
POE version
```

---

## 25. Performance goals

### Static commands

`inspect`, `experts`, `estimate`, `plan`, and `diff` should feel effectively immediate on normal local hardware, bounded mostly by metadata parsing or intentional tensor scans.

### Apply

`apply` should trend toward storage/memory bandwidth limits.

Avoid materializing the whole model in RAM.

Prefer:

```text
mmap source
stream/copy retained tensors
rewrite transformed tensors
sequential output
```

### Profile

Primary optimization target.

Use:

- CUDA
- batching
- streaming accumulators
- no unnecessary activation retention
- layer-wise observation
- early convergence
- optional token budget
- optional CPU fallback

Benchmark **tokens-to-ranking-convergence**, not merely tokens/sec.

---

## 26. Benchmark suite

Create a repeatable benchmark harness early.

Models should include at least:

- one small MoE suitable for CI/dev
- Mixtral-family model
- Qwen MoE
- GPT-OSS or equivalent supported architecture

For each:

```text
model
format
precision
backend
GPU
CPU
RAM
calibration tokens
wall time
peak VRAM
peak RAM
ranking stability
rewrite time
output size
quality metrics
```

Key experiment:

### Expert ranking convergence

For:

```text
1K
2K
5K
10K
25K
50K
100K
```

calibration tokens, compare the selected prune set against a large reference profile.

This determines whether POE can offer a genuinely fast interactive mode.

---

## 27. MVP milestones

### Milestone 0 — repository/bootstrap

- create `mynah-org/poe`
- CLI skeleton
- link/use Ingot
- logging
- error handling
- test harness
- model fingerprint abstraction

Deliverable:

```bash
poe --help
```

---

### Milestone 1 — static GGUF MoE inspector

Support one architecture first.

Implement:

```bash
poe inspect
poe experts
poe estimate
```

Detect:

- MoE layers
- experts/layer
- top-k
- tensor mapping
- expert storage size

No CUDA.

Success criterion:

POE correctly explains the MoE structure of known GGUF files.

---

### Milestone 2 — router profiler

Instrument a working inference backend.

Implement:

```bash
poe profile --metric routing
poe profile --metric gate
```

Record:

- selected expert IDs
- gate values
- layer
- token
- aggregate statistics

CUDA first.

Success criterion:

routing distributions are reproducible and can be compared across workloads.

---

### Milestone 3 — task/expert analysis

Implement:

```bash
poe compare
```

Metrics:

- frequency
- overlap
- rank correlation
- task specialization
- entropy
- hot/cold experts

Success criterion:

answer empirically whether coding/math/general workloads exhibit distinguishable expert fingerprints.

---

### Milestone 4 — REAP observer

Implement REAP-compatible saliency.

Requirements:

- streaming aggregation
- normalized router weights
- activation norms
- layer-wise mode
- CUDA implementation
- CPU fallback if feasible

Compare output rankings against reference REAP on supported models.

Success criterion:

expert rankings closely match reference implementation.

---

### Milestone 5 — convergence-driven profiling

Implement:

```bash
--until-stable
--confidence
--max-tokens
```

Measure ranking stability continuously.

Success criterion:

determine a reliable token budget for representative models/workloads.

---

### Milestone 6 — POE plan format

Implement:

```bash
poe plan
poe estimate
poe diff
```

No checkpoint mutation yet.

Success criterion:

profiles generate deterministic expert-removal plans and accurate resulting-size estimates.

---

### Milestone 7 — structural GGUF pruning

Use Ingot to:

- omit removed expert tensors
- compact/remap expert indices
- update model metadata
- emit valid GGUF

Implement:

```bash
poe apply
```

Success criterion:

output GGUF loads and runs in target inference runtime.

---

### Milestone 8 — forge

Combine pipeline:

```bash
poe forge
```

Success criterion:

one command turns a source GGUF + coding dataset into a smaller, valid coding-profiled variant without training.

This is the first strong public release candidate.

---

## 28. Post-MVP milestones

### v0.2

- multiple model architectures
- richer GGUF quant support
- multi-workload profiles
- profile diff
- CUDA optimization
- benchmark harness

### v0.3

- DERN
- REAM
- layer-adaptive pruning
- C-Prune experiments
- COMPEL-inspired allocation

### v0.4

- quantization planner
- expert GPU residency planning
- CPU/GPU offload planning
- hardware-aware optimizer
- Pareto search

### v1.0 candidate

```bash
poe optimize model.gguf \
  --task coding \
  --gpu-vram 24G \
  --max-quality-loss 2% \
  --no-train
```

produces a reproducible deployment recommendation and can materialize the resulting model.

---

## 29. First supported model

Choose the first development model based on:

- small enough for rapid iteration
- standard sparse MoE topology
- GGUF support
- inference backend support
- available reference checkpoint
- compatible with reference REAP implementation if possible
- manageable calibration time on one consumer NVIDIA GPU

Do **not** start with a 100B+ model.

Goal is algorithm correctness and architecture design first.

---

## 30. Implementation order for Claude Code

Claude Code should **not implement the entire roadmap at once**.

First task:

1. inspect the Ingot public API and current repository conventions
2. inspect how target GGUF MoE tensors are named
3. define a minimal internal MoE model descriptor
4. implement read-only `poe inspect`
5. implement tests against one known GGUF
6. only then design the profiling backend

Suggested internal descriptor:

```c
struct poe_expert {
    uint32_t layer;
    uint32_t index;
    // tensor references: gate/up/down/etc.
};

struct poe_moe_layer {
    uint32_t layer;
    uint32_t expert_count;
    uint32_t experts_per_token;
    // router tensor reference
    // expert collection
};

struct poe_model {
    // model fingerprint
    // architecture
    // format
    // layers
};
```

Adapt names/types to Ingot conventions rather than forcing this exact API.

---

## 31. Non-goals for MVP

Avoid initially implementing:

- full training
- LoRA
- router fine-tuning
- distributed inference
- model serving
- GUI
- arbitrary Hugging Face architecture support
- every published pruning algorithm
- automatic quality-loss prediction presented as fact
- custom transformer runtime unless unavoidable

The MVP wins if it does a small set of things exceptionally well:

```text
inspect
profile
understand expert specialization
REAP rank
plan
prune
write valid GGUF
```

---

## 32. Research questions POE should expose

POE is useful not only as a compressor but as a MoE research instrument.

Questions worth making easy to answer:

1. How task-specific are experts in modern MoEs?
2. Is specialization stable across prompts within the same task?
3. Which layers specialize most?
4. How many calibration tokens are required for stable expert ranking?
5. Does quantization alter routing significantly?
6. Are expert frequency and expert contribution strongly correlated?
7. How often are rare experts actually high-impact?
8. Is uniform layer pruning suboptimal?
9. Does a coding-derived prune set generalize to tool-use?
10. How much does task-conditioned REAP outperform general calibration for specialized forks?
11. At equal size, when does DERN/REAM outperform deletion?
12. Can a small hot expert set cover most of a workload sufficiently well for residency/offload?
13. How stable are expert fingerprints across model revisions?
14. Can static weight similarity predict dynamic redundancy?
15. What is the Pareto frontier of disk vs VRAM vs latency vs quality?

---

## 33. Potential differentiators

POE should aim to differentiate itself through the combination of:

### GGUF-native

Analyze and transform local deployment artifacts directly.

### Ingot-backed

Fast, safe, minimal-dependency checkpoint handling.

### Workload-aware

Optimize for *my coding workload*, not generic benchmark averages only.

### Training-free first

Useful without expensive fine-tuning.

### Convergence-aware

Stop calibration based on ranking stability rather than arbitrary sample counts.

### Hardware-aware

Optimize not only parameter count but actual RAM/VRAM/residency constraints.

### Algorithm-agnostic

REAP is the first backend, not the product identity.

### Reproducible

Profiles and plans are persistent artifacts.

---

## 34. Suggested README positioning

```text
# POE

Pruning & Optimization of Experts.

POE is a lightweight toolkit for inspecting, profiling, specializing and
compressing Mixture-of-Experts language models.

It can discover which experts matter for a workload, build task-specific
expert profiles, estimate disk/RAM/VRAM savings, and create smaller MoE
checkpoints using training-free compression methods such as REAP.

GGUF-first. CUDA-accelerated. Ingot-backed.

Poe, the reaper of MoE.
```

---

## 35. Initial example UX

```bash
$ poe inspect Qwen-MoE.gguf

MoE architecture
  layers                48
  experts/layer         128
  active/token          8

Storage
  experts               25.7 GiB
  other                  4.3 GiB
  total                 30.0 GiB
```

```bash
$ poe profile Qwen-MoE.gguf \
    --dataset ./coding.jsonl \
    --metric reap \
    --device cuda \
    --until-stable 0.99

Calibration
  tokens                 12,288
  ranking stability       99.2%
  profile                 qwen-coding.poeprofile

Expert concentration
  top 25% contribution    71.4%
  bottom 25% contribution  3.1%
```

```bash
$ poe plan Qwen-MoE.gguf \
    --profile qwen-coding.poeprofile \
    --target-vram 20G \
    --method reap

Candidate plan
  experts removed         37.5%
  estimated model size    ...
  estimated VRAM          ...
  confidence              ...
```

```bash
$ poe apply Qwen-MoE.gguf \
    qwen-coding.poeplan \
    -o Qwen-MoE-Code.gguf
```

---

## 36. Algorithm priority

Recommended implementation order:

```text
1. routing frequency
2. gate-weight statistics
3. REAP
4. ablation validator for small models
5. layer-adaptive budget allocation
6. DERN
7. REAM
8. C-Prune
9. COMPEL / Fisher-based approaches
10. hardware-aware hybrid optimizer
```

Reasoning:

- simple metrics validate instrumentation
- REAP provides the strongest practical early compression baseline
- ablation provides a sanity-check reference
- layer-adaptive allocation can improve many methods
- DERN and REAM broaden the transformation space
- more complex research algorithms should follow after the profiler and transformation pipeline are stable

---

## 37. References / algorithms to study

### REAP

CerebrasResearch implementation:

https://github.com/CerebrasResearch/reap

Paper:

https://arxiv.org/abs/2510.13999

Key concepts:

- Router-weighted Expert Activation Pruning
- one-shot/retraining-free expert pruning
- generative/code/tool-use evaluation
- normalized top-k router weights
- memory-efficient layer-wise calibration

### DERN

Paper:

https://arxiv.org/abs/2509.10377

ACL/EMNLP publication page:

https://aclanthology.org/2025.findings-emnlp.820/

Key concepts:

- retraining-free
- drop experts
- decompose removed experts into neuron segments
- recombine useful neuron-level information

### REAM

Paper:

https://arxiv.org/abs/2604.04356

Key concepts:

- router-weighted expert activation merging
- contrast pruning vs merging
- calibration-data composition / Pareto trade-offs

### C-Prune

Paper:

https://arxiv.org/abs/2504.07807

Key concepts:

- clustering
- expert similarity
- layer-wise homogeneity
- adaptive task-specific pruning

### COMPEL

ACL Anthology:

https://aclanthology.org/2026.findings-acl.1521/

Key concepts:

- layer-adaptive pruning
- Fisher-information expert importance
- layer importance
- weight compensation

---

## 38. Final architecture summary

```text
                         POE CLI
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
      inspect            profile            optimize
         │                  │                  │
         ▼                  ▼                  ▼
       Ingot          compute backend      planner
         │              │       │             │
         │             CPU     CUDA            │
         │              │       │             │
         │              └───┬───┘             │
         │                  ▼                  │
         │              observers             │
         │                  │                  │
         │       ┌──────────┼──────────┐       │
         │       ▼          ▼          ▼       │
         │   routing      REAP      future     │
         │                           metrics    │
         │       └──────────┬──────────┘       │
         │                  ▼                  │
         │               profile ──────────────┘
         │                  │
         │                  ▼
         │                plan
         │                  │
         └──────────────────┤
                            ▼
                         apply
                            │
                          Ingot
                            │
                            ▼
                 specialized GGUF fork
```

---

## 39. Definition of success

POE's first meaningful release is successful if a user with a supported GGUF MoE can:

```bash
poe inspect model.gguf

poe profile model.gguf \
  --dataset coding.jsonl \
  --metric reap \
  --device cuda

poe compare coding.poeprofile general.poeprofile

poe plan model.gguf \
  --profile coding.poeprofile \
  --prune 30%

poe apply model.gguf coding.poeplan \
  -o model-coding.gguf
```

and obtain:

1. trustworthy MoE structural information,
2. reproducible evidence of which experts matter for coding,
3. an explicit expert-removal plan,
4. realistic disk/RAM/VRAM estimates,
5. a smaller valid GGUF,
6. no retraining,
7. measurable post-transform quality.

That is enough for POE to be useful before adding DERN, REAM, dynamic expert residency, or a full automatic optimizer.
