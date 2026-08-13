# Do language experts exist? (L1, step 1)

The hypothesis behind an English-only specialization is that a multilingual
MoE spends part of its expert budget on languages you never use, and that
those experts separate cleanly enough to delete. If they do, an English-only
profile should be able to prune far deeper than a general one.

This is the hour-long experiment designed to kill the idea or greenlight it,
before any pruning run is spent on it.

## Design: control the topic, vary only the language

Comparing an English corpus against a Chinese one usually compares two
different subjects as much as two languages. So the corpus is built from the
**same 48 concepts in all three languages**, resolved through Wikipedia's
interlanguage links: `Volcano` / `Vulcano` / `火山`, and so on. What the
profiles then disagree about is language, not subject matter.

The floor is measured the same way as in
[profile-noise-floor.md](profile-noise-floor.md): the concepts are split in
two disjoint halves and each half profiled independently, **within one
language**. All profiles: Qwen3.6-35B-A3B Q8_0, 8192 tokens, `--metric reap`.

## Result

| pair | REAP bottom-25% Jaccard | top-25% |
|---|---|---|
| **floor, English** — en-A vs en-B | **0.566** | 0.679 |
| **floor, Chinese** — zh-A vs zh-B | **0.551** | 0.637 |
| en vs it | 0.469 | 0.516 |
| en vs zh | 0.425 | 0.439 |
| it vs zh | 0.411 | 0.436 |
| *for scale* — en vs code | 0.223 | 0.099 |
| *for scale* — zh vs code | 0.181 | 0.100 |

**The floor is a property of the metric, not of the language.** English 0.566
and Chinese 0.551 — measured on entirely different text in different scripts.
That is what makes the rest of the table readable.

## What it says

1. **Language structure is real.** Every cross-language pair sits below the
   floor, and the reading is conservative: the floor's two halves differ in
   *topic* as well as in sampling, while the language pairs are topic-matched
   by construction. The true no-language-effect baseline is therefore at
   least 0.55, and probably above it.

2. **And the pairs order themselves by linguistic distance**: en–it 0.469 >
   en–zh 0.425 > it–zh 0.411. Three independent comparisons landing in the
   order a linguist would predict — with English, the hub language of the
   training data, overlapping more with everything — is not what sampling
   noise does.

3. **But it is weak, and far weaker than domain structure.** Measured as
   distance below the floor, which is the only way these numbers compare:

   | contrast | distance below the floor |
   |---|---|
   | language (en/it/zh) | 0.09 – 0.15 |
   | domain (prose vs code) | 0.33 – 0.38 |

   Domain identity separates **2.5–3× more sharply**. The busiest experts
   make the point even more bluntly: prose and code share 10% of their top
   quarter, while two languages share 44–52% of theirs.

## Consequence

The step-1 rule was written as "en↔zh ≈ 0.9 → the idea dies; ≈ 0.4–0.5 →
real structure, proceed". At 0.425 the naive reading says proceed — but that
threshold was set before the metric had a floor, and against a floor of 0.55
the separation is a third of what a genuine domain contrast produces on the
same model, the same day, with the same settings.

So: **the experts a language uses are not a cleanly separable set, and an
English-only profile should not be expected to buy much extra prune margin.**
That is exactly the pessimistic end of the honest prior recorded before the
run (+5–15% of margin, not a doubling), and it is the reason not to spend a
prune-and-evaluate cycle chasing a doubling.

The three structural reasons given for that prior stand, and this measurement
is consistent with all of them: Qwen3.6's always-active shared experts absorb
generalist load and are unprunable by construction, embeddings and attention
are untouched by expert pruning and carry much of what is language-specific,
and MoE routers key on token-level patterns more than on language semantics —
which is precisely why two languages writing about the same 48 concepts still
agree on half of their busiest experts.
