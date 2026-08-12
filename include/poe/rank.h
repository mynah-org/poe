/* poe/rank.h — ordering a layer's experts by how much a workload needs them.
 *
 * One definition, shared by everything that has to draw a line between hot
 * and cold experts: requantization, splitting, and any future policy. Two
 * signals, because they are not the same ordering and the difference is
 * measurable (Jaccard 0.496 on Qwen3.6 at a 57% cut):
 *
 *   REAP saliency   an expert's contribution to the layer output. Measured
 *                   better at predicting quantization sensitivity, and the
 *                   default for that reason.
 *   frequency       how often the workload routes there at all — the M9
 *                   thesis stated literally, and the weaker of the two.
 *
 * Within a layer, selection counts are a usable ranking. Summed *over* a
 * layer they are tokens x top_k in every layer, a conserved quantity that
 * ranks nothing — which is why per-layer allocation must never use them.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_RANK_H
#define POE_RANK_H

#include <stdint.h>

#include "poe/profile.h"

typedef enum {
    POE_RANK_REAP = 0,     /* falls back to counts when the profile has none */
    POE_RANK_COUNTS = 1
} poe_rank_by;

/* Fill `order` with the layer's expert indices, most needed first. Ties
 * resolve by index so the same profile always yields the same order.
 * Returns 1 if REAP saliency was used, 0 if selection counts were, or -1
 * when neither is available. */
int poe_rank_experts(const poe_profile *pr, uint32_t layer, uint32_t n_experts,
                     poe_rank_by by, uint32_t *order);

#endif
