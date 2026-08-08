/* profiler/accum.h — streaming accumulators for MoE routing observation.
 *
 * Pure C, no inference-backend dependency: the runner (tools/poe_profile.c)
 * copies tensors off the backend and feeds them here; these functions never
 * see llama.cpp/ggml types, so they are unit-testable in CI without a model.
 *
 * Everything is O(1) memory in the token count — nothing per-token is
 * retained (vision.md §8: streaming accumulators only).
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_ACCUM_H
#define POE_ACCUM_H

#include <stdint.h>
#include <stdio.h>

/* Cumulative-mass thresholds tracked per token: the mean minimal k whose
 * renormalized probability mass reaches T answers idea2's "mean K required
 * for X% mass" directly from the routing distribution. */
#define POE_ACCUM_NMASS 4
extern const double poe_accum_mass_thresholds[POE_ACCUM_NMASS]; /* .80 .90 .95 .99 */

typedef struct {
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;

    /* per layer × expert, indexed [layer * n_experts + expert] */
    uint64_t *sel_count;      /* how often the expert was in the applied top-k */
    double   *gate_sum;       /* sum of applied (renormalized) gate weights    */

    /* per layer */
    uint64_t *tok_probs;      /* tokens whose full distribution was observed   */
    uint64_t *tok_sel;        /* tokens whose selection was observed           */
    double   *entropy_sum;    /* Σ over tokens of H(p) in bits                 */
    double   *mass_k_sum[POE_ACCUM_NMASS]; /* Σ of min-k reaching threshold    */
} poe_accum;

int  poe_accum_init(poe_accum *a, uint32_t n_layers, uint32_t n_experts,
                    uint32_t top_k);
void poe_accum_free(poe_accum *a);

/* Observe one layer's full routing distribution for a batch of T tokens.
 * `probs` is [n_experts × T], token-major columns (probs[e + t*n_experts]).
 * If the rows are not a probability distribution (delayed-softmax archs like
 * gpt-oss expose raw biased scores under the same graph name), pass
 * `is_logits = 1` and they are soft-maxed here before accumulation. */
void poe_accum_observe_probs(poe_accum *a, uint32_t layer, uint32_t T,
                             const float *probs, int is_logits);

/* Observe one layer's applied expert selection for a batch of T tokens.
 * `ids` is [top_k × T] (ids[s + t*top_k]), `weights` the matching applied
 * gate weights, or NULL when only frequencies are wanted. Out-of-range ids
 * are counted in *bad_ids (routing corruption detector) and skipped. */
void poe_accum_observe_selection(poe_accum *a, uint32_t layer, uint32_t T,
                                 const int32_t *ids, const float *weights,
                                 uint64_t *bad_ids);

/* Write the accumulated statistics as the per-layer body of a profile:
 * a JSON array (one object per layer). Caller wraps it in the envelope. */
void poe_accum_write_json(const poe_accum *a, FILE *f, const char *indent);

#endif
