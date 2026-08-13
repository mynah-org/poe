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

/* The mean min-k answers "how many experts does a token need on average".
 * It cannot answer the question that decides whether *token-adaptive* K is
 * worth kernel work: whether tokens differ from each other. A schedule can
 * only exploit variation across layers; adaptivity needs variation across
 * tokens, and a mean hides it completely. So the full distribution is kept
 * as a histogram. The bins span the whole expert count rather than a fixed
 * prefix: on a 256-expert router with entropy near 8 bits, the mass a token
 * needs is measured in hundreds of experts, and a histogram capped at a
 * small prefix reports saturation instead of an answer — it did, on the
 * first run. Bin b covers k in [b*w+1, (b+1)*w] with w = ceil(E/KHIST), so
 * the resolution is exact whenever E <= KHIST. */
#define POE_ACCUM_KHIST 64


typedef struct {
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t top_k;

    /* per layer × expert, indexed [layer * n_experts + expert] */
    uint64_t *sel_count;      /* how often the expert was in the applied top-k */
    double   *gate_sum;       /* sum of applied (renormalized) gate weights    */

    /* REAP saliency (milestone M4), same indexing. Only fed when expert
     * outputs are captured (--metric reap):
     *   reap_sum  = Σ over routed tokens of gate · ‖expert_output‖₂
     *   norm_sum  = Σ over routed tokens of      ‖expert_output‖₂
     * REAP score = reap_sum / reap_count (mean over routed tokens, as in
     * the reference); norm_sum/reap_count is the unweighted activation-norm
     * baseline metric. */
    uint64_t *reap_count;
    double   *reap_sum;
    double   *norm_sum;

    /* per layer */
    uint64_t *tok_probs;      /* tokens whose full distribution was observed   */
    uint64_t *tok_sel;        /* tokens whose selection was observed           */
    double   *entropy_sum;    /* Σ over tokens of H(p) in bits                 */
    double   *mass_k_sum[POE_ACCUM_NMASS]; /* Σ of min-k reaching threshold    */
    uint64_t *mass_k_hist[POE_ACCUM_NMASS];/* [layer][bin], see khist_bin      */
    double   *topk_mass_sum;  /* Σ of the probability the applied top-k holds  */

    /* How much of what the applied experts actually *contribute* comes from
     * how few of them. Router mass turned out not to be the quantity that
     * governs which experts matter (R9: the top-8 holds 15.7% of it and the
     * model is fine), and the direction M9b found is that contribution beats
     * frequency. So this is the same min-k question asked of the quantity
     * that decides: per token, sort the applied slots by gate x ||output||
     * and count how many are needed to reach the threshold.
     *
     * Bounded by top_k rather than by the expert count, which is what makes
     * it actionable: "3 of the 8 we run carry 90%" is a statement about K.
     * Indexed [layer * top_k + (k-1)]; fed only under --metric reap. */
    double   *contrib_k_sum[POE_ACCUM_NMASS];  /* Σ of min-k, per layer      */
    uint64_t *contrib_k_hist[POE_ACCUM_NMASS]; /* [layer][k-1]               */
    uint64_t *contrib_tok;    /* tokens with a usable contribution total     */
} poe_accum;

/* Which histogram bin a min-k of `k` out of `n_experts` lands in. Exposed
 * because a reader has to invert it to report percentiles in experts. */
uint32_t poe_accum_khist_bin(uint32_t k, uint32_t n_experts);

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

/* Observe expert-output norms for REAP saliency. `ids` and `weights` as in
 * observe_selection; `norms` holds the matching ‖expert_output‖₂ per slot,
 * all [top_k × T]. Out-of-range ids are skipped (counted in *bad_ids). */
void poe_accum_observe_reap(poe_accum *a, uint32_t layer, uint32_t T,
                            const int32_t *ids, const float *weights,
                            const float *norms, uint64_t *bad_ids);

/* Fill out[n_layers * n_experts] with the current per-expert score used
 * for the pruning decision: REAP mean when reap observations exist,
 * selection count otherwise. Returns 1 when REAP was used, 0 otherwise. */
int poe_accum_scores(const poe_accum *a, double *out);

/* Write the accumulated statistics as the per-layer body of a profile:
 * a JSON array (one object per layer). Caller wraps it in the envelope.
 * REAP arrays are emitted only when reap observations exist. */
void poe_accum_write_json(const poe_accum *a, FILE *f, const char *indent);

#endif
