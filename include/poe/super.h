/* poe/super.h — super experts: rare, enormous, and unsafe to delete.
 *
 * Constraint #3 of the routing track, made computable. arXiv:2507.23279
 * reports that a handful of experts carry activation magnitudes orders of
 * magnitude above their neighbours, and that removing as few as three of
 * them breaks Qwen3-30B-A3B outright. They are dangerous precisely because
 * the obvious safety signal misses them: they are *rarely selected*, so any
 * frequency-driven ranking puts them at the bottom of the list.
 *
 * The detector is deliberately robust rather than clever. Activation norms
 * are heavy-tailed by hypothesis, so a mean and a standard deviation would
 * be dragged by the very outliers being looked for; the median and the
 * median absolute deviation are not. An expert is flagged when its
 * `actnorm_mean` sits far above its layer's median on that scale.
 *
 * Detection is a report. Protection is a separate decision, taken by
 * whatever consumes the flags — see poe_plan_build_opts.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_SUPER_H
#define POE_SUPER_H

#include <stddef.h>
#include <stdint.h>

#include "poe/profile.h"

/* Robust z above the layer median, in MAD units, past which an expert counts
 * as an activation outlier. Six is deliberately far out: this flag exists to
 * catch a documented failure mode, not to relabel the top of a distribution.
 */
#define POE_SUPER_DEFAULT_Z 6.0

/* An expert is "rare" when the workload routes to it less than this share of
 * the uniform expectation (tokens x top_k / n_experts). Rarity is not part
 * of the protection test — an outlier is protected whether it is rare or not
 * — but the intersection is the published signature and is reported apart. */
#define POE_SUPER_RARE_SHARE 0.5

typedef struct {
    uint32_t n_layers, n_experts;

    /* [n_layers * n_experts], 1 when the expert is an activation outlier. */
    uint8_t *is_outlier;
    /* The subset that is also rarely selected: the published signature. */
    uint8_t *is_rare_outlier;

    /* Robust z-score per expert, kept for reporting and for --json. */
    double  *z;

    double   z_threshold;
    uint64_t n_outliers, n_rare_outliers;
    uint32_t max_per_layer, max_layer;
    double   z_max;
    uint32_t z_max_layer, z_max_expert;

    /* Layers whose actnorm data was missing or degenerate (every value
     * identical, so MAD is zero and no z-score exists). Reported rather than
     * silently treated as "no outliers here". */
    uint32_t n_layers_undecidable;
} poe_super;

/* Detect over a profile carrying actnorm data (--metric reap). `z_threshold`
 * <= 0 selects POE_SUPER_DEFAULT_Z. Returns 0 on success, -1 with a message
 * when the profile has no activation norms to look at. */
int  poe_super_detect(poe_super **out, const poe_profile *pr,
                      double z_threshold, char *err, size_t errsz);
void poe_super_free(poe_super *s);

#endif
