/* profiler/stability.h — convergence tracking for calibration (M5).
 *
 * The question "how many calibration tokens are enough?" is answered by
 * watching the *pruning decision*, not the raw scores: between consecutive
 * snapshots of the per-expert saliency, how stable is the bottom-X% set
 * per layer? When that set stops moving, more tokens no longer change what
 * would be pruned. Pure C, no backend dependency.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_STABILITY_H
#define POE_STABILITY_H

#include <stdint.h>

typedef struct {
    uint32_t n_layers, n_experts;
    uint32_t bottom_n;         /* prune-decision set size per layer        */
    double  *prev;             /* previous snapshot [n_layers * n_experts] */
    int      have_prev;
    /* scratch */
    void    *scratch;
} poe_stability;

typedef struct {
    double bottom_jaccard;     /* mean over layers, consecutive snapshots  */
    double spearman;           /* mean rank correlation, consecutive       */
} poe_stability_step;

/* bottom_frac: fraction of experts whose membership defines the pruning
 * decision (e.g. 0.25). */
int  poe_stability_init(poe_stability *s, uint32_t n_layers,
                        uint32_t n_experts, double bottom_frac);
void poe_stability_free(poe_stability *s);

/* Feed the current cumulative scores [n_layers * n_experts]; computes the
 * stability vs the previous snapshot and retains this one. Returns 0 and
 * fills `out`, or -1 on the first call (no previous snapshot yet). */
int poe_stability_update(poe_stability *s, const double *scores,
                         poe_stability_step *out);

#endif
