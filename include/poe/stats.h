/* poe/stats.h — the two rank statistics POE reports on its own artifacts.
 *
 * Kept separate from profile.c's internal versions on purpose: those work on
 * per-expert selection counts (u64), these on the per-layer scores an
 * allocation is driven by (double). Same definition, different unit.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_STATS_H
#define POE_STATS_H

#include <stddef.h>

/* Spearman rank correlation with average ranks for ties. Returns 0.0 when
 * n < 2 or when either vector is constant — a constant vector has no
 * ranking, and reporting 0 says exactly that. */
double poe_spearman(const double *a, const double *b, size_t n);

/* Spearman of `v` against its own index: how close a per-layer score is to
 * being a monotone function of depth.
 *
 * This is the diagnostic two negative results paid for. Per-layer REAP
 * totals on Qwen3.6 rise 0.03 -> 1.00 with depth because the residual
 * stream's norm grows, not because deep layers are more fragile, and an
 * allocation driven by that ranking lost to uniform — as did its inverse.
 * A score that correlates ~1.0 with depth is guilty until proven innocent. */
double poe_depth_rho(const double *v, size_t n);

#endif
