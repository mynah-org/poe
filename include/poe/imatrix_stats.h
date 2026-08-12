/* poe/imatrix_stats.h — reading an imatrix back as a per-layer statistic.
 *
 * `poe profile --metric imatrix` writes llama.cpp's GGUF imatrix: per routed
 * weight tensor, `<name>.in_sum2` [cols, n_experts] and `<name>.counts`
 * [1, n_experts]. This reads one back and reduces it to the two per-layer
 * numbers a bit allocation could plausibly be driven by.
 *
 * Why the imatrix and not the profile: REAP saliency measures an expert's
 * contribution to the layer *output*, which is the right question for
 * deletion. Requantization damages the product of a weight matrix with its
 * *inputs*, and the imatrix is literally the quantity a weighted encoder
 * fits against. Deletion and requantization should not be assumed to want
 * the same ranking.
 *
 * Two statistics, deliberately of different kinds:
 *
 *   energy         mean squared input activation per element per routed
 *                  slot. Carries the activation scale, so on a residual
 *                  network it is expected to grow with depth — the exact
 *                  confound that made per-layer REAP look informative when
 *                  it was not. Read `poe_depth_rho` on it before believing
 *                  it.
 *   concentration  1 - participation ratio of the per-column energy vector,
 *                  in [0,1). Scale-free by construction: it measures how far
 *                  the input energy is concentrated in a few columns, which
 *                  is what makes a block hard to fit at two or three bits,
 *                  and it cannot be a norm ramp in disguise.
 *
 * Neither is a measured result. They are candidate rankings; the arm that
 * ships is whichever beats uniform at matched bytes, and so far none has.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_IMATRIX_STATS_H
#define POE_IMATRIX_STATS_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"

#define POE_IMSTAT_NPROJ 3         /* gate, up, down — the routed slabs */

typedef struct {
    uint32_t n_layers, n_experts;
    uint32_t cols[POE_IMSTAT_NPROJ];       /* 0 when the projection is absent */
    uint32_t n_entries;                    /* in_sum2 tensors read            */
    uint32_t chunk_count, chunk_size;
    char     dataset[192];                 /* imatrix.datasets[0], or ""      */

    double   *energy;                      /* [n_layers]                      */
    double   *concentration;               /* [n_layers]                      */
    uint64_t *counts;                      /* [n_layers * n_experts]          */
    uint64_t  slots_total;                 /* routed (token, expert) slots    */
    uint32_t  experts_unseen;              /* count 0: legal, and informative */
} poe_imatrix_stats;

/* Load and reduce. Fails when the file is not an imatrix, carries no routed
 * expert entries, or is internally inconsistent. Returns 0 on success. */
int  poe_imatrix_stats_load(poe_imatrix_stats **out, const char *path,
                            char *err, size_t errsz);
void poe_imatrix_stats_free(poe_imatrix_stats *s);

#endif
