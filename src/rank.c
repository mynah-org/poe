/* rank.c — order a layer's experts by how much a workload needs them.
 * SPDX-License-Identifier: MIT */
#include "poe/rank.h"

#include <stdlib.h>

typedef struct { double score; uint32_t idx; } entry;

static int cmp_desc(const void *a, const void *b) {
    const entry *x = a, *y = b;
    if (x->score > y->score) return -1;
    if (x->score < y->score) return 1;
    return x->idx < y->idx ? -1 : x->idx > y->idx ? 1 : 0;
}

int poe_rank_experts(const poe_profile *pr, uint32_t layer, uint32_t n_experts,
                     poe_rank_by by, uint32_t *order) {
    if (pr == NULL || order == NULL || n_experts == 0) return -1;
    if (layer >= pr->n_layers || pr->n_experts != n_experts) return -1;

    const int use_reap = by != POE_RANK_COUNTS && pr->reap_mean != NULL;
    if (!use_reap && pr->sel_count == NULL) return -1;

    entry *e = malloc((size_t)n_experts * sizeof *e);
    if (e == NULL) return -1;
    for (uint32_t i = 0; i < n_experts; i++) {
        const size_t k = (size_t)layer * n_experts + i;
        e[i].idx = i;
        e[i].score = use_reap ? pr->reap_mean[k] : (double)pr->sel_count[k];
    }
    qsort(e, n_experts, sizeof *e, cmp_desc);
    for (uint32_t i = 0; i < n_experts; i++) order[i] = e[i].idx;
    free(e);
    return use_reap;
}
