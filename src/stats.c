/* stats.c — rank correlation over per-layer scores.
 * SPDX-License-Identifier: MIT */
#include "poe/stats.h"

#include <math.h>
#include <stdlib.h>

typedef struct { double v; size_t i; } entry;

static int cmp_asc(const void *a, const void *b) {
    const double x = ((const entry *)a)->v, y = ((const entry *)b)->v;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Average ranks, so ties do not fabricate an ordering. */
static int ranks_of(const double *v, size_t n, entry *tmp, double *rank) {
    for (size_t i = 0; i < n; i++) { tmp[i].v = v[i]; tmp[i].i = i; }
    qsort(tmp, n, sizeof *tmp, cmp_asc);
    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        while (j < n && tmp[j].v == tmp[i].v) j++;
        const double r = 0.5 * ((double)i + (double)(j - 1)) + 1.0;
        for (size_t k = i; k < j; k++) rank[tmp[k].i] = r;
        i = j;
    }
    return 0;
}

static double pearson(const double *a, const double *b, size_t n) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    double sa = 0, sb = 0, sab = 0;
    for (size_t i = 0; i < n; i++) {
        const double da = a[i] - ma, db = b[i] - mb;
        sa += da * da; sb += db * db; sab += da * db;
    }
    if (sa <= 0.0 || sb <= 0.0) return 0.0;
    return sab / sqrt(sa * sb);
}

double poe_spearman(const double *a, const double *b, size_t n) {
    if (a == NULL || b == NULL || n < 2) return 0.0;
    entry  *tmp = malloc(n * sizeof *tmp);
    double *ra  = malloc(n * sizeof *ra);
    double *rb  = malloc(n * sizeof *rb);
    double out = 0.0;
    if (tmp && ra && rb) {
        ranks_of(a, n, tmp, ra);
        ranks_of(b, n, tmp, rb);
        out = pearson(ra, rb, n);
    }
    free(tmp); free(ra); free(rb);
    return out;
}

double poe_depth_rho(const double *v, size_t n) {
    if (v == NULL || n < 2) return 0.0;
    double *idx = malloc(n * sizeof *idx);
    if (idx == NULL) return 0.0;
    for (size_t i = 0; i < n; i++) idx[i] = (double)i;
    const double rho = poe_spearman(v, idx, n);
    free(idx);
    return rho;
}
