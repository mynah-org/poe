/* profiler/stability.c — convergence tracking for calibration.
 * SPDX-License-Identifier: MIT */
#include "stability.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double v; uint32_t idx; } ent;

static int cmp_asc(const void *x, const void *y) {
    const ent *a = x, *b = y;
    if (a->v != b->v) return a->v > b->v ? 1 : -1;
    return a->idx < b->idx ? -1 : 1;      /* deterministic ties */
}

int poe_stability_init(poe_stability *s, uint32_t n_layers,
                       uint32_t n_experts, double bottom_frac) {
    memset(s, 0, sizeof *s);
    s->n_layers  = n_layers;
    s->n_experts = n_experts;
    s->bottom_n  = (uint32_t)((double)n_experts * bottom_frac + 0.5);
    if (s->bottom_n == 0) s->bottom_n = 1;
    if (s->bottom_n > n_experts) s->bottom_n = n_experts;

    s->prev = malloc((size_t)n_layers * n_experts * sizeof *s->prev);
    /* scratch: two ent arrays + two rank arrays + two membership maps */
    s->scratch = malloc((size_t)n_experts *
                        (2 * sizeof(ent) + 2 * sizeof(double) + 2));
    if (s->prev == NULL || s->scratch == NULL) {
        poe_stability_free(s);
        return -1;
    }
    return 0;
}

void poe_stability_free(poe_stability *s) {
    free(s->prev);
    free(s->scratch);
    memset(s, 0, sizeof *s);
}

static void ranks_of(const double *vals, uint32_t n, ent *tmp, double *out) {
    for (uint32_t i = 0; i < n; i++) { tmp[i].v = vals[i]; tmp[i].idx = i; }
    qsort(tmp, n, sizeof *tmp, cmp_asc);
    uint32_t i = 0;
    while (i < n) {
        uint32_t j = i;
        while (j + 1 < n && tmp[j + 1].v == tmp[i].v) j++;
        double r = ((double)i + (double)j) / 2.0;
        for (uint32_t k = i; k <= j; k++) out[tmp[k].idx] = r;
        i = j + 1;
    }
}

static double pearson(const double *x, const double *y, uint32_t n) {
    double mx = 0, my = 0;
    for (uint32_t i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (uint32_t i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    if (sxx == 0.0 || syy == 0.0) return 1.0;   /* constant = unchanged */
    return sxy / sqrt(sxx * syy);
}

int poe_stability_update(poe_stability *s, const double *scores,
                         poe_stability_step *out) {
    const uint32_t L = s->n_layers, E = s->n_experts, B = s->bottom_n;

    if (!s->have_prev) {
        memcpy(s->prev, scores, (size_t)L * E * sizeof *scores);
        s->have_prev = 1;
        return -1;
    }

    ent    *ta = s->scratch;
    ent    *tb = ta + E;
    double *ra = (double *)(tb + E);
    double *rb = ra + E;
    unsigned char *ina = (unsigned char *)(rb + E);
    unsigned char *inb = ina + E;

    double jac_sum = 0, sp_sum = 0;
    for (uint32_t l = 0; l < L; l++) {
        const double *pa = s->prev + (size_t)l * E;
        const double *pb = scores  + (size_t)l * E;

        memset(ina, 0, E); memset(inb, 0, E);
        for (uint32_t i = 0; i < E; i++) { ta[i].v = pa[i]; ta[i].idx = i; }
        qsort(ta, E, sizeof *ta, cmp_asc);
        for (uint32_t i = 0; i < B; i++) ina[ta[i].idx] = 1;
        for (uint32_t i = 0; i < E; i++) { tb[i].v = pb[i]; tb[i].idx = i; }
        qsort(tb, E, sizeof *tb, cmp_asc);
        for (uint32_t i = 0; i < B; i++) inb[tb[i].idx] = 1;

        uint32_t inter = 0, uni = 0;
        for (uint32_t e = 0; e < E; e++) {
            if (ina[e] && inb[e]) inter++;
            if (ina[e] || inb[e]) uni++;
        }
        jac_sum += uni ? (double)inter / (double)uni : 1.0;

        ranks_of(pa, E, ta, ra);
        ranks_of(pb, E, tb, rb);
        sp_sum += pearson(ra, rb, E);
    }

    out->bottom_jaccard = jac_sum / L;
    out->spearman       = sp_sum / L;

    memcpy(s->prev, scores, (size_t)L * E * sizeof *scores);
    return 0;
}
