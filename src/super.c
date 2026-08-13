/* super.c — activation-magnitude outliers, per layer.
 *
 * One pass per layer: median and MAD of `actnorm_mean`, then a robust
 * z-score for every expert. The scale factor 1.4826 puts MAD on the same
 * footing as a standard deviation for normally distributed data, so a
 * threshold of 6 means what it usually means — while staying immune to the
 * outliers themselves, which is the whole point.
 *
 * A layer where every expert has the same activation norm has MAD 0 and
 * therefore no z-score at all. That is not "no outliers": it is a layer the
 * detector cannot speak about, and it is counted separately rather than
 * folded into the clean answer.
 *
 * SPDX-License-Identifier: MIT */
#include "poe/super.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return  1;
    return 0;
}

/* Median of a scratch buffer the caller does not need preserved. */
static double median_inplace(double *v, uint32_t n) {
    if (n == 0) return 0.0;
    qsort(v, n, sizeof *v, cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void poe_super_free(poe_super *s) {
    if (s == NULL) return;
    free(s->is_outlier);
    free(s->is_rare_outlier);
    free(s->z);
    free(s);
}

int poe_super_detect(poe_super **out, const poe_profile *pr,
                     double z_threshold, char *err, size_t errsz) {
    if (out == NULL || pr == NULL) return fail(err, errsz, "super: null argument");
    *out = NULL;
    if (pr->actnorm_mean == NULL)
        return fail(err, errsz, "super experts need a profile captured with "
                                "--metric reap (no activation norms here)");
    if (pr->n_layers == 0 || pr->n_experts == 0)
        return fail(err, errsz, "super: empty profile");

    const uint32_t L = pr->n_layers, E = pr->n_experts;
    poe_super *s = calloc(1, sizeof *s);
    if (s == NULL) return fail(err, errsz, "super: out of memory");
    s->n_layers = L;
    s->n_experts = E;
    s->z_threshold = z_threshold > 0 ? z_threshold : POE_SUPER_DEFAULT_Z;

    s->is_outlier      = calloc((size_t)L * E, 1);
    s->is_rare_outlier = calloc((size_t)L * E, 1);
    s->z               = calloc((size_t)L * E, sizeof *s->z);
    double *scratch    = malloc((size_t)E * sizeof *scratch);
    if (!s->is_outlier || !s->is_rare_outlier || !s->z || !scratch) {
        free(scratch);
        poe_super_free(s);
        return fail(err, errsz, "super: out of memory");
    }

    for (uint32_t l = 0; l < L; l++) {
        const double   *an  = pr->actnorm_mean + (size_t)l * E;
        const uint64_t *sel = pr->sel_count ? pr->sel_count + (size_t)l * E : NULL;

        memcpy(scratch, an, (size_t)E * sizeof *scratch);
        const double med = median_inplace(scratch, E);

        for (uint32_t e = 0; e < E; e++)
            scratch[e] = an[e] > med ? an[e] - med : med - an[e];
        const double mad = median_inplace(scratch, E);

        if (!(mad > 0.0)) { s->n_layers_undecidable++; continue; }

        /* the uniform share a layer's experts would each see if routing were
         * flat, used only to say what counts as "rarely selected" */
        double share = 0;
        if (sel != NULL) {
            uint64_t total = 0;
            for (uint32_t e = 0; e < E; e++) total += sel[e];
            share = (double)total / (double)E;
        }

        uint32_t here = 0;
        for (uint32_t e = 0; e < E; e++) {
            const double z = (an[e] - med) / (1.4826 * mad);
            s->z[(size_t)l * E + e] = z;
            if (z < s->z_threshold) continue;

            s->is_outlier[(size_t)l * E + e] = 1;
            s->n_outliers++;
            here++;
            if (sel != NULL && share > 0 &&
                (double)sel[e] < POE_SUPER_RARE_SHARE * share) {
                s->is_rare_outlier[(size_t)l * E + e] = 1;
                s->n_rare_outliers++;
            }
            if (z > s->z_max) {
                s->z_max = z;
                s->z_max_layer = l;
                s->z_max_expert = e;
            }
        }
        if (here > s->max_per_layer) { s->max_per_layer = here; s->max_layer = l; }
    }

    free(scratch);
    *out = s;
    return 0;
}
