/* profile.c — .poeprofile loading and cross-profile metrics.
 * SPDX-License-Identifier: MIT */
#include "poe/profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

void poe_profile_free(poe_profile *p) {
    if (p == NULL) return;
    free(p->layer_tokens);
    free(p->entropy_bits);
    for (int i = 0; i < POE_PROFILE_NMASS; i++) free(p->mass_k[i]);
    free(p->sel_count);
    free(p->gate_mean);
    free(p->reap_mean);
    free(p->actnorm_mean);
    free(p);
}

static const char *const MASS_KEYS[POE_PROFILE_NMASS] =
    { "0.80", "0.90", "0.95", "0.99" };

int poe_profile_load(poe_profile **out, const char *path,
                     char *err, size_t errsz) {
    if (out == NULL) return -1;
    *out = NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return fail(err, errsz, "cannot open profile");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return fail(err, errsz, "empty profile"); }
    char *text = malloc((size_t)sz);
    if (text == NULL || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        free(text); fclose(f);
        return fail(err, errsz, "cannot read profile");
    }
    fclose(f);

    poe_json *j = poe_json_parse(text, (size_t)sz, err, errsz);
    free(text);
    if (j == NULL) return -1;

    poe_profile *p = calloc(1, sizeof *p);
    if (p == NULL) { poe_json_free(j); return fail(err, errsz, "out of memory"); }

    const poe_json *v = poe_json_get(j, "poeprofile");
    if (v == NULL) {
        poe_json_free(j); free(p);
        return fail(err, errsz, "not a .poeprofile (missing \"poeprofile\" key)");
    }
    p->version = (uint32_t)poe_json_u64(v, 0);

    snprintf(p->poe_version, sizeof p->poe_version, "%s",
             poe_json_str(poe_json_get(j, "poe_version"), ""));
    snprintf(p->fingerprint, sizeof p->fingerprint, "%s",
             poe_json_str(poe_json_get(j, "model_fingerprint"), ""));
    snprintf(p->arch, sizeof p->arch, "%s",
             poe_json_str(poe_json_get(j, "arch"), ""));
    snprintf(p->dataset_hash, sizeof p->dataset_hash, "%s",
             poe_json_str(poe_json_get(j, "dataset_hash"), ""));

    p->n_layers  = (uint32_t)poe_json_u64(poe_json_get(j, "n_layers"), 0);
    p->n_experts = (uint32_t)poe_json_u64(poe_json_get(j, "n_experts"), 0);
    p->top_k     = (uint32_t)poe_json_u64(poe_json_get(j, "top_k"), 0);
    p->tokens    = poe_json_u64(poe_json_get(j, "tokens"), 0);

    const poe_json *layers = poe_json_get(j, "layers");
    if (p->n_layers == 0 || p->n_experts == 0 ||
        poe_json_len(layers) != p->n_layers) {
        poe_json_free(j); poe_profile_free(p);
        return fail(err, errsz, "profile shape is inconsistent");
    }

    size_t L = p->n_layers, LE = (size_t)p->n_layers * p->n_experts;
    p->layer_tokens = calloc(L, sizeof *p->layer_tokens);
    p->entropy_bits = calloc(L, sizeof *p->entropy_bits);
    p->sel_count    = calloc(LE, sizeof *p->sel_count);
    p->gate_mean    = calloc(LE, sizeof *p->gate_mean);
    int ok = p->layer_tokens && p->entropy_bits && p->sel_count && p->gate_mean;
    for (int i = 0; i < POE_PROFILE_NMASS; i++) {
        p->mass_k[i] = calloc(L, sizeof *p->mass_k[i]);
        ok = ok && p->mass_k[i];
    }
    if (!ok) {
        poe_json_free(j); poe_profile_free(p);
        return fail(err, errsz, "out of memory");
    }

    for (uint32_t l = 0; l < p->n_layers; l++) {
        const poe_json *lj = poe_json_at(layers, l);
        uint32_t li = (uint32_t)poe_json_u64(poe_json_get(lj, "layer"), l);
        if (li >= p->n_layers) li = l;

        p->layer_tokens[li] = poe_json_u64(poe_json_get(lj, "tokens"), 0);
        p->entropy_bits[li] = poe_json_num(poe_json_get(lj, "entropy_bits_mean"), 0.0);

        const poe_json *mk = poe_json_get(lj, "mass_k_mean");
        for (int i = 0; i < POE_PROFILE_NMASS; i++)
            p->mass_k[i][li] = poe_json_num(poe_json_get(mk, MASS_KEYS[i]), 0.0);

        const poe_json *sc = poe_json_get(lj, "sel_count");
        const poe_json *gm = poe_json_get(lj, "gate_mean");
        if (poe_json_len(sc) != p->n_experts || poe_json_len(gm) != p->n_experts) {
            poe_json_free(j); poe_profile_free(p);
            return fail(err, errsz, "layer expert arrays have wrong length");
        }
        for (uint32_t e = 0; e < p->n_experts; e++) {
            p->sel_count[(size_t)li * p->n_experts + e] =
                poe_json_u64(poe_json_at(sc, e), 0);
            p->gate_mean[(size_t)li * p->n_experts + e] =
                poe_json_num(poe_json_at(gm, e), 0.0);
        }

        /* optional REAP arrays (allocate lazily on first sight) */
        const poe_json *rm = poe_json_get(lj, "reap_mean");
        const poe_json *am = poe_json_get(lj, "actnorm_mean");
        if (poe_json_len(rm) == p->n_experts) {
            if (p->reap_mean == NULL)
                p->reap_mean = calloc(LE, sizeof *p->reap_mean);
            if (p->actnorm_mean == NULL)
                p->actnorm_mean = calloc(LE, sizeof *p->actnorm_mean);
            if (p->reap_mean == NULL || p->actnorm_mean == NULL) {
                poe_json_free(j); poe_profile_free(p);
                return fail(err, errsz, "out of memory");
            }
            for (uint32_t e = 0; e < p->n_experts; e++) {
                p->reap_mean[(size_t)li * p->n_experts + e] =
                    poe_json_num(poe_json_at(rm, e), 0.0);
                p->actnorm_mean[(size_t)li * p->n_experts + e] =
                    poe_json_num(poe_json_at(am, e), 0.0);
            }
        }
    }

    poe_json_free(j);
    *out = p;
    return 0;
}

/* ── comparison ─────────────────────────────────────────────────────────── */

typedef struct { uint64_t v; uint32_t idx; } sort_ent;

static int cmp_desc_u64(const void *x, const void *y) {
    const sort_ent *a = x, *b = y;
    if (a->v != b->v) return a->v < b->v ? 1 : -1;
    return a->idx < b->idx ? -1 : 1;          /* deterministic tie order */
}

typedef struct { double v; uint32_t idx; } sort_entd;

static int cmp_desc_d(const void *x, const void *y) {
    const sort_entd *a = x, *b = y;
    if (a->v != b->v) return a->v < b->v ? 1 : -1;
    return a->idx < b->idx ? -1 : 1;
}

/* average-rank assignment for Spearman over doubles */
static void avg_ranks_d(const double *vals, uint32_t n, sort_entd *tmp,
                        double *ranks) {
    for (uint32_t i = 0; i < n; i++) { tmp[i].v = vals[i]; tmp[i].idx = i; }
    qsort(tmp, n, sizeof *tmp, cmp_desc_d);
    uint32_t i = 0;
    while (i < n) {
        uint32_t jx = i;
        while (jx + 1 < n && tmp[jx + 1].v == tmp[i].v) jx++;
        double r = ((double)i + (double)jx) / 2.0;
        for (uint32_t k2 = i; k2 <= jx; k2++) ranks[tmp[k2].idx] = r;
        i = jx + 1;
    }
}

/* average-rank assignment for Spearman */
static void avg_ranks(const uint64_t *vals, uint32_t n, sort_ent *tmp,
                      double *ranks) {
    for (uint32_t i = 0; i < n; i++) { tmp[i].v = vals[i]; tmp[i].idx = i; }
    qsort(tmp, n, sizeof *tmp, cmp_desc_u64);
    uint32_t i = 0;
    while (i < n) {
        uint32_t jx = i;
        while (jx + 1 < n && tmp[jx + 1].v == tmp[i].v) jx++;
        double r = ((double)i + (double)jx) / 2.0;
        for (uint32_t k2 = i; k2 <= jx; k2++) ranks[tmp[k2].idx] = r;
        i = jx + 1;
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
    if (sxx == 0.0 || syy == 0.0) return 0.0;
    return sxy / sqrt(sxx * syy);
}

int poe_profile_compare(const poe_profile *a, const poe_profile *b,
                        double top_frac, poe_profile_cmp *out,
                        double *per_layer_jaccard) {
    if (a->n_layers != b->n_layers || a->n_experts != b->n_experts)
        return -1;

    const uint32_t L = a->n_layers, E = a->n_experts;
    uint32_t topn = (uint32_t)((double)E * top_frac + 0.5);
    if (topn == 0) topn = 1;
    if (topn > E) topn = E;

    memset(out, 0, sizeof *out);
    out->jaccard_min = 2.0;
    out->jaccard_max = -1.0;

    sort_ent *tmp   = malloc(E * sizeof *tmp);
    double   *ra    = malloc(E * sizeof *ra);
    double   *rb    = malloc(E * sizeof *rb);
    unsigned char *ina = malloc(E), *inb = malloc(E);
    if (!tmp || !ra || !rb || !ina || !inb) {
        free(tmp); free(ra); free(rb); free(ina); free(inb);
        return -1;
    }

    double jac_sum = 0, wjac_sum = 0, sp_sum = 0, jsd_sum = 0;

    for (uint32_t l = 0; l < L; l++) {
        const uint64_t *ca = a->sel_count + (size_t)l * E;
        const uint64_t *cb = b->sel_count + (size_t)l * E;

        /* top-X% sets */
        memset(ina, 0, E); memset(inb, 0, E);
        for (uint32_t i = 0; i < E; i++) { tmp[i].v = ca[i]; tmp[i].idx = i; }
        qsort(tmp, E, sizeof *tmp, cmp_desc_u64);
        for (uint32_t i = 0; i < topn; i++) ina[tmp[i].idx] = 1;
        for (uint32_t i = 0; i < E; i++) { tmp[i].v = cb[i]; tmp[i].idx = i; }
        qsort(tmp, E, sizeof *tmp, cmp_desc_u64);
        for (uint32_t i = 0; i < topn; i++) inb[tmp[i].idx] = 1;

        uint32_t inter = 0, uni = 0;
        uint64_t ex_a = 0, ex_b = 0, cold = 0;
        for (uint32_t e = 0; e < E; e++) {
            if (ina[e] && inb[e]) inter++;
            if (ina[e] || inb[e]) uni++;
            if (ina[e] && !inb[e]) ex_a++;
            if (!ina[e] && inb[e]) ex_b++;
            if (ca[e] == 0 && cb[e] == 0) cold++;
        }
        double jac = uni ? (double)inter / (double)uni : 1.0;
        jac_sum += jac;
        if (per_layer_jaccard) per_layer_jaccard[l] = jac;
        if (jac < out->jaccard_min) { out->jaccard_min = jac; out->jaccard_min_layer = l; }
        if (jac > out->jaccard_max) { out->jaccard_max = jac; out->jaccard_max_layer = l; }
        out->exclusive_a += ex_a;
        out->exclusive_b += ex_b;
        out->cold_both   += cold;

        /* weighted Jaccard on selection frequencies */
        uint64_t ta = 0, tb = 0;
        for (uint32_t e = 0; e < E; e++) { ta += ca[e]; tb += cb[e]; }
        double wmin = 0, wmax = 0, jsd = 0;
        for (uint32_t e = 0; e < E; e++) {
            double fa = ta ? (double)ca[e] / (double)ta : 0.0;
            double fb = tb ? (double)cb[e] / (double)tb : 0.0;
            wmin += fa < fb ? fa : fb;
            wmax += fa > fb ? fa : fb;
            double m = 0.5 * (fa + fb);
            if (fa > 0) jsd += 0.5 * fa * log2(fa / m);
            if (fb > 0) jsd += 0.5 * fb * log2(fb / m);
        }
        wjac_sum += wmax > 0 ? wmin / wmax : 1.0;
        jsd_sum  += jsd;

        /* Spearman with average ranks */
        avg_ranks(ca, E, tmp, ra);
        avg_ranks(cb, E, tmp, rb);
        sp_sum += pearson(ra, rb, E);
    }

    out->jaccard_mean  = jac_sum / L;
    out->wjaccard_mean = wjac_sum / L;
    out->spearman_mean = sp_sum / L;
    out->jsd_bits_mean = jsd_sum / L;

    /* REAP agreement: rank correlation of saliency and — what pruning
     * actually consumes — agreement of the bottom-X% prune sets. */
    if (a->reap_mean && b->reap_mean) {
        sort_entd *td = malloc(E * sizeof *td);
        if (td) {
            out->has_reap = 1;
            double rsp = 0, rjac = 0;
            for (uint32_t l = 0; l < L; l++) {
                const double *sa = a->reap_mean + (size_t)l * E;
                const double *sb = b->reap_mean + (size_t)l * E;
                avg_ranks_d(sa, E, td, ra);
                avg_ranks_d(sb, E, td, rb);
                rsp += pearson(ra, rb, E);

                memset(ina, 0, E); memset(inb, 0, E);
                for (uint32_t i = 0; i < E; i++) { td[i].v = sa[i]; td[i].idx = i; }
                qsort(td, E, sizeof *td, cmp_desc_d);
                for (uint32_t i = 0; i < topn; i++) ina[td[E - 1 - i].idx] = 1;
                for (uint32_t i = 0; i < E; i++) { td[i].v = sb[i]; td[i].idx = i; }
                qsort(td, E, sizeof *td, cmp_desc_d);
                for (uint32_t i = 0; i < topn; i++) inb[td[E - 1 - i].idx] = 1;
                uint32_t inter = 0, uni = 0;
                for (uint32_t e = 0; e < E; e++) {
                    if (ina[e] && inb[e]) inter++;
                    if (ina[e] || inb[e]) uni++;
                }
                rjac += uni ? (double)inter / (double)uni : 1.0;
            }
            out->reap_spearman_mean = rsp / L;
            out->reap_bottom_jaccard = rjac / L;
            free(td);
        }
    }

    free(tmp); free(ra); free(rb); free(ina); free(inb);
    return 0;
}
