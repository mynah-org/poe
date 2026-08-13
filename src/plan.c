/* plan.c — building, writing and loading .poeplan files.
 * SPDX-License-Identifier: MIT */
#include "poe/plan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/super.h"

#include "json.h"

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

static void warn(poe_plan *p, const char *msg) {
    if (p->n_warnings < POE_PLAN_MAX_WARN)
        snprintf(p->warnings[p->n_warnings++], sizeof p->warnings[0],
                 "%s", msg);
}

void poe_plan_free(poe_plan *p) {
    if (p == NULL) return;
    free(p->keep);
    free(p);
}

/* ── build ──────────────────────────────────────────────────────────────── */

typedef struct { double v; uint32_t idx; } ent;

static int cmp_asc(const void *x, const void *y) {
    const ent *a = x, *b = y;
    if (a->v != b->v) return a->v > b->v ? 1 : -1;
    return a->idx < b->idx ? -1 : 1;      /* deterministic ties */
}

/* per-layer score of one profile under `method`, or -1 if unavailable */
static int layer_scores(const poe_profile *pr, const char *method,
                        uint32_t l, double *out) {
    const uint32_t E = pr->n_experts;
    if (strcmp(method, "reap") == 0) {
        if (pr->reap_mean == NULL) return -1;
        for (uint32_t e = 0; e < E; e++)
            out[e] = pr->reap_mean[(size_t)l * E + e];
    } else if (strcmp(method, "frequency") == 0) {
        for (uint32_t e = 0; e < E; e++)
            out[e] = (double)pr->sel_count[(size_t)l * E + e];
    } else if (strcmp(method, "gate") == 0) {
        for (uint32_t e = 0; e < E; e++)
            out[e] = pr->gate_mean[(size_t)l * E + e];
    } else {
        return -1;
    }
    return 0;
}

int poe_plan_build(poe_plan **out, const poe_model *m,
                   const poe_profile *const *profiles, const double *weights,
                   size_t n_profiles, const char *method, double prune_frac,
                   int force, char *err, size_t errsz) {
    const poe_plan_opts o = {
        .profiles = profiles, .weights = weights, .n_profiles = n_profiles,
        .method = method, .prune_frac = prune_frac, .force = force,
        .protect_super = 0, .super_z = 0
    };
    return poe_plan_build_opts(out, m, &o, err, errsz);
}

int poe_plan_build_opts(poe_plan **out, const poe_model *m,
                        const poe_plan_opts *o, char *err, size_t errsz) {
    if (out == NULL || o == NULL) return -1;
    *out = NULL;

    const poe_profile *const *profiles = o->profiles;
    const double *weights = o->weights;
    const size_t  n_profiles = o->n_profiles;
    const char   *method = o->method;
    const double  prune_frac = o->prune_frac;
    const int     force = o->force;

    if (n_profiles == 0) return fail(err, errsz, "at least one profile needed");
    if (prune_frac <= 0.0 || prune_frac >= 1.0)
        return fail(err, errsz, "prune fraction must be in (0,1)");
    if (m->n_moe_blocks == 0)
        return fail(err, errsz, "model has no MoE structure");

    const uint32_t L = m->n_blocks, E = m->expert_count;

    poe_plan *p = calloc(1, sizeof *p);
    if (p == NULL) return fail(err, errsz, "out of memory");
    p->version = 0;
    snprintf(p->poe_version, sizeof p->poe_version, "%s", POE_VERSION);
    snprintf(p->model_fingerprint, sizeof p->model_fingerprint, "%s",
             m->fingerprint);
    snprintf(p->arch, sizeof p->arch, "%s", m->arch);
    snprintf(p->method, sizeof p->method, "%s", method);
    p->prune_fraction = prune_frac;
    p->n_layers  = L;
    p->n_experts = E;
    p->top_k     = m->experts_per_token;

    for (size_t i = 0; i < n_profiles; i++) {
        const poe_profile *pr = profiles[i];
        if (pr->n_layers != L || pr->n_experts != E) {
            poe_plan_free(p);
            return fail(err, errsz, "profile shape does not match the model");
        }
        if (strcmp(pr->fingerprint, m->fingerprint) != 0) {
            if (!force) {
                poe_plan_free(p);
                return fail(err, errsz,
                            "profile fingerprint does not match the model "
                            "(use --force to override)");
            }
            warn(p, "profile was captured from a different model/quantization");
        }
        if (pr->tokens < 8192)
            warn(p, "calibration used fewer than 8192 tokens — ranking may be unstable");
    }

    /* selection sizes: keep at least top_k experts per layer */
    uint32_t prune_n = (uint32_t)((double)E * prune_frac + 0.5);
    uint32_t min_keep = p->top_k ? p->top_k : 1;
    if (prune_n >= E) prune_n = E - 1;
    if (E - prune_n < min_keep) {
        prune_n = E - min_keep;
        warn(p, "prune fraction clamped so top_k experts survive per layer");
    }
    if (prune_frac > 0.5)
        warn(p, "pruning beyond 50% is outside the range validated by REAP");
    p->keep_per_layer = E - prune_n;

    p->keep = malloc((size_t)L * E);
    if (p->keep == NULL) { poe_plan_free(p); return fail(err, errsz, "out of memory"); }
    memset(p->keep, 1, (size_t)L * E);

    /* Super-expert protection: detected once, over the first profile that
     * carries activation norms. Asking for it and not getting it is a
     * warning, not a silent no-op — the guard exists against a failure that
     * leaves the model fluent. */
    poe_super *sup = NULL;
    p->protect_super = o->protect_super;
    if (o->protect_super) {
        char serr[256] = { 0 };
        for (size_t i = 0; i < n_profiles && sup == NULL; i++)
            poe_super_detect(&sup, profiles[i], o->super_z, serr, sizeof serr);
        if (sup == NULL) {
            warn(p, "super-expert protection asked for but no profile carries "
                    "activation norms (--metric reap): NOT protected");
            p->protect_super = 0;
        } else {
            p->n_super_flagged = (uint32_t)sup->n_outliers;
            p->n_super_rare    = (uint32_t)sup->n_rare_outliers;
            if (sup->n_layers_undecidable) {
                char msg[160];
                snprintf(msg, sizeof msg, "%u layers have a flat activation "
                         "profile: no outlier can be decided there",
                         sup->n_layers_undecidable);
                warn(p, msg);
            }
        }
    }

    double *score = malloc(E * sizeof *score);
    double *one   = malloc(E * sizeof *one);
    ent    *order = malloc(E * sizeof *order);
    if (!score || !one || !order) {
        free(score); free(one); free(order);
        poe_super_free(sup);
        poe_plan_free(p);
        return fail(err, errsz, "out of memory");
    }

    for (uint32_t l = 0; l < L; l++) {
        /* combined score: min-max normalized per profile, weighted sum */
        memset(score, 0, E * sizeof *score);
        for (size_t i = 0; i < n_profiles; i++) {
            if (layer_scores(profiles[i], method, l, one) != 0) {
                free(score); free(one); free(order);
                poe_super_free(sup);
                poe_plan_free(p);
                return fail(err, errsz, strcmp(method, "reap") == 0
                            ? "method 'reap' needs profiles captured with --metric reap"
                            : "unknown method (reap|frequency|gate)");
            }
            double lo = one[0], hi = one[0];
            for (uint32_t e = 1; e < E; e++) {
                if (one[e] < lo) lo = one[e];
                if (one[e] > hi) hi = one[e];
            }
            double span = hi > lo ? hi - lo : 1.0;
            double w = weights ? weights[i] : 1.0;
            for (uint32_t e = 0; e < E; e++)
                score[e] += w * (one[e] - lo) / span;
        }

        for (uint32_t e = 0; e < E; e++) { order[e].v = score[e]; order[e].idx = e; }
        qsort(order, E, sizeof *order, cmp_asc);

        if (sup == NULL) {
            for (uint32_t i = 0; i < prune_n; i++)
                p->keep[(size_t)l * E + order[i].idx] = 0;
            continue;
        }

        /* The unprotected cut is exactly order[0 .. prune_n-1]; anything
         * flagged inside it is what protection is actually buying. */
        const uint8_t *flag = sup->is_outlier + (size_t)l * E;
        for (uint32_t i = 0; i < prune_n; i++)
            if (flag[order[i].idx]) p->n_super_rescued++;

        /* Prune the same number of experts, skipping the flagged ones: the
         * per-layer keep count has to stay uniform, because poe apply
         * rejects a plan where it is not. */
        uint32_t pruned = 0;
        for (uint32_t i = 0; i < E && pruned < prune_n; i++) {
            if (flag[order[i].idx]) continue;
            p->keep[(size_t)l * E + order[i].idx] = 0;
            pruned++;
        }
        /* Only reachable when a layer holds more flagged experts than the
         * cut can spare. Take the least salient of them and say so. */
        for (uint32_t i = 0; i < E && pruned < prune_n; i++) {
            if (!p->keep[(size_t)l * E + order[i].idx]) continue;
            p->keep[(size_t)l * E + order[i].idx] = 0;
            p->n_super_still_pruned++;
            if (i < prune_n) p->n_super_rescued--;   /* it was not rescued */
            pruned++;
        }
    }
    free(score); free(one); free(order);
    if (p->n_super_still_pruned) {
        char msg[160];
        snprintf(msg, sizeof msg, "%u flagged experts had to be pruned anyway: "
                 "a layer holds more of them than this cut can spare",
                 p->n_super_still_pruned);
        warn(p, msg);
    }
    poe_super_free(sup);

    /* exact accounting from the model: pruned expert slices + router rows */
    p->bytes_before  = m->total_bytes;
    p->params_before = m->total_params;
    for (uint32_t l = 0; l < L; l++) {
        const poe_block *blk = &m->blocks[l];
        if (!blk->is_moe || blk->expert_count == 0) continue;
        if (blk->expert_count != E) {
            warn(p, "a block has a different expert count — accounting is approximate there");
            continue;
        }
        if (blk->expert_bytes % E || blk->expert_params % E)
            warn(p, "expert tensor bytes are not divisible per expert — accounting rounded down");
        uint64_t row_b = (blk->router_w ? blk->router_w->nbytes / E : 0) +
                         (blk->router_b ? blk->router_b->nbytes / E : 0);
        uint64_t row_p = (blk->router_w ? blk->router_w->nelem  / E : 0) +
                         (blk->router_b ? blk->router_b->nelem  / E : 0);
        p->bytes_removed  += (uint64_t)prune_n * (blk->expert_bytes  / E + row_b);
        p->params_removed += (uint64_t)prune_n * (blk->expert_params / E + row_p);
    }

    *out = p;
    return 0;
}

/* ── write ──────────────────────────────────────────────────────────────── */

int poe_plan_write(const poe_plan *p, const char *path,
                   char *err, size_t errsz) {
    FILE *f = fopen(path, "w");
    if (f == NULL) return fail(err, errsz, "cannot write plan file");

    fprintf(f, "{\n  \"poeplan\": %u,\n", p->version);
    fprintf(f, "  \"poe_version\": \"%s\",\n", p->poe_version);
    fprintf(f, "  \"model_fingerprint\": \"%s\",\n", p->model_fingerprint);
    fprintf(f, "  \"arch\": \"%s\",\n", p->arch);
    fprintf(f, "  \"method\": \"%s\",\n", p->method);
    fprintf(f, "  \"prune_fraction\": %.6f,\n", p->prune_fraction);
    fprintf(f, "  \"n_layers\": %u, \"n_experts\": %u, \"top_k\": %u,\n",
            p->n_layers, p->n_experts, p->top_k);
    fprintf(f, "  \"keep_per_layer\": %u,\n", p->keep_per_layer);
    fprintf(f, "  \"super_experts\": {\n");
    fprintf(f, "    \"protected\": %s,\n", p->protect_super ? "true" : "false");
    fprintf(f, "    \"flagged\": %u,\n", p->n_super_flagged);
    fprintf(f, "    \"rare\": %u,\n", p->n_super_rare);
    fprintf(f, "    \"rescued\": %u,\n", p->n_super_rescued);
    fprintf(f, "    \"pruned_anyway\": %u\n", p->n_super_still_pruned);
    fprintf(f, "  },\n");
    fprintf(f, "  \"expected\": {\n");
    fprintf(f, "    \"bytes_before\": %llu,\n", (unsigned long long)p->bytes_before);
    fprintf(f, "    \"bytes_removed\": %llu,\n", (unsigned long long)p->bytes_removed);
    fprintf(f, "    \"bytes_after\": %llu,\n",
            (unsigned long long)(p->bytes_before - p->bytes_removed));
    fprintf(f, "    \"params_before\": %llu,\n", (unsigned long long)p->params_before);
    fprintf(f, "    \"params_removed\": %llu\n", (unsigned long long)p->params_removed);
    fprintf(f, "  },\n");
    fprintf(f, "  \"warnings\": [");
    for (uint32_t i = 0; i < p->n_warnings; i++)
        fprintf(f, "%s\n    \"%s\"", i ? "," : "", p->warnings[i]);
    fprintf(f, "%s],\n", p->n_warnings ? "\n  " : "");

    fprintf(f, "  \"layers\": [\n");
    for (uint32_t l = 0; l < p->n_layers; l++) {
        const uint8_t *k = p->keep + (size_t)l * p->n_experts;
        fprintf(f, "    {\"layer\": %u,\n     \"keep\": [", l);
        int first = 1;
        for (uint32_t e = 0; e < p->n_experts; e++)
            if (k[e]) { fprintf(f, "%s%u", first ? "" : ",", e); first = 0; }
        fprintf(f, "],\n     \"prune\": [");
        first = 1;
        for (uint32_t e = 0; e < p->n_experts; e++)
            if (!k[e]) { fprintf(f, "%s%u", first ? "" : ",", e); first = 0; }
        fprintf(f, "]}%s\n", l + 1 < p->n_layers ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    return 0;
}

/* ── load ───────────────────────────────────────────────────────────────── */

int poe_plan_load(poe_plan **out, const char *path, char *err, size_t errsz) {
    if (out == NULL) return -1;
    *out = NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return fail(err, errsz, "cannot open plan");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return fail(err, errsz, "empty plan"); }
    char *text = malloc((size_t)sz);
    if (text == NULL || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        free(text); fclose(f);
        return fail(err, errsz, "cannot read plan");
    }
    fclose(f);

    poe_json *j = poe_json_parse(text, (size_t)sz, err, errsz);
    free(text);
    if (j == NULL) return -1;

    poe_plan *p = calloc(1, sizeof *p);
    if (p == NULL) { poe_json_free(j); return fail(err, errsz, "out of memory"); }

    const poe_json *v = poe_json_get(j, "poeplan");
    if (v == NULL) {
        poe_json_free(j); free(p);
        return fail(err, errsz, "not a .poeplan (missing \"poeplan\" key)");
    }
    p->version = (uint32_t)poe_json_u64(v, 0);
    snprintf(p->poe_version, sizeof p->poe_version, "%s",
             poe_json_str(poe_json_get(j, "poe_version"), ""));
    snprintf(p->model_fingerprint, sizeof p->model_fingerprint, "%s",
             poe_json_str(poe_json_get(j, "model_fingerprint"), ""));
    snprintf(p->arch, sizeof p->arch, "%s",
             poe_json_str(poe_json_get(j, "arch"), ""));
    snprintf(p->method, sizeof p->method, "%s",
             poe_json_str(poe_json_get(j, "method"), ""));
    p->prune_fraction = poe_json_num(poe_json_get(j, "prune_fraction"), 0.0);
    p->n_layers  = (uint32_t)poe_json_u64(poe_json_get(j, "n_layers"), 0);
    p->n_experts = (uint32_t)poe_json_u64(poe_json_get(j, "n_experts"), 0);
    p->top_k     = (uint32_t)poe_json_u64(poe_json_get(j, "top_k"), 0);
    p->keep_per_layer =
        (uint32_t)poe_json_u64(poe_json_get(j, "keep_per_layer"), 0);
    const poe_json *se = poe_json_get(j, "super_experts");
    if (se != NULL) {
        p->protect_super = poe_json_bool(poe_json_get(se, "protected"), 0);
        p->n_super_flagged     = (uint32_t)poe_json_u64(poe_json_get(se, "flagged"), 0);
        p->n_super_rare        = (uint32_t)poe_json_u64(poe_json_get(se, "rare"), 0);
        p->n_super_rescued     = (uint32_t)poe_json_u64(poe_json_get(se, "rescued"), 0);
        p->n_super_still_pruned =
            (uint32_t)poe_json_u64(poe_json_get(se, "pruned_anyway"), 0);
    }

    const poe_json *ex = poe_json_get(j, "expected");
    p->bytes_before   = poe_json_u64(poe_json_get(ex, "bytes_before"), 0);
    p->bytes_removed  = poe_json_u64(poe_json_get(ex, "bytes_removed"), 0);
    p->params_before  = poe_json_u64(poe_json_get(ex, "params_before"), 0);
    p->params_removed = poe_json_u64(poe_json_get(ex, "params_removed"), 0);

    const poe_json *layers = poe_json_get(j, "layers");
    if (p->n_layers == 0 || p->n_experts == 0 ||
        poe_json_len(layers) != p->n_layers) {
        poe_json_free(j); poe_plan_free(p);
        return fail(err, errsz, "plan shape is inconsistent");
    }

    p->keep = calloc((size_t)p->n_layers * p->n_experts, 1);
    if (p->keep == NULL) {
        poe_json_free(j); poe_plan_free(p);
        return fail(err, errsz, "out of memory");
    }
    for (uint32_t l = 0; l < p->n_layers; l++) {
        const poe_json *lj = poe_json_at(layers, l);
        uint32_t li = (uint32_t)poe_json_u64(poe_json_get(lj, "layer"), l);
        if (li >= p->n_layers) li = l;
        const poe_json *keep = poe_json_get(lj, "keep");
        size_t nk = poe_json_len(keep);
        if (nk != p->keep_per_layer) {
            poe_json_free(j); poe_plan_free(p);
            return fail(err, errsz, "layer keep list has wrong length");
        }
        for (size_t i = 0; i < nk; i++) {
            uint64_t e = poe_json_u64(poe_json_at(keep, i), UINT64_MAX);
            if (e >= p->n_experts) {
                poe_json_free(j); poe_plan_free(p);
                return fail(err, errsz, "keep list contains an invalid expert id");
            }
            p->keep[(size_t)li * p->n_experts + e] = 1;
        }
    }

    poe_json_free(j);
    *out = p;
    return 0;
}
