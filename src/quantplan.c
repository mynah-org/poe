/* quantplan.c — per-slab mixed-precision allocation with exact byte
 * accounting.
 * SPDX-License-Identifier: MIT */
#include "poe/quantplan.h"
#include "poe/stats.h"
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POE_QUANTPLAN_VERSION 0

/* The candidate ladder, cheapest first, and what each type costs in
 * relative L2 on a bell-shaped block — the shape a weight matrix actually
 * has. These are ingot's own round-trip measurements (tests/test_quant.c),
 * not a bits-per-weight model: they are what OUR encoder delivers, and the
 * ladder halves cleanly per bit, which is the sanity check that they are
 * measuring the format rather than a bug.
 *
 * Two honest limits. They are unweighted fits, so an imatrix-weighted
 * encoder would shift the low end down; and they are one distribution, so
 * they rank types correctly but should not be read as a prediction of KLD
 * on a real model. The allocator only needs the ranking and the ratios. */
static const int    ladder_type[]  = { INGOT_TYPE_Q2_K, INGOT_TYPE_Q3_K,
                                       INGOT_TYPE_Q4_K, INGOT_TYPE_Q5_K,
                                       INGOT_TYPE_Q6_K };
static const double ladder_error[] = { 0.2682, 0.1615, 0.0724, 0.0366, 0.0179 };
#define LADDER_N ((int)(sizeof ladder_type / sizeof *ladder_type))

size_t poe_quantplan_ladder(const int **types, const double **errors) {
    if (types)  *types  = ladder_type;
    if (errors) *errors = ladder_error;
    return LADDER_N;
}

static const char *proj_suffix[POE_QSLAB_NPROJ] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

static void warn(poe_quantplan *p, const char *fmt, ...) {
    if (p->n_warnings >= POE_QUANT_MAX_WARN) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->warnings[p->n_warnings], sizeof p->warnings[0], fmt, ap);
    va_end(ap);
    p->n_warnings++;
}

static const ingot_tensor *slab_of(const poe_block *b, poe_qslab proj) {
    switch (proj) {
    case POE_QSLAB_GATE: return b->gate_exps_w;
    case POE_QSLAB_UP:   return b->up_exps_w;
    case POE_QSLAB_DOWN: return b->down_exps_w;
    default:             return NULL;
    }
}

/* Exact bytes for `nelem` elements in `type`, or 0 when the type has no
 * known geometry — never an estimate. */
static uint64_t type_bytes(int type, uint64_t nelem) {
    uint64_t out = 0;
    if (ingot_type_nbytes(type, nelem, &out) != 0) return 0;
    return out;
}

void poe_quantplan_free(poe_quantplan *p) {
    if (p == NULL) return;
    free(p->type);
    free(p->type_before);
    free(p->bytes_before);
    free(p->bytes_after);
    free(p->nelem);
    free(p->layer_score);
    free(p);
}

int poe_quantplan_build(poe_quantplan **out, const poe_model *m,
                        const poe_profile *profile, uint64_t target_bytes,
                        const int *types, size_t n_types, int force,
                        char *err, size_t errsz) {
    return poe_quantplan_build_ex(out, m, profile, target_bytes, types,
                                  n_types, force, 0, err, errsz);
}

int poe_quantplan_build_ex(poe_quantplan **out, const poe_model *m,
                           const poe_profile *profile, uint64_t target_bytes,
                           const int *types, size_t n_types, int force,
                           int invert, char *err, size_t errsz) {
    const poe_quantplan_opts o = {
        .profile = profile, .layer_scores = NULL, .score_label = NULL,
        .target_bytes = target_bytes, .types = types, .n_types = n_types,
        .force = force, .invert = invert
    };
    return poe_quantplan_build_opts(out, m, &o, err, errsz);
}

int poe_quantplan_build_opts(poe_quantplan **out, const poe_model *m,
                             const poe_quantplan_opts *o,
                             char *err, size_t errsz) {
    if (o == NULL) {
        if (err) snprintf(err, errsz, "null argument");
        return -1;
    }
    const poe_profile *profile = o->profile;
    const uint64_t target_bytes = o->target_bytes;
    const int *types = o->types;
    const size_t n_types = o->n_types;
    const int force = o->force, invert = o->invert;

    if (out == NULL || m == NULL) {
        if (err) snprintf(err, errsz, "null argument");
        return -1;
    }
    *out = NULL;
    if (m->n_moe_blocks == 0) {
        if (err) snprintf(err, errsz, "%s has no MoE blocks", m->arch);
        return -1;
    }
    if (profile != NULL && strcmp(profile->fingerprint, m->fingerprint) != 0 &&
        !force) {
        if (err) snprintf(err, errsz,
                          "profile is for a different checkpoint (%s vs %s); "
                          "pass --force to override",
                          profile->fingerprint, m->fingerprint);
        return -1;
    }

    /* Restrict the ladder if asked, keeping it cheapest-first. */
    int    cand[LADDER_N];
    double cand_err[LADDER_N];
    int    n_cand = 0;
    for (int i = 0; i < LADDER_N; i++) {
        int wanted = (types == NULL);
        for (size_t j = 0; !wanted && j < n_types; j++)
            wanted = types[j] == ladder_type[i];
        if (wanted) { cand[n_cand] = ladder_type[i];
                      cand_err[n_cand++] = ladder_error[i]; }
    }
    if (n_cand == 0) {
        if (err) snprintf(err, errsz, "no candidate quantization types");
        return -1;
    }

    poe_quantplan *p = calloc(1, sizeof *p);
    if (p == NULL) { if (err) snprintf(err, errsz, "out of memory"); return -1; }
    p->version = POE_QUANTPLAN_VERSION;
    snprintf(p->poe_version, sizeof p->poe_version, "%s", POE_VERSION);
    snprintf(p->model_fingerprint, sizeof p->model_fingerprint, "%s", m->fingerprint);
    snprintf(p->arch, sizeof p->arch, "%s", m->arch);
    snprintf(p->method, sizeof p->method, "%s",
             o->layer_scores != NULL
                 ? (o->score_label ? o->score_label : "external")
                 : profile != NULL ? "saliency" : "uniform");
    p->n_layers = m->n_blocks;
    p->n_experts = m->expert_count;
    p->target_bytes = target_bytes;
    p->model_bytes_before = m->total_bytes;

    const size_t n_slabs = (size_t)p->n_layers * POE_QSLAB_NPROJ;
    p->type         = calloc(n_slabs, sizeof *p->type);
    p->type_before  = calloc(n_slabs, sizeof *p->type_before);
    p->bytes_before = calloc(n_slabs, sizeof *p->bytes_before);
    p->bytes_after  = calloc(n_slabs, sizeof *p->bytes_after);
    p->nelem        = calloc(n_slabs, sizeof *p->nelem);
    p->layer_score  = calloc(p->n_layers, sizeof *p->layer_score);
    if (!p->type || !p->type_before || !p->bytes_before || !p->bytes_after ||
        !p->nelem || !p->layer_score) {
        poe_quantplan_free(p);
        if (err) snprintf(err, errsz, "out of memory");
        return -1;
    }

    /* Per-layer saliency: the summed REAP score of the layer's experts,
     * min-max normalized across layers.
     *
     * There is deliberately NO fallback to selection counts. Summed over a
     * layer's experts they are tokens x top_k in EVERY layer — a conserved
     * quantity, identical by construction, so ranking layers by it is not a
     * weak signal but a guaranteed tie. Saying so is the useful behaviour;
     * silently returning a uniform plan labelled "saliency" is not. */
    double lo = 0.0, hi = 0.0;
    int have_scores = 0;
    if (o->layer_scores != NULL) {
        /* An external ranking (an imatrix statistic, say). The allocator
         * does not care where a score came from, only that it is per layer
         * and comparable across layers. */
        for (uint32_t l = 0; l < p->n_layers; l++) {
            const double s = o->layer_scores[l];
            p->layer_score[l] = s;
            if (l == 0 || s < lo) lo = s;
            if (l == 0 || s > hi) hi = s;
        }
        have_scores = hi > lo;
        if (!have_scores) {
            warn(p, "every layer has the same score; allocation degenerates "
                    "to uniform");
            snprintf(p->method, sizeof p->method, "uniform");
        }
    } else if (profile == NULL) {
        /* nothing to say: uniform is what was asked for */
    } else if (profile->n_layers != p->n_layers ||
               profile->n_experts != p->n_experts) {
        warn(p, "profile shape does not match the model; ignoring it");
        snprintf(p->method, sizeof p->method, "uniform");
    } else if (profile->reap_mean == NULL) {
        warn(p, "profile has no REAP data, so layers cannot be ranked "
                "(selection counts sum to tokens x top_k in every layer); "
                "re-profile with --metric reap or accept uniform");
        snprintf(p->method, sizeof p->method, "uniform");
    } else {
        for (uint32_t l = 0; l < p->n_layers; l++) {
            double s = 0.0;
            for (uint32_t e = 0; e < p->n_experts; e++)
                s += profile->reap_mean[(size_t)l * p->n_experts + e];
            p->layer_score[l] = s;
            if (l == 0 || s < lo) lo = s;
            if (l == 0 || s > hi) hi = s;
        }
        have_scores = hi > lo;
        if (!have_scores) {
            warn(p, "every layer has the same REAP total; allocation "
                    "degenerates to uniform");
            snprintf(p->method, sizeof p->method, "uniform");
        }
    }
    for (uint32_t l = 0; l < p->n_layers; l++)
        p->layer_score[l] = have_scores
            ? (p->layer_score[l] - lo) / (hi - lo) : 1.0;
    if (invert && have_scores) {
        for (uint32_t l = 0; l < p->n_layers; l++)
            p->layer_score[l] = 1.0 - p->layer_score[l];
        snprintf(p->method, sizeof p->method, "inverted");
        warn(p, "scores are INVERTED: this is a deliberately wrong control "
                "allocation, not a plan to ship");
    } else if (invert) {
        warn(p, "--invert had nothing to invert: no usable layer scores");
    }

    /* The depth check. Per-layer REAP totals on Qwen3.6 rise monotonically
     * with depth because the residual stream's norm does, and an allocation
     * driven by that ranking lost to uniform — as did its inverse, which is
     * how we know the ranking carried no direction. Any score that is
     * essentially the layer index has that same problem, whatever it was
     * derived from, so the plan says so rather than leaving it to be
     * discovered by a measurement. */
    p->depth_rho = have_scores ? poe_depth_rho(p->layer_score, p->n_layers)
                               : 0.0;
    if (have_scores && (p->depth_rho >= 0.9 || p->depth_rho <= -0.9))
        warn(p, "layer scores are near-monotone in depth (Spearman %+.2f): "
                "both allocations that lost to uniform had this shape — "
                "measure against uniform, and carry --invert as a control",
             p->depth_rho);

    /* Every routed slab starts at the cheapest candidate. */
    uint64_t floor_bytes = 0;
    uint32_t n_present = 0;
    for (uint32_t l = 0; l < p->n_layers; l++) {
        for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
            const size_t i = (size_t)l * POE_QSLAB_NPROJ + pr;
            const ingot_tensor *t = slab_of(&m->blocks[l], (poe_qslab)pr);
            if (t == NULL) { p->type[i] = -1; p->type_before[i] = -1; continue; }
            p->nelem[i] = t->nelem;
            p->type_before[i] = t->type;
            p->bytes_before[i] = t->nbytes;
            p->bytes_before_total += t->nbytes;
            p->type[i] = cand[0];
            p->bytes_after[i] = type_bytes(cand[0], t->nelem);
            if (p->bytes_after[i] == 0) {
                poe_quantplan_free(p);
                if (err) snprintf(err, errsz,
                                  "cannot size %s as %s (bad geometry)",
                                  t->name, ingot_type_name(cand[0]));
                return -1;
            }
            floor_bytes += p->bytes_after[i];
            n_present++;
        }
    }
    if (n_present == 0) {
        poe_quantplan_free(p);
        if (err) snprintf(err, errsz, "no packed routed expert tensors found");
        return -1;
    }
    if (target_bytes < floor_bytes) {
        char want[32], got[32];
        poe_format_bytes(target_bytes, want, sizeof want);
        poe_format_bytes(floor_bytes, got, sizeof got);
        warn(p, "target %s is below the %s floor of all-%s; using the floor",
             want, got, ingot_type_name(cand[0]));
    }

    /* Greedy upgrades: repeatedly take the step that buys the most quality
     * per extra byte and still fits. Quality is the layer's saliency times
     * the slab's size times the error the step removes, so a big slab in a
     * salient layer outranks a small one in a cold layer — which is the
     * whole point of spending a fixed budget unevenly.
     *
     * A greedy pass is not the LP the GEMQ line of work solves, and with a
     * ladder this coarse it is very close to it: every item has the same
     * five rungs and the same convex error curve, so the ratio ordering
     * barely changes as the budget is spent. Worth revisiting only if the
     * ladder gains rungs. */
    int *rung = calloc(n_slabs, sizeof *rung);
    if (rung == NULL) {
        poe_quantplan_free(p);
        if (err) snprintf(err, errsz, "out of memory");
        return -1;
    }
    uint64_t spent = floor_bytes;
    for (;;) {
        double best_ratio = 0.0;
        size_t best_i = 0;
        uint64_t best_bytes = 0;
        int found = 0;
        for (size_t i = 0; i < n_slabs; i++) {
            if (p->type[i] < 0 || rung[i] + 1 >= n_cand) continue;
            const uint64_t next = type_bytes(cand[rung[i] + 1], p->nelem[i]);
            if (next == 0 || next <= p->bytes_after[i]) continue;
            const uint64_t extra = next - p->bytes_after[i];
            if (spent + extra > target_bytes) continue;
            const double gain = (cand_err[rung[i]] - cand_err[rung[i] + 1]) *
                                (double)p->nelem[i] *
                                p->layer_score[i / POE_QSLAB_NPROJ];
            const double ratio = gain / (double)extra;
            if (!found || ratio > best_ratio) {
                found = 1; best_ratio = ratio; best_i = i; best_bytes = next;
            }
        }
        if (!found) break;
        spent += best_bytes - p->bytes_after[best_i];
        p->bytes_after[best_i] = best_bytes;
        rung[best_i]++;
        p->type[best_i] = cand[rung[best_i]];
    }
    /* Distinguish "the budget ran out" from "the ladder ran out" — the
     * second silently returns a plan far under target, which reads as a bug
     * unless it is said out loud. */
    int all_maxed = 1;
    for (size_t i = 0; i < n_slabs && all_maxed; i++)
        if (p->type[i] >= 0 && rung[i] + 1 < n_cand) all_maxed = 0;
    free(rung);
    p->bytes_after_total = spent;

    if (all_maxed && spent < target_bytes) {
        char left[32];
        poe_format_bytes(target_bytes - spent, left, sizeof left);
        warn(p, "every slab is at %s, the top of the candidate ladder; "
                "%s of the budget is unspent",
             ingot_type_name(cand[n_cand - 1]), left);
    }

    if (p->bytes_after_total > p->bytes_before_total)
        warn(p, "the plan is larger than the source slabs; the target asks "
                "for more precision than the checkpoint carries");

    *out = p;
    return 0;
}

/* ── artifacts ──────────────────────────────────────────────────────────── */

int poe_quantplan_write(const poe_quantplan *p, const char *path,
                        char *err, size_t errsz) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        if (err) snprintf(err, errsz, "cannot write '%s'", path);
        return -1;
    }
    fprintf(f, "{\n  \"poequant\": %u,\n", p->version);
    fprintf(f, "  \"poe_version\": \"%s\",\n", p->poe_version);
    fprintf(f, "  \"model_fingerprint\": \"%s\",\n", p->model_fingerprint);
    fprintf(f, "  \"arch\": \"%s\",\n", p->arch);
    fprintf(f, "  \"method\": \"%s\",\n", p->method);
    fprintf(f, "  \"depth_rho\": %.6f,\n", p->depth_rho);
    fprintf(f, "  \"n_layers\": %u, \"n_experts\": %u,\n",
            p->n_layers, p->n_experts);
    fprintf(f, "  \"expected\": {\n");
    fprintf(f, "    \"target_bytes\": %llu,\n",
            (unsigned long long)p->target_bytes);
    fprintf(f, "    \"slab_bytes_before\": %llu,\n",
            (unsigned long long)p->bytes_before_total);
    fprintf(f, "    \"slab_bytes_after\": %llu,\n",
            (unsigned long long)p->bytes_after_total);
    fprintf(f, "    \"model_bytes_before\": %llu\n  },\n",
            (unsigned long long)p->model_bytes_before);
    fprintf(f, "  \"warnings\": [");
    for (uint32_t i = 0; i < p->n_warnings; i++)
        fprintf(f, "%s\n    \"%s\"", i ? "," : "", p->warnings[i]);
    fprintf(f, "%s],\n", p->n_warnings ? "\n  " : "");
    fprintf(f, "  \"layers\": [");
    for (uint32_t l = 0; l < p->n_layers; l++) {
        const size_t base = (size_t)l * POE_QSLAB_NPROJ;
        if (p->type[base] < 0 && p->type[base + 1] < 0 && p->type[base + 2] < 0)
            continue;
        fprintf(f, "%s\n    {\"layer\": %u, \"score\": %.6f",
                l ? "," : "", l, p->layer_score[l]);
        for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
            const size_t i = base + pr;
            if (p->type[i] < 0) continue;
            fprintf(f, ", \"%s\": {\"from\": \"%s\", \"to\": \"%s\", "
                       "\"bytes_before\": %llu, \"bytes_after\": %llu}",
                    proj_suffix[pr], ingot_type_name(p->type_before[i]),
                    ingot_type_name(p->type[i]),
                    (unsigned long long)p->bytes_before[i],
                    (unsigned long long)p->bytes_after[i]);
        }
        fprintf(f, "}");
    }
    fprintf(f, "\n  ]\n}\n");
    if (ferror(f)) { fclose(f); if (err) snprintf(err, errsz, "write failed"); return -1; }
    fclose(f);
    return 0;
}

/* Type ids are ggml's, so the name is the only stable key a file can carry.
 * The scan is over the small id space ggml uses; an unknown id yields a name
 * that matches nothing, which is the right answer. */
static int type_from_name(const char *name) {
    if (name == NULL || *name == '\0') return -1;
    for (int t = 0; t < 64; t++) {
        const char *n = ingot_type_name(t);
        if (n != NULL && strcmp(n, name) == 0) return t;
    }
    return -1;
}

static int qfail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

int poe_quantplan_load(poe_quantplan **out, const char *path,
                       char *err, size_t errsz) {
    if (out == NULL || path == NULL) return qfail(err, errsz, "null argument");
    *out = NULL;

    FILE *f = fopen(path, "rb");
    if (f == NULL) return qfail(err, errsz, "cannot open the plan");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return qfail(err, errsz, "empty plan"); }
    char *text = malloc((size_t)sz);
    if (text == NULL || fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        free(text); fclose(f);
        return qfail(err, errsz, "cannot read the plan");
    }
    fclose(f);

    poe_json *j = poe_json_parse(text, (size_t)sz, err, errsz);
    free(text);
    if (j == NULL) return -1;

    if (poe_json_get(j, "poequant") == NULL) {
        poe_json_free(j);
        return qfail(err, errsz, "not a .poequant (missing \"poequant\" key)");
    }

    poe_quantplan *p = calloc(1, sizeof *p);
    if (p == NULL) { poe_json_free(j); return qfail(err, errsz, "out of memory"); }
    p->version = (uint32_t)poe_json_u64(poe_json_get(j, "poequant"), 0);
    snprintf(p->poe_version, sizeof p->poe_version, "%s",
             poe_json_str(poe_json_get(j, "poe_version"), ""));
    snprintf(p->model_fingerprint, sizeof p->model_fingerprint, "%s",
             poe_json_str(poe_json_get(j, "model_fingerprint"), ""));
    snprintf(p->arch, sizeof p->arch, "%s",
             poe_json_str(poe_json_get(j, "arch"), ""));
    snprintf(p->method, sizeof p->method, "%s",
             poe_json_str(poe_json_get(j, "method"), ""));
    p->depth_rho = poe_json_num(poe_json_get(j, "depth_rho"), 0.0);
    p->n_layers  = (uint32_t)poe_json_u64(poe_json_get(j, "n_layers"), 0);
    p->n_experts = (uint32_t)poe_json_u64(poe_json_get(j, "n_experts"), 0);

    const poe_json *ex = poe_json_get(j, "expected");
    p->target_bytes       = poe_json_u64(poe_json_get(ex, "target_bytes"), 0);
    p->bytes_before_total = poe_json_u64(poe_json_get(ex, "slab_bytes_before"), 0);
    p->bytes_after_total  = poe_json_u64(poe_json_get(ex, "slab_bytes_after"), 0);
    p->model_bytes_before = poe_json_u64(poe_json_get(ex, "model_bytes_before"), 0);

    const poe_json *warns = poe_json_get(j, "warnings");
    for (size_t i = 0; i < poe_json_len(warns) && i < POE_QUANT_MAX_WARN; i++)
        snprintf(p->warnings[p->n_warnings++], sizeof p->warnings[0], "%s",
                 poe_json_str(poe_json_at(warns, i), ""));

    const poe_json *layers = poe_json_get(j, "layers");
    if (p->n_layers == 0 || poe_json_len(layers) == 0 ||
        poe_json_len(layers) > p->n_layers) {
        poe_json_free(j); poe_quantplan_free(p);
        return qfail(err, errsz, "plan shape is inconsistent");
    }

    const size_t n_slabs = (size_t)p->n_layers * POE_QSLAB_NPROJ;
    p->type         = calloc(n_slabs, sizeof *p->type);
    p->type_before  = calloc(n_slabs, sizeof *p->type_before);
    p->bytes_before = calloc(n_slabs, sizeof *p->bytes_before);
    p->bytes_after  = calloc(n_slabs, sizeof *p->bytes_after);
    p->nelem        = calloc(n_slabs, sizeof *p->nelem);
    p->layer_score  = calloc(p->n_layers, sizeof *p->layer_score);
    if (!p->type || !p->type_before || !p->bytes_before || !p->bytes_after ||
        !p->nelem || !p->layer_score) {
        poe_json_free(j); poe_quantplan_free(p);
        return qfail(err, errsz, "out of memory");
    }
    for (size_t i = 0; i < n_slabs; i++) { p->type[i] = -1; p->type_before[i] = -1; }

    for (size_t i = 0; i < poe_json_len(layers); i++) {
        const poe_json *lj = poe_json_at(layers, i);
        const uint32_t l = (uint32_t)poe_json_u64(poe_json_get(lj, "layer"),
                                                  p->n_layers);
        if (l >= p->n_layers) continue;
        p->layer_score[l] = poe_json_num(poe_json_get(lj, "score"), 1.0);
        for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
            const poe_json *s = poe_json_get(lj, proj_suffix[pr]);
            if (s == NULL) continue;
            const size_t k = (size_t)l * POE_QSLAB_NPROJ + pr;
            p->type[k]         = type_from_name(poe_json_str(poe_json_get(s, "to"), ""));
            p->type_before[k]  = type_from_name(poe_json_str(poe_json_get(s, "from"), ""));
            p->bytes_before[k] = poe_json_u64(poe_json_get(s, "bytes_before"), 0);
            p->bytes_after[k]  = poe_json_u64(poe_json_get(s, "bytes_after"), 0);
        }
    }

    poe_json_free(j);
    *out = p;
    return 0;
}

int poe_quantplan_write_tensor_types(const poe_quantplan *p, const char *path,
                                     char *err, size_t errsz) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        if (err) snprintf(err, errsz, "cannot write '%s'", path);
        return -1;
    }
    /* llama-quantize compiles each line as a std::regex and matches it with
     * regex_search against the full tensor name, so the dots are escaped and
     * the name is anchored at both ends: an unescaped "blk.1." would also
     * match blk.11, blk.12, ... and silently re-type a dozen layers. */
    for (uint32_t l = 0; l < p->n_layers; l++) {
        for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
            const size_t i = (size_t)l * POE_QSLAB_NPROJ + pr;
            if (p->type[i] < 0) continue;
            char escaped[96];
            size_t k = 0;
            const char *s = proj_suffix[pr];
            for (size_t j = 0; s[j] && k + 2 < sizeof escaped; j++) {
                if (s[j] == '.') escaped[k++] = '\\';
                escaped[k++] = s[j];
            }
            escaped[k] = '\0';
            fprintf(f, "^blk\\.%u\\.%s$=%s\n", l, escaped,
                    ingot_type_name(p->type[i]));
        }
    }
    if (ferror(f)) { fclose(f); if (err) snprintf(err, errsz, "write failed"); return -1; }
    fclose(f);
    return 0;
}
