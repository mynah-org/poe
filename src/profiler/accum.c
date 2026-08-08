/* profiler/accum.c — streaming accumulators for MoE routing observation.
 * SPDX-License-Identifier: MIT */
#include "accum.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const double poe_accum_mass_thresholds[POE_ACCUM_NMASS] = { 0.80, 0.90, 0.95, 0.99 };

int poe_accum_init(poe_accum *a, uint32_t n_layers, uint32_t n_experts,
                   uint32_t top_k) {
    memset(a, 0, sizeof *a);
    a->n_layers  = n_layers;
    a->n_experts = n_experts;
    a->top_k     = top_k;

    size_t le = (size_t)n_layers * n_experts;
    a->sel_count   = calloc(le, sizeof *a->sel_count);
    a->gate_sum    = calloc(le, sizeof *a->gate_sum);
    a->tok_probs   = calloc(n_layers, sizeof *a->tok_probs);
    a->tok_sel     = calloc(n_layers, sizeof *a->tok_sel);
    a->entropy_sum = calloc(n_layers, sizeof *a->entropy_sum);
    int ok = a->sel_count && a->gate_sum && a->tok_probs && a->tok_sel &&
             a->entropy_sum;
    for (int i = 0; i < POE_ACCUM_NMASS; i++) {
        a->mass_k_sum[i] = calloc(n_layers, sizeof *a->mass_k_sum[i]);
        ok = ok && a->mass_k_sum[i];
    }
    if (!ok) { poe_accum_free(a); return -1; }
    return 0;
}

void poe_accum_free(poe_accum *a) {
    free(a->sel_count);
    free(a->gate_sum);
    free(a->tok_probs);
    free(a->tok_sel);
    free(a->entropy_sum);
    for (int i = 0; i < POE_ACCUM_NMASS; i++) free(a->mass_k_sum[i]);
    memset(a, 0, sizeof *a);
}

/* descending float comparator for qsort */
static int cmp_desc(const void *x, const void *y) {
    float fx = *(const float *)x, fy = *(const float *)y;
    return (fx < fy) - (fx > fy);
}

void poe_accum_observe_probs(poe_accum *a, uint32_t layer, uint32_t T,
                             const float *probs, int is_logits) {
    if (layer >= a->n_layers) return;
    const uint32_t E = a->n_experts;

    float *row = malloc(E * sizeof *row);
    if (row == NULL) return;

    for (uint32_t t = 0; t < T; t++) {
        const float *p = probs + (size_t)t * E;

        if (is_logits) {
            /* softmax on host (delayed-softmax archs expose raw scores) */
            float mx = p[0];
            for (uint32_t e = 1; e < E; e++) if (p[e] > mx) mx = p[e];
            double sum = 0.0;
            for (uint32_t e = 0; e < E; e++) {
                row[e] = expf(p[e] - mx);
                sum += row[e];
            }
            for (uint32_t e = 0; e < E; e++) row[e] = (float)(row[e] / sum);
        } else {
            memcpy(row, p, E * sizeof *row);
        }

        double h = 0.0;
        for (uint32_t e = 0; e < E; e++)
            if (row[e] > 0.0f) h -= (double)row[e] * log2((double)row[e]);
        a->entropy_sum[layer] += h;

        /* min-k to reach each cumulative-mass threshold */
        qsort(row, E, sizeof *row, cmp_desc);
        double cum = 0.0;
        uint32_t k = 0;
        int ti = 0;
        for (uint32_t e = 0; e < E && ti < POE_ACCUM_NMASS; e++) {
            cum += row[e];
            k = e + 1;
            while (ti < POE_ACCUM_NMASS && cum >= poe_accum_mass_thresholds[ti]) {
                a->mass_k_sum[ti][layer] += (double)k;
                ti++;
            }
        }
        /* numerically short distributions: charge the full expert count */
        for (; ti < POE_ACCUM_NMASS; ti++)
            a->mass_k_sum[ti][layer] += (double)E;

        a->tok_probs[layer]++;
    }
    free(row);
}

void poe_accum_observe_selection(poe_accum *a, uint32_t layer, uint32_t T,
                                 const int32_t *ids, const float *weights,
                                 uint64_t *bad_ids) {
    if (layer >= a->n_layers) return;
    const uint32_t K = a->top_k, E = a->n_experts;
    uint64_t *cnt = a->sel_count + (size_t)layer * E;
    double   *gsm = a->gate_sum  + (size_t)layer * E;

    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < K; s++) {
            int32_t e = ids[s + (size_t)t * K];
            if (e < 0 || (uint32_t)e >= E) {
                if (bad_ids) (*bad_ids)++;
                continue;
            }
            cnt[e]++;
            if (weights) gsm[e] += (double)weights[s + (size_t)t * K];
        }
        a->tok_sel[layer]++;
    }
}

void poe_accum_write_json(const poe_accum *a, FILE *f, const char *indent) {
    fprintf(f, "%s[\n", indent);
    for (uint32_t l = 0; l < a->n_layers; l++) {
        const uint64_t tp = a->tok_probs[l], ts = a->tok_sel[l];
        fprintf(f, "%s  {\"layer\": %u, \"tokens\": %llu,\n",
                indent, l, (unsigned long long)(ts ? ts : tp));
        fprintf(f, "%s   \"entropy_bits_mean\": %.6f,\n",
                indent, tp ? a->entropy_sum[l] / (double)tp : 0.0);
        fprintf(f, "%s   \"mass_k_mean\": {", indent);
        for (int i = 0; i < POE_ACCUM_NMASS; i++)
            fprintf(f, "%s\"%.2f\": %.3f", i ? ", " : "",
                    poe_accum_mass_thresholds[i],
                    tp ? a->mass_k_sum[i][l] / (double)tp : 0.0);
        fprintf(f, "},\n");

        fprintf(f, "%s   \"sel_count\": [", indent);
        for (uint32_t e = 0; e < a->n_experts; e++)
            fprintf(f, "%s%llu", e ? "," : "",
                    (unsigned long long)a->sel_count[(size_t)l * a->n_experts + e]);
        fprintf(f, "],\n");

        fprintf(f, "%s   \"gate_mean\": [", indent);
        for (uint32_t e = 0; e < a->n_experts; e++) {
            uint64_t c = a->sel_count[(size_t)l * a->n_experts + e];
            fprintf(f, "%s%.6f", e ? "," : "",
                    c ? a->gate_sum[(size_t)l * a->n_experts + e] / (double)c : 0.0);
        }
        fprintf(f, "]}%s\n", l + 1 < a->n_layers ? "," : "");
    }
    fprintf(f, "%s]", indent);
}
