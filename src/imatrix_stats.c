/* imatrix_stats.c — read a GGUF imatrix and reduce it to per-layer scores.
 * SPDX-License-Identifier: MIT */
#include "poe/imatrix_stats.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The three routed projections, in the order poe_imat_proj names them. The
 * strings are the model's own tensor names, which is what the imatrix keys
 * its entries by. */
static const char *proj_name[POE_IMSTAT_NPROJ] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

static int fail(char *err, size_t errsz, const char *fmt, ...) {
    if (err && errsz) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* "blk.<L>.<proj>.in_sum2" / ".counts" — anything else is not ours. Returns
 * 1 on a match, with *layer, *proj and *is_counts filled. */
static int parse_entry(const char *name, uint32_t *layer, int *proj,
                       int *is_counts) {
    if (strncmp(name, "blk.", 4) != 0) return 0;
    const char *p = name + 4;
    if (*p < '0' || *p > '9') return 0;
    unsigned long l = 0;
    while (*p >= '0' && *p <= '9') l = l * 10 + (unsigned long)(*p++ - '0');
    if (*p++ != '.') return 0;

    for (int i = 0; i < POE_IMSTAT_NPROJ; i++) {
        const size_t n = strlen(proj_name[i]);
        if (strncmp(p, proj_name[i], n) != 0) continue;
        const char *suf = p + n;
        if      (strcmp(suf, ".in_sum2") == 0) *is_counts = 0;
        else if (strcmp(suf, ".counts")  == 0) *is_counts = 1;
        else continue;
        *layer = (uint32_t)l;
        *proj  = i;
        return 1;
    }
    return 0;
}

void poe_imatrix_stats_free(poe_imatrix_stats *s) {
    if (s == NULL) return;
    free(s->energy);
    free(s->concentration);
    free(s->counts);
    free(s);
}

int poe_imatrix_stats_load(poe_imatrix_stats **out, const char *path,
                           char *err, size_t errsz) {
    if (out == NULL || path == NULL) return fail(err, errsz, "null argument");
    *out = NULL;

    ingot_gguf *g = NULL;
    if (ingot_gguf_open(&g, path, err, errsz) != 0) return -1;

    int rc = -1;
    poe_imatrix_stats *s = NULL;
    const float **sum2 = NULL, **cnts = NULL;

    const ingot_kv *kv = ingot_gguf_kv_find(g, "general.type");
    const char *type = NULL;
    if (kv == NULL || ingot_kv_str(kv, &type) != 0 ||
        strcmp(type, "imatrix") != 0) {
        fail(err, errsz, "'%s' is not an imatrix (general.type is not \"imatrix\")",
             path);
        goto done;
    }

    /* Shape discovery: the layer count is the largest index present, the
     * expert count the second dimension of any in_sum2. Column counts are
     * per projection and must agree across layers — a silent disagreement
     * would average two different widths together. */
    uint32_t n_layers = 0, n_experts = 0;
    uint32_t cols[POE_IMSTAT_NPROJ] = {0, 0, 0};
    uint32_t n_entries = 0;
    for (size_t i = 0; i < ingot_gguf_count(g); i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint32_t l; int p, is_counts;
        if (!parse_entry(t->name, &l, &p, &is_counts)) continue;
        if (l + 1 > n_layers) n_layers = l + 1;
        if (is_counts) continue;
        n_entries++;
        if (t->type != INGOT_TYPE_F32 || t->rank != 2) {
            fail(err, errsz, "%s: expected a 2-D F32 tensor", t->name);
            goto done;
        }
        if (cols[p] == 0) cols[p] = (uint32_t)t->ne[0];
        else if (cols[p] != (uint32_t)t->ne[0]) {
            fail(err, errsz, "%s: column count changes between layers "
                             "(%u vs %llu)", t->name, cols[p],
                 (unsigned long long)t->ne[0]);
            goto done;
        }
        if (n_experts == 0) n_experts = (uint32_t)t->ne[1];
        else if (n_experts != (uint32_t)t->ne[1]) {
            fail(err, errsz, "%s: expert count changes between entries",
                 t->name);
            goto done;
        }
    }
    if (n_entries == 0 || n_layers == 0 || n_experts == 0) {
        fail(err, errsz, "'%s' has no routed expert entries "
                         "(blk.N.ffn_*_exps.weight.in_sum2)", path);
        goto done;
    }

    s = calloc(1, sizeof *s);
    sum2 = calloc((size_t)n_layers * POE_IMSTAT_NPROJ, sizeof *sum2);
    cnts = calloc((size_t)n_layers * POE_IMSTAT_NPROJ, sizeof *cnts);
    if (s == NULL || sum2 == NULL || cnts == NULL) {
        fail(err, errsz, "out of memory");
        goto done;
    }
    s->n_layers  = n_layers;
    s->n_experts = n_experts;
    s->n_entries = n_entries;
    for (int p = 0; p < POE_IMSTAT_NPROJ; p++) s->cols[p] = cols[p];
    s->energy        = calloc(n_layers, sizeof *s->energy);
    s->concentration = calloc(n_layers, sizeof *s->concentration);
    s->counts        = calloc((size_t)n_layers * n_experts, sizeof *s->counts);
    if (!s->energy || !s->concentration || !s->counts) {
        fail(err, errsz, "out of memory");
        goto done;
    }

    /* Provenance, for the report. Absent keys are legal. */
    if ((kv = ingot_gguf_kv_find(g, "imatrix.chunk_count")) != NULL) {
        uint64_t v = 0;
        if (ingot_kv_u64(kv, &v) == 0) s->chunk_count = (uint32_t)v;
    }
    if ((kv = ingot_gguf_kv_find(g, "imatrix.chunk_size")) != NULL) {
        uint64_t v = 0;
        if (ingot_kv_u64(kv, &v) == 0) s->chunk_size = (uint32_t)v;
    }
    if ((kv = ingot_gguf_kv_find(g, "imatrix.datasets")) != NULL) {
        uint64_t n = 0;
        const char *d = NULL; size_t dl = 0;
        if (ingot_kv_arr_len(kv, &n) == 0 && n > 0 &&
            ingot_kv_arr_str(kv, 0, &d, &dl) == 0) {
            if (dl >= sizeof s->dataset) dl = sizeof s->dataset - 1;
            memcpy(s->dataset, d, dl);
            s->dataset[dl] = '\0';
        }
    }

    for (size_t i = 0; i < ingot_gguf_count(g); i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint32_t l; int p, is_counts;
        if (!parse_entry(t->name, &l, &p, &is_counts)) continue;
        const float *d = ingot_gguf_data(g, t);
        if (d == NULL) {
            fail(err, errsz, "%s: tensor data is not readable", t->name);
            goto done;
        }
        if (is_counts) cnts[(size_t)l * POE_IMSTAT_NPROJ + p] = d;
        else           sum2[(size_t)l * POE_IMSTAT_NPROJ + p] = d;
    }

    for (uint32_t l = 0; l < n_layers; l++) {
        double energy_sum = 0.0, conc_sum = 0.0;
        int n_present = 0;
        for (int p = 0; p < POE_IMSTAT_NPROJ; p++) {
            const float *sm = sum2[(size_t)l * POE_IMSTAT_NPROJ + p];
            const float *ct = cnts[(size_t)l * POE_IMSTAT_NPROJ + p];
            if (sm == NULL || ct == NULL || cols[p] == 0) continue;

            /* Column energy summed over experts, and the routed-slot total.
             * The file stores raw sums of squares, so dividing by slots and
             * by width is what turns them into a mean per element. */
            double *col = calloc(cols[p], sizeof *col);
            if (col == NULL) { fail(err, errsz, "out of memory"); goto done; }
            double slots = 0.0;
            for (uint32_t e = 0; e < n_experts; e++) {
                const double c = (double)ct[e];
                slots += c;
                const float *row = sm + (size_t)e * cols[p];
                for (uint32_t j = 0; j < cols[p]; j++) col[j] += (double)row[j];
                /* counts agree across projections by construction (one
                 * routing decision feeds all three); keep the largest so a
                 * partially written file still reports something honest. */
                const uint64_t cu = (uint64_t)c;
                if (cu > s->counts[(size_t)l * n_experts + e])
                    s->counts[(size_t)l * n_experts + e] = cu;
            }
            if (slots <= 0.0) { free(col); continue; }

            double sum = 0.0, sumsq = 0.0;
            for (uint32_t j = 0; j < cols[p]; j++) {
                sum   += col[j];
                sumsq += col[j] * col[j];
            }
            free(col);
            energy_sum += sum / (slots * (double)cols[p]);
            /* Participation ratio in (0,1]: 1 when every column carries the
             * same energy, 1/cols when one column carries all of it. */
            if (sumsq > 0.0) {
                const double pr = (sum * sum) / ((double)cols[p] * sumsq);
                conc_sum += 1.0 - pr;
            }
            n_present++;
        }
        if (n_present > 0) {
            s->energy[l]        = energy_sum / n_present;
            s->concentration[l] = conc_sum   / n_present;
        }
    }

    for (uint32_t l = 0; l < n_layers; l++)
        for (uint32_t e = 0; e < n_experts; e++) {
            const uint64_t c = s->counts[(size_t)l * n_experts + e];
            s->slots_total += c;
            if (c == 0) s->experts_unseen++;
        }

    *out = s;
    s = NULL;
    rc = 0;

done:
    free(sum2);
    free(cnts);
    poe_imatrix_stats_free(s);
    ingot_gguf_close(g);
    return rc;
}
