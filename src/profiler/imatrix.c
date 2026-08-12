/* profiler/imatrix.c — per-expert activation statistics in llama.cpp's
 * GGUF imatrix format.
 * SPDX-License-Identifier: MIT */
#include "imatrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot.h"

/* The weight tensor each projection's statistic belongs to. llama-quantize
 * looks entries up by the model's own tensor name, so these must match what
 * the GGUF carries — the same names src/model.c discovers. */
static const char *proj_tensor[POE_IMAT_NPROJ] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

int poe_imatrix_init(poe_imatrix *m, uint32_t n_layers, uint32_t n_experts) {
    memset(m, 0, sizeof *m);
    if (n_layers == 0 || n_experts == 0) return -1;
    m->n_layers = n_layers;
    m->n_experts = n_experts;
    return 0;   /* the per-projection arrays wait for a column count */
}

void poe_imatrix_free(poe_imatrix *m) {
    for (int p = 0; p < POE_IMAT_NPROJ; p++) {
        free(m->sum2[p]);
        free(m->counts[p]);
    }
    for (size_t i = 0; i < m->n_plain; i++) free(m->plain[i].sum2);
    free(m->plain);
    memset(m, 0, sizeof *m);
}

int poe_imatrix_has_data(const poe_imatrix *m) {
    for (int p = 0; p < POE_IMAT_NPROJ; p++)
        if (m->sum2[p] != NULL) return 1;
    return m->n_plain > 0;
}

int poe_imatrix_observe_plain(poe_imatrix *m, const char *wname,
                              uint32_t cols, uint32_t rows, const float *act) {
    if (m == NULL || wname == NULL || act == NULL) return -1;
    if (cols == 0 || rows == 0) return -1;
    if (strlen(wname) + 1 > sizeof m->plain[0].name) return -1;

    poe_imat_plain *e = NULL;
    for (size_t i = 0; i < m->n_plain; i++)
        if (strcmp(m->plain[i].name, wname) == 0) { e = &m->plain[i]; break; }

    if (e == NULL) {
        if (m->n_plain == m->cap_plain) {
            const size_t cap = m->cap_plain ? m->cap_plain * 2 : 64;
            poe_imat_plain *q = realloc(m->plain, cap * sizeof *q);
            if (q == NULL) return -1;
            m->plain = q;
            m->cap_plain = cap;
        }
        e = &m->plain[m->n_plain];
        memset(e, 0, sizeof *e);
        snprintf(e->name, sizeof e->name, "%s", wname);
        e->cols = cols;
        e->sum2 = calloc(cols, sizeof *e->sum2);
        if (e->sum2 == NULL) return -1;
        m->n_plain++;
    } else if (e->cols != cols) {
        return -1;    /* two widths under one name would be silent garbage */
    }

    for (uint32_t r = 0; r < rows; r++) {
        const float *x = act + (size_t)r * cols;
        for (uint32_t j = 0; j < cols; j++) {
            const double v = (double)x[j] * (double)x[j];
            if (!isfinite(v)) { m->nonfinite++; continue; }
            e->sum2[j] += v;
        }
        e->count++;
    }
    m->observations++;
    return 0;
}

int poe_imatrix_observe(poe_imatrix *m, uint32_t layer, poe_imat_proj proj,
                        uint32_t cols, uint32_t act_rows, uint32_t n_used,
                        uint32_t T, const float *act, const int32_t *ids,
                        uint64_t *bad_ids) {
    if (m == NULL || act == NULL || ids == NULL) return -1;
    if (proj >= POE_IMAT_NPROJ || layer >= m->n_layers) return -1;
    if (cols == 0 || act_rows == 0 || n_used == 0 || T == 0) return -1;

    if (m->sum2[proj] == NULL) {
        const size_t n = (size_t)m->n_layers * m->n_experts * cols;
        m->sum2[proj] = calloc(n, sizeof *m->sum2[proj]);
        m->counts[proj] = calloc((size_t)m->n_layers * m->n_experts,
                                 sizeof *m->counts[proj]);
        if (m->sum2[proj] == NULL || m->counts[proj] == NULL) {
            free(m->sum2[proj]);   m->sum2[proj] = NULL;
            free(m->counts[proj]); m->counts[proj] = NULL;
            return -1;
        }
        m->cols[proj] = cols;
    } else if (m->cols[proj] != cols) {
        return -1;    /* averaging two widths together would be silent garbage */
    }

    const size_t layer_base = (size_t)layer * m->n_experts;
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t s = 0; s < n_used; s++) {
            const int32_t e = ids[(size_t)t * n_used + s];
            if (e < 0 || (uint32_t)e >= m->n_experts) {
                if (bad_ids) (*bad_ids)++;
                continue;
            }
            /* One activation row per selected expert, or a single row
             * broadcast to all of them — s % act_rows covers both. */
            const float *x = act + ((size_t)t * act_rows + (s % act_rows)) * cols;
            double *dst = m->sum2[proj] + (layer_base + (uint32_t)e) * cols;
            m->counts[proj][layer_base + (uint32_t)e]++;
            for (uint32_t j = 0; j < cols; j++) {
                const double v = (double)x[j] * (double)x[j];
                if (!isfinite(v)) { m->nonfinite++; continue; }
                dst[j] += v;
            }
        }
    }
    m->observations++;
    return 0;
}

/* ── writing ────────────────────────────────────────────────────────────── */

/* One GGUF tensor to emit: the name, and the f32 payload the writer borrows
 * until save() (ingot does not copy tensor data). */
typedef struct {
    char   name[160];      /* a weight name plus ".in_sum2", never truncated */
    float *data;
    uint64_t ne[2];
} entry;

static int entry_cmp(const void *a, const void *b) {
    return strcmp(((const entry *)a)->name, ((const entry *)b)->name);
}

static void fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
}

int poe_imatrix_write_gguf(const poe_imatrix *m, const char *path,
                           const char *dataset, uint32_t chunk_count,
                           uint32_t chunk_size, char *err, size_t errsz) {
    if (m == NULL || path == NULL) { fail(err, errsz, "null argument"); return -1; }
    if (!poe_imatrix_has_data(m)) {
        fail(err, errsz, "no activations were observed");
        return -1;
    }

    /* Two tensors per (layer, projection) that has data, plus two per
     * non-MoE weight matrix. */
    size_t cap = m->n_plain * 2;
    for (int p = 0; p < POE_IMAT_NPROJ; p++)
        if (m->sum2[p]) cap += (size_t)m->n_layers * 2;

    entry *entries = calloc(cap, sizeof *entries);
    if (entries == NULL) { fail(err, errsz, "out of memory"); return -1; }

    int rc = -1;
    size_t n = 0;
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (w == NULL) { fail(err, errsz, "out of memory"); goto done; }

    for (int p = 0; p < POE_IMAT_NPROJ; p++) {
        if (m->sum2[p] == NULL) continue;
        const uint32_t cols = m->cols[p];
        for (uint32_t l = 0; l < m->n_layers; l++) {
            const size_t base = (size_t)l * m->n_experts;

            entry *sums = &entries[n++];
            snprintf(sums->name, sizeof sums->name, "blk.%u.%s.in_sum2",
                     l, proj_tensor[p]);
            sums->ne[0] = cols;
            sums->ne[1] = m->n_experts;
            sums->data = malloc((size_t)cols * m->n_experts * sizeof *sums->data);

            entry *cnt = &entries[n++];
            snprintf(cnt->name, sizeof cnt->name, "blk.%u.%s.counts",
                     l, proj_tensor[p]);
            cnt->ne[0] = 1;
            cnt->ne[1] = m->n_experts;
            cnt->data = malloc((size_t)m->n_experts * sizeof *cnt->data);

            if (sums->data == NULL || cnt->data == NULL) {
                fail(err, errsz, "out of memory");
                goto done;
            }
            /* The loader divides sums by counts, so the raw sum of squares
             * is what goes on disk — never a mean. */
            for (uint32_t e = 0; e < m->n_experts; e++) {
                const double *src = m->sum2[p] + (base + e) * cols;
                float *dst = sums->data + (size_t)e * cols;
                for (uint32_t j = 0; j < cols; j++) dst[j] = (float)src[j];
                cnt->data[e] = (float)m->counts[p][base + e];
            }
        }
    }

    /* Non-MoE matrices: one row, and counts[0] is the token count, which is
     * exactly the shape llama.cpp writes for a plain matmul. */
    for (size_t i = 0; i < m->n_plain; i++) {
        const poe_imat_plain *src = &m->plain[i];

        entry *sums = &entries[n++];
        snprintf(sums->name, sizeof sums->name, "%s.in_sum2", src->name);
        sums->ne[0] = src->cols;
        sums->ne[1] = 1;
        sums->data = malloc((size_t)src->cols * sizeof *sums->data);

        entry *cnt = &entries[n++];
        snprintf(cnt->name, sizeof cnt->name, "%s.counts", src->name);
        cnt->ne[0] = 1;
        cnt->ne[1] = 1;
        cnt->data = malloc(sizeof *cnt->data);

        if (sums->data == NULL || cnt->data == NULL) {
            fail(err, errsz, "out of memory");
            goto done;
        }
        for (uint32_t j = 0; j < src->cols; j++)
            sums->data[j] = (float)src->sum2[j];
        cnt->data[0] = (float)src->count;
    }

    qsort(entries, n, sizeof *entries, entry_cmp);

    if (ingot_gguf_kv_string(w, "general.type", "imatrix") != 0 ||
        ingot_gguf_kv_u32(w, "imatrix.chunk_count", chunk_count) != 0 ||
        ingot_gguf_kv_u32(w, "imatrix.chunk_size", chunk_size) != 0) {
        fail(err, errsz, "cannot write imatrix metadata");
        goto done;
    }
    if (dataset != NULL) {
        const char *one[1] = { dataset };
        if (ingot_gguf_kv_array_string(w, "imatrix.datasets", one, 1) != 0) {
            fail(err, errsz, "cannot write imatrix.datasets");
            goto done;
        }
    }

    for (size_t i = 0; i < n; i++) {
        if (ingot_gguf_add_tensor(w, entries[i].name, INGOT_TYPE_F32, 2,
                                  entries[i].ne, entries[i].data) != 0) {
            fail(err, errsz, "cannot add an imatrix tensor");
            goto done;
        }
    }

    rc = ingot_gguf_writer_save(w, path, err, errsz);

done:
    if (w) ingot_gguf_writer_free(w);
    for (size_t i = 0; i < n; i++) free(entries[i].data);
    free(entries);
    return rc;
}
