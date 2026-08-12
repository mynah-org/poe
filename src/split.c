/* split.c — reorder experts by need and store hot and cold at different
 * precisions, as two tensors per slab.
 * SPDX-License-Identifier: MIT */
#include "poe/split.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ingot.h"
#include "ggufw.h"

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

/* What a tensor in the output is made of. */
enum { OUT_COPY, OUT_HOT, OUT_COLD, OUT_PERM };

typedef struct {
    char     name[160];
    int      type;
    uint32_t rank;
    uint64_t ne[INGOT_MAX_RANK];
    uint64_t nbytes;
    const ingot_tensor *src;
    int      kind;
    uint32_t layer;
} out_tensor;

/* One thread's share of a slab: a contiguous range of output experts. */
typedef struct {
    const float *fbuf;          /* the whole source slab, dequantized       */
    uint8_t     *dst;
    const uint32_t *order;      /* output expert i takes source order[i]    */
    uint32_t     first, last;   /* over output experts                      */
    uint32_t     base;          /* offset into order for this half          */
    uint64_t     per_expert, out_bytes;
    int          type;
    int          rc;
} enc_job;

static void *enc_worker(void *arg) {
    enc_job *j = arg;
    for (uint32_t i = j->first; i < j->last && j->rc == 0; i++) {
        const float *src = j->fbuf + (uint64_t)j->order[j->base + i] * j->per_expert;
        if (ingot_quantize(j->type, src, (size_t)j->per_expert,
                           j->dst + (uint64_t)i * j->out_bytes) != 0)
            j->rc = -1;
    }
    return NULL;
}

static double type_bits(int type) {
    uint64_t elems = 0, bytes = 0;
    if (ingot_type_geometry(type, &elems, &bytes) != 0 || elems == 0) return 0.0;
    return 8.0 * (double)bytes / (double)elems;
}

/* Which MoE block a tensor belongs to, and what it is. */
static const poe_block *block_of(const poe_model *m, const ingot_tensor *t,
                                 uint32_t *layer) {
    unsigned bidx;
    if (sscanf(t->name, "blk.%u.", &bidx) != 1 || bidx >= m->n_blocks)
        return NULL;
    *layer = bidx;
    return m->blocks[bidx].is_moe ? &m->blocks[bidx] : NULL;
}

int poe_split(const poe_model *m, const poe_split_opts *o, const char *out_path,
              poe_split_stats *stats, char *err, size_t errsz) {
    poe_split_stats st;
    memset(&st, 0, sizeof st);
    if (stats) *stats = st;
    if (m == NULL || o == NULL || out_path == NULL)
        return fail(err, errsz, "bad arguments");

    const ingot_gguf *g = m->g;
    const uint32_t E = m->expert_count, L = m->n_blocks;

    if (ingot_gguf_shard_count(g) != 1)
        return fail(err, errsz, "split GGUF sources are not supported yet");
    if (E == 0 || m->n_moe_blocks == 0)
        return fail(err, errsz, "the model has no routed experts");
    if (!ingot_can_quantize(o->hot_type) || !ingot_can_quantize(o->cold_type))
        return fail(err, errsz, "no encoder for one of the requested types");
    if (o->profile == NULL)
        return fail(err, errsz, "splitting needs a profile: the expert order "
                                "is the whole point");
    if (o->profile->n_layers != L || o->profile->n_experts != E)
        return fail(err, errsz, "profile shape does not match the model");
    if (strcmp(o->profile->fingerprint, m->fingerprint) != 0 && !o->force)
        return fail(err, errsz, "profile is for a different checkpoint "
                                "(use --force to override)");

    const uint32_t H = (uint32_t)((double)E * o->hot_fraction + 0.5);
    if (H == 0 || H >= E)
        return fail(err, errsz, "the hot fraction must leave both a hot and a "
                                "cold set");
    st.hot_per_layer = H;
    st.cold_per_layer = E - H;
    st.hot_bits = type_bits(o->hot_type);
    st.cold_bits = type_bits(o->cold_type);
    st.mean_bits = ((double)H * st.hot_bits + (double)(E - H) * st.cold_bits) /
                   (double)E;

    uint64_t hot_elems = 0, hot_blk = 0, cold_elems = 0, cold_blk = 0;
    ingot_type_geometry(o->hot_type, &hot_elems, &hot_blk);
    ingot_type_geometry(o->cold_type, &cold_elems, &cold_blk);

    /* per-layer expert order, hottest first */
    uint32_t *order = malloc((size_t)L * E * sizeof *order);
    if (order == NULL) return fail(err, errsz, "out of memory");
    for (uint32_t l = 0; l < L; l++) {
        const int used = poe_rank_experts(o->profile, l, E, o->rank_by,
                                          order + (size_t)l * E);
        if (used < 0) {
            free(order);
            return fail(err, errsz, "the profile carries no ranking signal");
        }
        st.ranked_by_reap = used;
        if (o->invert) {                 /* the control: keep the least needed */
            uint32_t *row = order + (size_t)l * E;
            for (uint32_t i = 0; i < E / 2; i++) {
                const uint32_t tmp = row[i];
                row[i] = row[E - 1 - i];
                row[E - 1 - i] = tmp;
            }
        }
    }

    /* ── plan the output tensor table ───────────────────────────────────── */
    const size_t T = ingot_gguf_count(g);
    out_tensor *out = calloc(T + (size_t)L * 3, sizeof *out);
    if (out == NULL) { free(order); return fail(err, errsz, "out of memory"); }

    size_t n_out = 0;
    uint64_t max_nelem = 0;
    int rc = -1;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint32_t layer = 0;
        const poe_block *blk = block_of(m, t, &layer);
        const int is_slab = blk != NULL && (t == blk->gate_exps_w ||
                                            t == blk->up_exps_w ||
                                            t == blk->down_exps_w);
        const int is_perm = blk != NULL && (t == blk->router_w ||
                                            t == blk->router_b);
        const int is_ebias = blk != NULL && (t == blk->gate_exps_b ||
                                             t == blk->up_exps_b ||
                                             t == blk->down_exps_b);
        if (is_ebias) {
            fail(err, errsz, "per-expert biases are not split yet "
                             "(this architecture needs them handled)");
            goto done;
        }

        if (is_slab) {
            if (!ingot_type_can_dequant(t->type) || t->nelem % E != 0) {
                snprintf(err, errsz, "'%s' cannot be decoded and split", t->name);
                goto done;
            }
            if (t->ne[0] % hot_elems != 0 || t->ne[0] % cold_elems != 0) {
                snprintf(err, errsz, "'%s': rows are %llu elements, not a whole "
                         "number of quantization blocks", t->name,
                         (unsigned long long)t->ne[0]);
                goto done;
            }
            if (t->nelem > max_nelem) max_nelem = t->nelem;

            const uint64_t per_expert = t->nelem / E;
            for (int half = 0; half < 2; half++) {
                out_tensor *ot = &out[n_out++];
                const uint32_t count = half == 0 ? H : E - H;
                const int type = half == 0 ? o->hot_type : o->cold_type;
                if (half == 0)
                    snprintf(ot->name, sizeof ot->name, "%s", t->name);
                else {
                    /* blk.N.ffn_gate_exps.weight -> ..._exps_cold.weight */
                    const char *dot = strrchr(t->name, '.');
                    const size_t stem = dot ? (size_t)(dot - t->name)
                                            : strlen(t->name);
                    snprintf(ot->name, sizeof ot->name, "%.*s_cold%s",
                             (int)stem, t->name, dot ? dot : "");
                }
                ot->type  = type;
                ot->rank  = t->rank;
                for (uint32_t d = 0; d < t->rank; d++) ot->ne[d] = t->ne[d];
                ot->ne[t->rank - 1] = count;
                if (ingot_type_nbytes(type, per_expert * count, &ot->nbytes) != 0) {
                    snprintf(err, errsz, "cannot size '%s'", ot->name);
                    goto done;
                }
                ot->src   = t;
                ot->kind  = half == 0 ? OUT_HOT : OUT_COLD;
                ot->layer = layer;
            }
            st.slabs_split++;
            continue;
        }

        out_tensor *ot = &out[n_out++];
        snprintf(ot->name, sizeof ot->name, "%s", t->name);
        ot->type = t->type;
        ot->rank = t->rank;
        for (uint32_t d = 0; d < t->rank; d++) ot->ne[d] = t->ne[d];
        ot->nbytes = t->nbytes;
        ot->src = t;
        ot->layer = layer;
        ot->kind = OUT_COPY;
        if (is_perm) {
            if (t->ne[t->rank - 1] != E || t->nbytes % E != 0) {
                snprintf(err, errsz, "'%s' does not split evenly by expert",
                         t->name);
                goto done;
            }
            ot->kind = OUT_PERM;
            st.routers_permuted++;
        }
    }
    if (st.slabs_split == 0) {
        fail(err, errsz, "no packed routed expert tensors found");
        goto done;
    }

    /* ── metadata ───────────────────────────────────────────────────────── */
    const void *basep; size_t fsize;
    if (ingot_gguf_mapping(g, 0, &basep, &fsize) != 0) {
        fail(err, errsz, "cannot access the source mapping");
        goto done;
    }
    const uint8_t *base = basep;
    uint64_t hoff = 8, nten_src, nkv;
    if (fsize < 24 || memcmp(base, "GGUF", 4) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nten_src) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nkv) != 0 || nten_src != T) {
        fail(err, errsz, "source GGUF header is inconsistent");
        goto done;
    }
    poe_kv_span *spans = malloc((nkv ? nkv : 1) * sizeof *spans);
    poe_buf prov = { 0, 0, 0 }, infos = { 0, 0, 0 };
    uint64_t kv_end = 24, kv_out_bytes = 0, nkv_out = 0;
    if (spans == NULL) { fail(err, errsz, "out of memory"); goto done2; }
    if (poe_kv_walk(base, fsize, nkv, spans, &kv_end) != 0) {
        fail(err, errsz, "source GGUF metadata does not parse");
        goto done2;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        if (spans[i].keylen >= 4 && memcmp(spans[i].key, "poe.", 4) == 0) continue;
        kv_out_bytes += spans[i].end - spans[i].start;
        nkv_out++;
    }

    {
        char hot[16], bits[32];
        snprintf(hot, sizeof hot, "%u", H);
        snprintf(bits, sizeof bits, "%.4f", st.mean_bits);
        int oom = 0;
        oom |= poe_buf_kv_str(&prov, "poe.version", POE_VERSION);
        oom |= poe_buf_kv_str(&prov, "poe.method",
                              o->invert ? "split-inverted" : "split");
        oom |= poe_buf_kv_str(&prov, "poe.source_fingerprint", m->fingerprint);
        oom |= poe_buf_kv_str(&prov, "poe.split.hot_type",
                              ingot_type_name(o->hot_type));
        oom |= poe_buf_kv_str(&prov, "poe.split.cold_type",
                              ingot_type_name(o->cold_type));
        oom |= poe_buf_kv_str(&prov, "poe.split.mean_bits", bits);
        oom |= poe_buf_kv_str(&prov, "poe.split.hot_count_str", hot);
        /* the one a runtime reads is an integer, not a string */
        oom |= poe_buf_str(&prov, "poe.split.hot_count");
        oom |= poe_buf_u32(&prov, INGOT_KV_UINT32);
        oom |= poe_buf_u32(&prov, H);
        if (oom) { fail(err, errsz, "out of memory"); goto done2; }
        nkv_out += 8;
        kv_out_bytes += prov.n;
    }

    uint64_t align = ingot_gguf_alignment(g);
    if (align == 0) align = 32;
    uint64_t data_off = 0;
    int oom = 0;
    for (size_t i = 0; i < n_out; i++) {
        data_off = poe_align_up(data_off, align);
        oom |= poe_buf_str(&infos, out[i].name);
        oom |= poe_buf_u32(&infos, out[i].rank);
        for (uint32_t d = 0; d < out[i].rank; d++)
            oom |= poe_buf_u64(&infos, out[i].ne[d]);
        oom |= poe_buf_u32(&infos, (uint32_t)out[i].type);
        oom |= poe_buf_u64(&infos, data_off);
        data_off += out[i].nbytes;
    }
    if (oom) { fail(err, errsz, "out of memory"); goto done2; }
    const uint64_t data_base = poe_align_up(24 + kv_out_bytes + infos.n, align);

    /* ── write ──────────────────────────────────────────────────────────── */
    float   *fbuf = malloc((size_t)max_nelem * sizeof *fbuf);
    uint8_t *ebuf = NULL;
    uint64_t ebuf_cap = 0;
    (void)ingot_cpu();
    long cores = o->threads > 0 ? o->threads : sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    if (cores > 64) cores = 64;
    uint32_t n_threads = (uint32_t)cores;
    enc_job   *jobs = calloc(n_threads, sizeof *jobs);
    pthread_t *tids = calloc(n_threads, sizeof *tids);
    FILE *f = fopen(out_path, "wb");
    const ingot_tensor *cached = NULL;

    if (fbuf == NULL || jobs == NULL || tids == NULL || f == NULL) {
        fail(err, errsz, f == NULL ? "cannot create the output file"
                                   : "out of memory");
        goto done3;
    }

    {
        uint8_t head[24];
        memcpy(head, "GGUF", 4);
        poe_enc_le(head + 4,  ingot_gguf_version(g), 4);
        poe_enc_le(head + 8,  n_out, 8);
        poe_enc_le(head + 16, nkv_out, 8);
        if (poe_wput(f, head, sizeof head) != 0) goto wfail;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        const poe_kv_span *s = &spans[i];
        if (s->keylen >= 4 && memcmp(s->key, "poe.", 4) == 0) continue;
        if (poe_wput(f, base + s->start, (size_t)(s->end - s->start)) != 0)
            goto wfail;
    }
    if (poe_wput(f, prov.p, prov.n) != 0) goto wfail;
    if (poe_wput(f, infos.p, infos.n) != 0) goto wfail;
    if (poe_wpad(f, data_base - (24 + kv_out_bytes + infos.n)) != 0) goto wfail;

    uint64_t at = 0;
    for (size_t i = 0; i < n_out; i++) {
        out_tensor *ot = &out[i];
        const uint8_t *src = ingot_gguf_data(g, ot->src);
        if (src == NULL) goto wfail;
        const uint64_t aligned = poe_align_up(at, align);
        if (poe_wpad(f, aligned - at) != 0) goto wfail;
        at = aligned;

        if (ot->kind == OUT_COPY) {
            if (poe_wput(f, src, (size_t)ot->nbytes) != 0) goto wfail;
        } else if (ot->kind == OUT_PERM) {
            const uint64_t unit = ot->nbytes / E;
            const uint32_t *ord = order + (size_t)ot->layer * E;
            for (uint32_t e = 0; e < E; e++)
                if (poe_wput(f, src + (uint64_t)ord[e] * unit, (size_t)unit) != 0)
                    goto wfail;
        } else {
            const uint64_t per_expert = ot->src->nelem / E;
            const uint32_t count = (uint32_t)ot->ne[ot->rank - 1];
            const uint64_t out_bytes = ot->nbytes / count;

            if (cached != ot->src) {
                if (ingot_gguf_dequant(g, ot->src, fbuf) != 0) {
                    snprintf(err, errsz, "cannot decode '%s'", ot->src->name);
                    goto done3;
                }
                cached = ot->src;
            }
            if (ot->nbytes > ebuf_cap) {
                uint8_t *q = realloc(ebuf, (size_t)ot->nbytes);
                if (q == NULL) { fail(err, errsz, "out of memory"); goto done3; }
                ebuf = q;
                ebuf_cap = ot->nbytes;
            }

            const uint32_t nt = n_threads < count ? n_threads : count;
            for (uint32_t w = 0; w < nt; w++) {
                jobs[w] = (enc_job){ fbuf, ebuf, order + (size_t)ot->layer * E,
                                     (uint32_t)((uint64_t)count * w / nt),
                                     (uint32_t)((uint64_t)count * (w + 1) / nt),
                                     ot->kind == OUT_HOT ? 0 : H,
                                     per_expert, out_bytes, ot->type, 0 };
            }
            for (uint32_t w = 1; w < nt; w++)
                if (pthread_create(&tids[w], NULL, enc_worker, &jobs[w]) != 0) {
                    enc_worker(&jobs[w]);
                    tids[w] = 0;
                }
            enc_worker(&jobs[0]);
            for (uint32_t w = 1; w < nt; w++)
                if (tids[w] != 0) pthread_join(tids[w], NULL);
            int worker_rc = 0;
            for (uint32_t w = 0; w < nt; w++) worker_rc |= jobs[w].rc;
            if (worker_rc != 0) {
                snprintf(err, errsz, "cannot encode '%s' as %s", ot->name,
                         ingot_type_name(ot->type));
                goto done3;
            }
            if (poe_wput(f, ebuf, (size_t)ot->nbytes) != 0) goto wfail;
        }
        at += ot->nbytes;
    }
    if (fflush(f) != 0 || ferror(f)) goto wfail;

    st.bytes_before  = m->total_bytes;
    st.bytes_written = data_base + data_off;
    rc = 0;
    goto done3;

wfail:
    fail(err, errsz, "writing the output file failed");

done3:
    if (f != NULL) {
        if (fclose(f) != 0 && rc == 0)
            rc = fail(err, errsz, "writing the output file failed");
        if (rc != 0) remove(out_path);
    }
    free(fbuf); free(ebuf); free(jobs); free(tids);
done2:
    free(spans); free(prov.p); free(infos.p);
done:
    free(out); free(order);
    if (rc == 0 && stats) *stats = st;
    return rc;
}
