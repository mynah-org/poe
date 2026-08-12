/* requant.c — emulate per-expert precision inside one tensor type.
 * SPDX-License-Identifier: MIT */
#include "poe/requant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot.h"
#include "ggufw.h"

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

double poe_type_bits(int type) {
    uint64_t elems = 0, bytes = 0;
    if (ingot_type_geometry(type, &elems, &bytes) != 0 || elems == 0) return 0.0;
    return 8.0 * (double)bytes / (double)elems;
}

/* The MoE block whose packed expert weights this tensor is, or NULL. Biases
 * and routers are left alone: the experiment is about expert weights, and
 * touching anything else would blur what a number means. */
static const poe_block *slab_owner(const poe_model *m, const ingot_tensor *t) {
    unsigned bidx;
    if (sscanf(t->name, "blk.%u.", &bidx) != 1 || bidx >= m->n_blocks)
        return NULL;
    const poe_block *blk = &m->blocks[bidx];
    if (!blk->is_moe) return NULL;
    if (t == blk->gate_exps_w || t == blk->up_exps_w || t == blk->down_exps_w)
        return blk;
    return NULL;
}

typedef struct { double score; uint32_t idx; } ranked;

static int cmp_score_asc(const void *a, const void *b) {
    const double x = ((const ranked *)a)->score, y = ((const ranked *)b)->score;
    if (x < y) return -1;
    if (x > y) return 1;
    /* ties resolve by index, so the same profile always yields the same set */
    return (int)((const ranked *)a)->idx - (int)((const ranked *)b)->idx;
}

/* Mark the experts to degrade, per layer. Within a layer, selection counts
 * are a usable ranking — unlike summed over a layer, where they are
 * tokens x top_k by construction and rank nothing. REAP saliency is
 * preferred when the profile carries it. */
static int mark_cold(uint8_t *cold, const poe_profile *pr, uint32_t L,
                     uint32_t E, uint32_t n_cold, int invert, int *by_reap) {
    ranked *r = malloc(E * sizeof *r);
    if (r == NULL) return -1;
    *by_reap = pr != NULL && pr->reap_mean != NULL;
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t e = 0; e < E; e++) {
            const size_t k = (size_t)l * E + e;
            r[e].idx = e;
            r[e].score = *by_reap ? pr->reap_mean[k]
                                  : (double)pr->sel_count[k];
        }
        qsort(r, E, sizeof *r, cmp_score_asc);
        /* ascending: the coldest are first, unless the control flips it */
        for (uint32_t i = 0; i < n_cold; i++) {
            const uint32_t e = invert ? r[E - 1 - i].idx : r[i].idx;
            cold[(size_t)l * E + e] = 1;
        }
    }
    free(r);
    return 0;
}

int poe_requant(const poe_model *m, const poe_requant_opts *o,
                const char *out_path, poe_requant_stats *stats,
                char *err, size_t errsz) {
    poe_requant_stats st = { 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0 };
    if (stats) *stats = st;
    if (m == NULL || o == NULL || out_path == NULL)
        return fail(err, errsz, "bad arguments");

    const ingot_gguf *g = m->g;
    const uint32_t E = m->expert_count, L = m->n_blocks;

    if (ingot_gguf_shard_count(g) != 1)
        return fail(err, errsz, "split GGUF sources are not supported yet");
    if (E == 0 || m->n_moe_blocks == 0)
        return fail(err, errsz, "the model has no routed experts");
    if (!ingot_can_quantize(o->carrier_type))
        return fail(err, errsz, "no encoder for the carrier type");
    if (!ingot_can_quantize(o->degrade_type))
        return fail(err, errsz, "no encoder for the degrade type");
    if (!(o->degrade_fraction > 0.0 && o->degrade_fraction <= 1.0))
        return fail(err, errsz, "degrade fraction must be in (0,1]");

    const uint32_t n_cold = (uint32_t)((double)E * o->degrade_fraction + 0.5);
    if (n_cold == 0)
        return fail(err, errsz, "the degrade fraction rounds to zero experts");
    if (n_cold < E && o->profile == NULL)
        return fail(err, errsz, "ranking experts needs a profile "
                                "(or degrade every expert with a fraction of 1)");
    if (o->profile != NULL) {
        if (o->profile->n_layers != L || o->profile->n_experts != E)
            return fail(err, errsz, "profile shape does not match the model");
        if (strcmp(o->profile->fingerprint, m->fingerprint) != 0 && !o->force)
            return fail(err, errsz, "profile is for a different checkpoint "
                                    "(use --force to override)");
    }

    st.carrier_bits = poe_type_bits(o->carrier_type);
    st.degrade_bits = poe_type_bits(o->degrade_type);
    if (st.carrier_bits <= 0.0 || st.degrade_bits <= 0.0)
        return fail(err, errsz, "a type has no known block geometry");
    if (st.degrade_bits > st.carrier_bits)
        return fail(err, errsz, "the degrade type is wider than the carrier: "
                                "that emulates nothing");
    const double f = (double)n_cold / (double)E;
    st.emulated_bits = f * st.degrade_bits + (1.0 - f) * st.carrier_bits;
    st.degraded_per_layer = n_cold;

    uint64_t deg_elems = 0, deg_bytes = 0, car_elems = 0, car_bytes = 0;
    ingot_type_geometry(o->degrade_type, &deg_elems, &deg_bytes);
    ingot_type_geometry(o->carrier_type, &car_elems, &car_bytes);

    /* which tensors are rewritten, and can every one of them be? */
    const size_t T = ingot_gguf_count(g);
    const poe_block **owner = calloc(T, sizeof *owner);
    if (owner == NULL) return fail(err, errsz, "out of memory");

    uint64_t max_nelem = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        owner[i] = slab_owner(m, t);
        if (owner[i] == NULL) continue;
        if (!ingot_type_can_dequant(t->type)) {
            free(owner);
            snprintf(err, errsz, "cannot decode '%s' (%s)", t->name,
                     ingot_type_name(t->type));
            return -1;
        }
        if (t->nelem % E != 0) {
            free(owner);
            snprintf(err, errsz, "'%s' does not divide by the expert count",
                     t->name);
            return -1;
        }
        /* Quantization blocks are laid out along the row, so it is ne[0]
         * that has to divide — a slab whose total element count divides but
         * whose rows do not would encode into a tensor no runtime can read. */
        if (t->ne[0] % deg_elems != 0 || t->ne[0] % car_elems != 0) {
            free(owner);
            snprintf(err, errsz, "'%s': rows are %llu elements, not a whole "
                     "number of %s blocks", t->name,
                     (unsigned long long)t->ne[0],
                     t->ne[0] % car_elems != 0 ? ingot_type_name(o->carrier_type)
                                               : ingot_type_name(o->degrade_type));
            return -1;
        }
        const uint64_t per_expert = t->nelem / E;
        if (per_expert % deg_elems != 0 || per_expert % car_elems != 0) {
            free(owner);
            snprintf(err, errsz, "'%s': one expert is %llu elements, which is "
                     "not a whole number of quantization blocks", t->name,
                     (unsigned long long)per_expert);
            return -1;
        }
        if (t->nelem > max_nelem) max_nelem = t->nelem;
    }
    if (max_nelem == 0) {
        free(owner);
        return fail(err, errsz, "no packed routed expert tensors found");
    }

    uint8_t *cold = calloc((size_t)L * E, 1);
    if (cold == NULL) { free(owner); return fail(err, errsz, "out of memory"); }
    if (n_cold >= E) {
        memset(cold, 1, (size_t)L * E);
        st.ranked_by_reap = 0;
    } else if (mark_cold(cold, o->profile, L, E, n_cold, o->invert,
                         &st.ranked_by_reap) != 0) {
        free(owner); free(cold);
        return fail(err, errsz, "out of memory");
    }

    /* metadata: copy every KV except stale provenance */
    const void *basep; size_t fsize;
    if (ingot_gguf_mapping(g, 0, &basep, &fsize) != 0) {
        free(owner); free(cold);
        return fail(err, errsz, "cannot access the source mapping");
    }
    const uint8_t *base = basep;
    uint64_t hoff = 8, nten_src, nkv;
    if (fsize < 24 || memcmp(base, "GGUF", 4) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nten_src) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nkv) != 0 || nten_src != T) {
        free(owner); free(cold);
        return fail(err, errsz, "source GGUF header is inconsistent");
    }
    poe_kv_span *spans = malloc((nkv ? nkv : 1) * sizeof *spans);
    if (spans == NULL) {
        free(owner); free(cold);
        return fail(err, errsz, "out of memory");
    }
    uint64_t kv_end = 24;
    if (poe_kv_walk(base, fsize, nkv, spans, &kv_end) != 0) {
        free(owner); free(cold); free(spans);
        return fail(err, errsz, "source GGUF metadata does not parse");
    }
    uint64_t kv_out_bytes = 0, nkv_out = 0;
    for (uint64_t i = 0; i < nkv; i++) {
        if (spans[i].keylen >= 4 && memcmp(spans[i].key, "poe.", 4) == 0) continue;
        kv_out_bytes += spans[i].end - spans[i].start;
        nkv_out++;
    }

    /* provenance: enough to reconstruct the experiment from the file alone */
    char frac[32], bits[32];
    snprintf(frac, sizeof frac, "%.6f", f);
    snprintf(bits, sizeof bits, "%.4f", st.emulated_bits);
    poe_buf prov = { 0, 0, 0 };
    int oom = 0;
    oom |= poe_buf_kv_str(&prov, "poe.version", POE_VERSION);
    oom |= poe_buf_kv_str(&prov, "poe.method",
                          n_cold >= E ? "requant-uniform"
                                      : o->invert ? "requant-per-expert-inverted"
                                                  : "requant-per-expert");
    oom |= poe_buf_kv_str(&prov, "poe.source_fingerprint", m->fingerprint);
    oom |= poe_buf_kv_str(&prov, "poe.requant.carrier",
                          ingot_type_name(o->carrier_type));
    oom |= poe_buf_kv_str(&prov, "poe.requant.degrade_type",
                          ingot_type_name(o->degrade_type));
    oom |= poe_buf_kv_str(&prov, "poe.requant.degrade_fraction", frac);
    oom |= poe_buf_kv_str(&prov, "poe.requant.emulated_bits", bits);
    if (oom) {
        free(owner); free(cold); free(spans); free(prov.p);
        return fail(err, errsz, "out of memory");
    }
    nkv_out += 7;
    kv_out_bytes += prov.n;

    /* tensor table: same shapes, carrier type on the routed slabs */
    uint64_t align = ingot_gguf_alignment(g);
    if (align == 0) align = 32;
    poe_buf infos = { 0, 0, 0 };
    uint64_t data_off = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint64_t nb = t->nbytes;
        if (owner[i] != NULL && ingot_type_nbytes(o->carrier_type, t->nelem, &nb) != 0)
            oom = 1;
        data_off = poe_align_up(data_off, align);
        oom |= poe_buf_str(&infos, t->name);
        oom |= poe_buf_u32(&infos, t->rank);
        for (uint32_t d = 0; d < t->rank; d++)
            oom |= poe_buf_u64(&infos, t->ne[d]);
        oom |= poe_buf_u32(&infos, (uint32_t)(owner[i] ? o->carrier_type : t->type));
        oom |= poe_buf_u64(&infos, data_off);
        data_off += nb;
    }
    if (oom) {
        free(owner); free(cold); free(spans); free(prov.p); free(infos.p);
        return fail(err, errsz, "out of memory");
    }
    const uint64_t data_base = poe_align_up(24 + kv_out_bytes + infos.n, align);

    /* one slab-sized set of buffers, reused for every tensor */
    uint64_t car_max = 0, deg_max = 0;
    ingot_type_nbytes(o->carrier_type, max_nelem, &car_max);
    ingot_type_nbytes(o->degrade_type, max_nelem / E, &deg_max);
    float   *fbuf = malloc((size_t)max_nelem * sizeof *fbuf);
    uint8_t *cbuf = malloc((size_t)car_max);
    uint8_t *dbuf = malloc((size_t)deg_max);
    FILE    *fout = fopen(out_path, "wb");
    int rc = -1;
    if (fbuf == NULL || cbuf == NULL || dbuf == NULL || fout == NULL) {
        fail(err, errsz, fout == NULL ? "cannot create the output file"
                                      : "out of memory");
        goto done;
    }

    {
        uint8_t head[24];
        memcpy(head, "GGUF", 4);
        poe_enc_le(head + 4,  ingot_gguf_version(g), 4);
        poe_enc_le(head + 8,  T, 8);
        poe_enc_le(head + 16, nkv_out, 8);
        if (poe_wput(fout, head, sizeof head) != 0) goto wfail;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        const poe_kv_span *s = &spans[i];
        if (s->keylen >= 4 && memcmp(s->key, "poe.", 4) == 0) continue;
        if (poe_wput(fout, base + s->start, (size_t)(s->end - s->start)) != 0)
            goto wfail;
    }
    if (poe_wput(fout, prov.p, prov.n) != 0) goto wfail;
    if (poe_wput(fout, infos.p, infos.n) != 0) goto wfail;
    if (poe_wpad(fout, data_base - (24 + kv_out_bytes + infos.n)) != 0) goto wfail;

    uint64_t at = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        const uint8_t *src = ingot_gguf_data(g, t);
        if (src == NULL) goto wfail;
        const uint64_t aligned = poe_align_up(at, align);
        if (poe_wpad(fout, aligned - at) != 0) goto wfail;
        at = aligned;

        if (owner[i] == NULL) {
            if (poe_wput(fout, src, (size_t)t->nbytes) != 0) goto wfail;
            at += t->nbytes;
            continue;
        }

        if (ingot_gguf_dequant(g, t, fbuf) != 0) {
            snprintf(err, errsz, "cannot decode '%s'", t->name);
            goto done;
        }
        const uint64_t per_expert = t->nelem / E;
        const uint8_t *layer_cold = cold + (size_t)owner[i]->block * E;
        for (uint32_t e = 0; e < E; e++) {
            if (!layer_cold[e]) continue;
            float *slice = fbuf + (uint64_t)e * per_expert;
            if (ingot_quantize(o->degrade_type, slice, (size_t)per_expert,
                               dbuf) != 0 ||
                ingot_dequant(o->degrade_type, dbuf, (size_t)per_expert,
                              slice) != 0) {
                snprintf(err, errsz, "cannot re-encode an expert of '%s' as %s",
                         t->name, ingot_type_name(o->degrade_type));
                goto done;
            }
            st.experts_degraded++;
        }
        uint64_t nb = 0;
        if (ingot_quantize(o->carrier_type, fbuf, (size_t)t->nelem, cbuf) != 0 ||
            ingot_type_nbytes(o->carrier_type, t->nelem, &nb) != 0) {
            snprintf(err, errsz, "cannot encode '%s' as %s", t->name,
                     ingot_type_name(o->carrier_type));
            goto done;
        }
        if (poe_wput(fout, cbuf, (size_t)nb) != 0) goto wfail;
        at += nb;
        st.slabs_rewritten++;
    }
    if (fflush(fout) != 0 || ferror(fout)) goto wfail;

    st.tensors_total = (uint32_t)T;
    st.bytes_written = data_base + data_off;
    rc = 0;
    goto done;

wfail:
    fail(err, errsz, "writing the output file failed");

done:
    if (fout != NULL) {
        if (fclose(fout) != 0 && rc == 0)
            rc = fail(err, errsz, "writing the output file failed");
        if (rc != 0) remove(out_path);
    }
    free(fbuf); free(cbuf); free(dbuf);
    free(owner); free(cold); free(spans); free(prov.p); free(infos.p);
    if (rc == 0 && stats) *stats = st;
    return rc;
}
