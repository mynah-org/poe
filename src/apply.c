/* apply.c — structural GGUF pruning: materialize a .poeplan.
 *
 * The output is assembled in three parts. The metadata section is copied
 * byte-for-byte from the source mapping — only the <arch>.expert_count
 * value is patched in place (same key, same type, smaller number) and any
 * stale poe.* provenance keys from a previous run are dropped before fresh
 * ones are appended. The tensor table is re-serialized with the expert
 * dimension shrunk and the offsets recomputed. Payloads are streamed from
 * the source mmap: an untouched tensor is written whole, a sliced one is
 * written as its kept expert slabs in ascending order — which is exactly
 * the plan's remap rule, so router rows and expert slabs stay aligned.
 *
 * SPDX-License-Identifier: MIT */
#include "poe/apply.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot.h"
#include "ggufw.h"

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

/* ── apply ──────────────────────────────────────────────────────────────── */

/* the MoE block owning tensor `t` iff `t` carries the expert dimension */
static const poe_block *slice_owner(const poe_model *m, const ingot_tensor *t) {
    unsigned bidx;
    if (sscanf(t->name, "blk.%u.", &bidx) != 1 || bidx >= m->n_blocks)
        return NULL;
    const poe_block *blk = &m->blocks[bidx];
    if (!blk->is_moe) return NULL;
    if (t == blk->router_w    || t == blk->router_b    ||
        t == blk->gate_exps_w || t == blk->up_exps_w   || t == blk->down_exps_w ||
        t == blk->gate_exps_b || t == blk->up_exps_b   || t == blk->down_exps_b)
        return blk;
    return NULL;
}

int poe_apply(const poe_model *m, const poe_plan *p, const char *out_path,
              uint32_t top_k, int force, poe_apply_stats *stats,
              char *err, size_t errsz) {
    poe_apply_stats st = { 0, 0, 0, 0, 0, 0, 0 };
    if (stats) *stats = st;
    if (m == NULL || p == NULL || out_path == NULL)
        return fail(err, errsz, "bad arguments");

    const ingot_gguf *g = m->g;
    const uint32_t E = p->n_experts, K = p->keep_per_layer;

    if (ingot_gguf_shard_count(g) != 1)
        return fail(err, errsz, "split GGUF sources are not supported yet");
    if (strcmp(m->fingerprint, p->model_fingerprint) != 0 && !force)
        return fail(err, errsz, "plan fingerprint does not match the model "
                                "(use --force to override)");
    if (p->n_layers != m->n_blocks || E != m->expert_count || p->keep == NULL)
        return fail(err, errsz, "plan topology does not match the model");
    if (K == 0 || K > E)
        return fail(err, errsz, "plan keep_per_layer is invalid");
    if (top_k > K)
        return fail(err, errsz, "top_k cannot exceed the kept expert count");

    for (uint32_t l = 0; l < m->n_blocks; l++) {
        const poe_block *blk = &m->blocks[l];
        if (blk->is_legacy_split)
            return fail(err, errsz, "legacy per-expert tensor layout is not "
                                    "supported — re-export with packed experts");
        if (blk->is_moe && blk->expert_count != E)
            return fail(err, errsz, "a block has a different expert count "
                                    "than the plan");
    }

    /* resolve which tensors get sliced, and validate their geometry */
    const size_t T = ingot_gguf_count(g);
    const poe_block **owner = calloc(T, sizeof *owner);
    if (owner == NULL) return fail(err, errsz, "out of memory");

    uint64_t payload_after = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        owner[i] = slice_owner(m, t);
        if (owner[i] == NULL) { payload_after += t->nbytes; continue; }
        if (t->rank == 0 || t->ne[t->rank - 1] != E || t->nbytes % E != 0) {
            free(owner);
            snprintf(err, errsz, "tensor '%s' does not slice evenly along "
                     "the expert dimension", t->name);
            return -1;
        }
        payload_after += t->nbytes / E * K;
    }

    /* the plan's exact accounting must reproduce from this checkpoint */
    if (!force) {
        if (m->total_bytes != p->bytes_before) {
            free(owner);
            return fail(err, errsz, "plan byte accounting does not match "
                                    "the model");
        }
        if (payload_after != p->bytes_before - p->bytes_removed) {
            free(owner);
            return fail(err, errsz, "recomputed output size disagrees with "
                                    "the plan's accounting");
        }
    }

    /* raw metadata: locate every KV's byte span in the source mapping */
    const void *basep; size_t fsize;
    if (ingot_gguf_mapping(g, 0, &basep, &fsize) != 0) {
        free(owner);
        return fail(err, errsz, "cannot access the source mapping");
    }
    const uint8_t *base = basep;
    uint64_t hoff = 8, nten_src, nkv;
    if (fsize < 24 || memcmp(base, "GGUF", 4) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nten_src) != 0 ||
        poe_rd_u64(base, fsize, &hoff, &nkv) != 0 || nten_src != T) {
        free(owner);
        return fail(err, errsz, "source GGUF header is inconsistent");
    }

    poe_kv_span *spans = malloc((nkv ? nkv : 1) * sizeof *spans);
    if (spans == NULL) { free(owner); return fail(err, errsz, "out of memory"); }
    uint64_t kv_end = 24;
    if (poe_kv_walk(base, fsize, nkv, spans, &kv_end) != 0) {
        free(owner); free(spans);
        return fail(err, errsz, "source GGUF metadata does not parse");
    }

    /* in-place integer patches: expert_count, optionally expert_used_count */
    char eck[80], euk[80];
    snprintf(eck, sizeof eck, "%s.expert_count", m->arch);
    snprintf(euk, sizeof euk, "%s.expert_used_count", m->arch);
    size_t   ec_idx = (size_t)-1, eu_idx = (size_t)-1;
    uint64_t ec_width = 0, eu_width = 0;
    uint64_t kv_out_bytes = 0, nkv_out = 0;
    for (uint64_t i = 0; i < nkv; i++) {
        const poe_kv_span *s = &spans[i];
        if (s->keylen >= 4 && memcmp(s->key, "poe.", 4) == 0) {
            st.kv_dropped++;                    /* stale provenance */
            continue;
        }
        int is_ec = poe_kv_is(s, eck);
        int is_eu = top_k > 0 && poe_kv_is(s, euk);
        if (is_ec || is_eu) {
            uint64_t w = poe_kv_scalar_size(s->type);
            if (w == 0 || s->type == INGOT_KV_FLOAT32 ||
                s->type == INGOT_KV_FLOAT64 || s->type == INGOT_KV_BOOL) {
                free(owner); free(spans);
                return fail(err, errsz, "expert count metadata has an "
                                        "unexpected type");
            }
            if (is_ec) { ec_idx = (size_t)i; ec_width = w; }
            else       { eu_idx = (size_t)i; eu_width = w; }
        }
        kv_out_bytes += s->end - s->start;
        nkv_out++;
    }
    if (top_k > 0 && eu_idx == (size_t)-1) {
        free(owner); free(spans);
        return fail(err, errsz, "model has no expert_used_count metadata "
                                "to patch");
    }

    /* fresh provenance, serialized once so its size is known up front */
    poe_buf prov = { 0, 0, 0 };
    if (poe_buf_kv_str(&prov, "poe.version", POE_VERSION) != 0 ||
        poe_buf_kv_str(&prov, "poe.method", p->method) != 0 ||
        poe_buf_kv_str(&prov, "poe.source_fingerprint", m->fingerprint) != 0) {
        free(owner); free(spans); free(prov.p);
        return fail(err, errsz, "out of memory");
    }
    nkv_out += 3;
    kv_out_bytes += prov.n;

    /* re-serialized tensor table with recomputed offsets */
    uint64_t align = ingot_gguf_alignment(g);
    if (align == 0) align = 32;
    poe_buf infos = { 0, 0, 0 };
    uint64_t data_off = 0;
    int oom = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint64_t nb = owner[i] ? t->nbytes / E * K : t->nbytes;
        data_off = poe_align_up(data_off, align);
        oom |= poe_buf_str(&infos, t->name);
        oom |= poe_buf_u32(&infos, t->rank);
        for (uint32_t d = 0; d < t->rank; d++)
            oom |= poe_buf_u64(&infos, owner[i] && d == t->rank - 1 ? K : t->ne[d]);
        oom |= poe_buf_u32(&infos, (uint32_t)t->type);
        oom |= poe_buf_u64(&infos, data_off);
        data_off += nb;
    }
    if (oom) {
        free(owner); free(spans); free(prov.p); free(infos.p);
        return fail(err, errsz, "out of memory");
    }

    const uint64_t data_base = poe_align_up(24 + kv_out_bytes + infos.n, align);

    /* ── write ──────────────────────────────────────────────────────────── */
    FILE *f = fopen(out_path, "wb");
    if (f == NULL) {
        free(owner); free(spans); free(prov.p); free(infos.p);
        return fail(err, errsz, "cannot create the output file");
    }

    int rc = -1;
    {
        uint8_t head[24];
        memcpy(head, "GGUF", 4);
        poe_enc_le(head + 4,  ingot_gguf_version(g), 4);
        poe_enc_le(head + 8,  T, 8);
        poe_enc_le(head + 16, nkv_out, 8);
        if (poe_wput(f, head, sizeof head) != 0) goto done;
    }
    for (uint64_t i = 0; i < nkv; i++) {
        const poe_kv_span *s = &spans[i];
        if (s->keylen >= 4 && memcmp(s->key, "poe.", 4) == 0) continue;
        if ((size_t)i == ec_idx || (size_t)i == eu_idx) {
            int is_ec = (size_t)i == ec_idx;
            uint64_t w = is_ec ? ec_width : eu_width;
            uint8_t v[8];
            poe_enc_le(v, is_ec ? K : top_k, (size_t)w);
            if (poe_wput(f, base + s->start, (size_t)(s->val_off - s->start)) != 0 ||
                poe_wput(f, v, (size_t)w) != 0 ||
                poe_wput(f, base + s->val_off + w,
                     (size_t)(s->end - s->val_off - w)) != 0) goto done;
            if (is_ec) st.expert_count_patched = 1;
            else       st.top_k_patched = 1;
        } else {
            if (poe_wput(f, base + s->start, (size_t)(s->end - s->start)) != 0)
                goto done;
        }
    }
    if (poe_wput(f, prov.p, prov.n) != 0) goto done;
    if (poe_wput(f, infos.p, infos.n) != 0) goto done;
    if (poe_wpad(f, data_base - (24 + kv_out_bytes + infos.n)) != 0) goto done;

    /* payloads, streamed straight from the source mapping */
    uint64_t at = 0;
    for (size_t i = 0; i < T; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        const uint8_t *src = ingot_gguf_data(g, t);
        if (src == NULL) goto done;
        uint64_t aligned = poe_align_up(at, align);
        if (poe_wpad(f, aligned - at) != 0) goto done;
        at = aligned;
        if (owner[i] == NULL) {
            if (poe_wput(f, src, (size_t)t->nbytes) != 0) goto done;
            at += t->nbytes;
        } else {
            const uint64_t slab = t->nbytes / E;
            const uint32_t blk_idx = owner[i]->block;
            const uint8_t *keep = p->keep + (size_t)blk_idx * E;
            for (uint32_t e = 0; e < E; e++) {
                if (!keep[e]) continue;
                if (poe_wput(f, src + (uint64_t)e * slab, (size_t)slab) != 0)
                    goto done;
                at += slab;
            }
            st.tensors_sliced++;
        }
    }
    if (fflush(f) != 0 || ferror(f)) goto done;

    st.tensors_total  = (uint32_t)T;
    st.payload_bytes  = payload_after;
    st.bytes_written  = data_base + data_off;
    rc = 0;

done:
    if (rc != 0) {
        fclose(f);
        remove(out_path);
        fail(err, errsz, "writing the output file failed");
    } else if (fclose(f) != 0) {
        remove(out_path);
        rc = fail(err, errsz, "writing the output file failed");
    }
    free(owner); free(spans); free(prov.p); free(infos.p);
    if (rc == 0 && stats) *stats = st;
    return rc;
}
