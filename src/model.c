/* model.c — MoE structure discovery over a GGUF opened through ingot.
 *
 * Two passes over the tensor table: the first sizes the block array (the
 * metadata block_count wins, but a file whose tensors go past it still
 * opens), the second classifies every tensor into expert / router / shared /
 * other and accumulates exact on-disk byte counts.
 *
 * Naming conventions recognized (llama.cpp GGUF exports):
 *   blk.N.ffn_gate_inp.{weight,bias}       router
 *   blk.N.exp_probs_b.bias                 router bias (DeepSeek-V3 style)
 *   blk.N.ffn_{gate,up,down}_exps.weight   packed routed experts, ne[2] = E
 *   blk.N.ffn_{gate,up,down}_exps.bias     per-expert biases (gpt-oss style)
 *   blk.N.ffn_{gate,up,down}.E.weight      legacy per-expert split tensors
 *   blk.N.ffn_*_shexp.*, ffn_gate_inp_shexp.*   shared expert
 *
 * SPDX-License-Identifier: MIT */
#include "poe/poe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── error helper ───────────────────────────────────────────────────────── */

static int fail(char *err, size_t errsz, const char *msg, const char *detail) {
    if (err && errsz) {
        if (detail) snprintf(err, errsz, "%s: %s", msg, detail);
        else        snprintf(err, errsz, "%s", msg);
    }
    return -1;
}

/* ── fingerprint ────────────────────────────────────────────────────────── */

#define FNV64_INIT  0xcbf29ce484222325ULL
#define FNV64_PRIME 0x00000100000001b3ULL

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= FNV64_PRIME; }
    return h;
}
static uint64_t fnv1a_str(uint64_t h, const char *s) {
    return fnv1a(h, s, strlen(s) + 1);   /* include NUL: "ab","c" != "a","bc" */
}
static uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    return fnv1a(h, b, 8);
}

/* Hash the structure (tensor table + metadata) plus the first bytes of every
 * tensor payload. Sampling one page per tensor keeps this effectively free
 * even on a multi-hundred-GB mmap while still distinguishing two fine-tunes
 * that share an identical layout. */
static void fingerprint(const ingot_gguf *g, char *dst, size_t dstsz) {
    uint64_t h = FNV64_INIT;

    h = fnv1a_str(h, ingot_gguf_arch(g));

    size_t nt = ingot_gguf_count(g);
    h = fnv1a_u64(h, nt);
    for (size_t i = 0; i < nt; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        h = fnv1a_str(h, t->name);
        h = fnv1a_u64(h, (uint64_t)t->type);
        h = fnv1a_u64(h, t->rank);
        for (uint32_t d = 0; d < t->rank; d++) h = fnv1a_u64(h, t->ne[d]);
        h = fnv1a_u64(h, t->nbytes);

        const void *data = ingot_gguf_data(g, t);
        if (data && t->nbytes)
            h = fnv1a(h, data, t->nbytes < 64 ? (size_t)t->nbytes : 64);
    }

    size_t nk = ingot_gguf_kv_count(g);
    h = fnv1a_u64(h, nk);
    for (size_t i = 0; i < nk; i++) {
        const ingot_kv *kv = ingot_gguf_kv_at(g, i);
        h = fnv1a_str(h, ingot_kv_key(kv));
        h = fnv1a_u64(h, (uint64_t)ingot_kv_type(kv));

        const char *s; uint64_t u; double f;
        if (ingot_kv_str(kv, &s) == 0)      h = fnv1a_str(h, s);
        else if (ingot_kv_u64(kv, &u) == 0) h = fnv1a_u64(h, u);
        else if (ingot_kv_f64(kv, &f) == 0) h = fnv1a(h, &f, sizeof f);
        else {                               /* array: hash type + length */
            uint64_t len = 0;
            (void)ingot_kv_arr_len(kv, &len);
            h = fnv1a_u64(h, (uint64_t)ingot_kv_arr_type(kv));
            h = fnv1a_u64(h, len);
        }
    }

    snprintf(dst, dstsz, "poe1:%016llx", (unsigned long long)h);
}

/* ── metadata helpers ───────────────────────────────────────────────────── */

static int arch_u32(const ingot_gguf *g, const char *arch, const char *suffix,
                    uint32_t *out) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    uint64_t v;
    if (kv == NULL || ingot_kv_u64(kv, &v) != 0) return -1;
    *out = (uint32_t)v;
    return 0;
}

/* ── tensor name parsing ────────────────────────────────────────────────── */

/* "blk.<N>.<rest>" -> block index + rest. -1 when not a block tensor. */
static int parse_block(const char *name, uint32_t *block, const char **rest) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    char *end;
    unsigned long v = strtoul(name + 4, &end, 10);
    if (end == name + 4 || *end != '.') return -1;
    *block = (uint32_t)v;
    *rest  = end + 1;
    return 0;
}

/* "ffn_gate.<E>.weight" (legacy split experts) -> expert index, or -1. */
static int parse_legacy_expert(const char *rest, uint32_t *expert) {
    static const char *const prefixes[] = { "ffn_gate.", "ffn_up.", "ffn_down." };
    for (size_t i = 0; i < sizeof prefixes / sizeof prefixes[0]; i++) {
        size_t n = strlen(prefixes[i]);
        if (strncmp(rest, prefixes[i], n) != 0) continue;
        char *end;
        unsigned long v = strtoul(rest + n, &end, 10);
        if (end == rest + n || strcmp(end, ".weight") != 0) continue;
        *expert = (uint32_t)v;
        return 0;
    }
    return -1;
}

static int name_is(const char *rest, const char *want) {
    return strcmp(rest, want) == 0;
}
static int name_has_prefix(const char *rest, const char *prefix) {
    return strncmp(rest, prefix, strlen(prefix)) == 0;
}

/* ── open / close ───────────────────────────────────────────────────────── */

int poe_model_open(poe_model **out, const char *path, char *err, size_t errsz) {
    if (out == NULL) return -1;
    *out = NULL;

    ingot_gguf *g;
    if (ingot_gguf_open_split(&g, path, err, errsz) != 0) return -1;

    poe_model *m = calloc(1, sizeof *m);
    if (m == NULL) { ingot_gguf_close(g); return fail(err, errsz, "out of memory", NULL); }
    m->g = g;

    snprintf(m->arch, sizeof m->arch, "%s", ingot_gguf_arch(g));

    const ingot_kv *kv = ingot_gguf_kv_find(g, "general.name");
    const char *s;
    if (kv && ingot_kv_str(kv, &s) == 0) snprintf(m->name, sizeof m->name, "%s", s);

    (void)arch_u32(g, m->arch, "block_count",                &m->n_blocks);
    (void)arch_u32(g, m->arch, "expert_count",               &m->expert_count);
    (void)arch_u32(g, m->arch, "expert_used_count",          &m->experts_per_token);
    (void)arch_u32(g, m->arch, "expert_shared_count",        &m->shared_expert_count);
    (void)arch_u32(g, m->arch, "embedding_length",           &m->embedding_length);
    (void)arch_u32(g, m->arch, "expert_feed_forward_length", &m->expert_ff_length);

    /* Pass 1: how many blocks do the tensors actually cover? A file whose
     * tensors go past block_count still opens — the tensors win. */
    size_t nt = ingot_gguf_count(g);
    uint32_t max_block = 0;
    int seen_block = 0;
    for (size_t i = 0; i < nt; i++) {
        uint32_t b; const char *rest;
        if (parse_block(ingot_gguf_at(g, i)->name, &b, &rest) == 0) {
            if (b > max_block) max_block = b;
            seen_block = 1;
        }
    }
    if (seen_block && max_block + 1 > m->n_blocks) m->n_blocks = max_block + 1;

    if (m->n_blocks > 0) {
        m->blocks = calloc(m->n_blocks, sizeof *m->blocks);
        if (m->blocks == NULL) {
            poe_model_close(m);
            return fail(err, errsz, "out of memory", NULL);
        }
        for (uint32_t b = 0; b < m->n_blocks; b++) m->blocks[b].block = b;
    }

    /* Pass 2: classify every tensor and account exact bytes. */
    for (size_t i = 0; i < nt; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        uint32_t b; const char *rest;

        if (parse_block(t->name, &b, &rest) != 0 || b >= m->n_blocks) {
            m->other_bytes += t->nbytes;
            continue;
        }
        poe_block *blk = &m->blocks[b];

        if (name_is(rest, "ffn_gate_inp.weight")) {
            blk->router_w = t;
            blk->router_bytes += t->nbytes;
        } else if (name_is(rest, "ffn_gate_inp.bias") ||
                   name_is(rest, "exp_probs_b.bias")) {
            blk->router_b = t;
            blk->router_bytes += t->nbytes;
        } else if (name_has_prefix(rest, "ffn_gate_exps.") ||
                   name_has_prefix(rest, "ffn_up_exps.")   ||
                   name_has_prefix(rest, "ffn_down_exps.")) {
            blk->is_moe = 1;
            blk->expert_bytes += t->nbytes;

            int is_weight = strstr(rest, ".weight") != NULL;
            if (name_has_prefix(rest, "ffn_gate_exps.")) {
                if (is_weight) blk->gate_exps_w = t; else blk->gate_exps_b = t;
            } else if (name_has_prefix(rest, "ffn_up_exps.")) {
                if (is_weight) blk->up_exps_w = t;   else blk->up_exps_b = t;
            } else {
                if (is_weight) blk->down_exps_w = t; else blk->down_exps_b = t;
            }
            /* The packed layout carries the expert count in the outermost
             * (slowest) dimension. */
            if (t->rank == 3 && t->ne[2] > blk->expert_count)
                blk->expert_count = (uint32_t)t->ne[2];
        } else if (name_has_prefix(rest, "ffn_gate_shexp.") ||
                   name_has_prefix(rest, "ffn_up_shexp.")   ||
                   name_has_prefix(rest, "ffn_down_shexp.") ||
                   name_has_prefix(rest, "ffn_gate_inp_shexp.")) {
            blk->has_shared_expert = 1;
            blk->shared_bytes += t->nbytes;
        } else {
            uint32_t e;
            if (parse_legacy_expert(rest, &e) == 0) {
                blk->is_moe = 1;
                blk->is_legacy_split = 1;
                blk->expert_bytes += t->nbytes;
                if (e + 1 > blk->expert_count) blk->expert_count = e + 1;
            } else {
                m->other_bytes += t->nbytes;
            }
        }
    }

    /* Roll blocks up into the model totals. */
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (blk->is_moe) {
            m->n_moe_blocks++;
            if (blk->expert_count > m->expert_count)
                m->expert_count = blk->expert_count;
        }
        m->expert_bytes += blk->expert_bytes;
        m->router_bytes += blk->router_bytes;
        m->shared_bytes += blk->shared_bytes;
    }
    m->total_bytes = m->expert_bytes + m->router_bytes +
                     m->shared_bytes + m->other_bytes;

    fingerprint(g, m->fingerprint, sizeof m->fingerprint);

    *out = m;
    return 0;
}

void poe_model_close(poe_model *m) {
    if (m == NULL) return;
    free(m->blocks);
    if (m->g) ingot_gguf_close(m->g);
    free(m);
}
