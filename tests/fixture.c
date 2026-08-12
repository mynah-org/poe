/* fixture.c — build the synthetic GGUF fixtures through the ingot writer.
 * SPDX-License-Identifier: MIT */
#include "fixture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ingot.h"

/* The writer keeps pointers until save(), so every buffer lives in this list
 * until the file is on disk. */
typedef struct { float **v; size_t n, cap; } arena;

static float *arr(arena *a, size_t nelem, uint32_t seed) {
    float *p = malloc(nelem * sizeof *p);
    if (p == NULL) abort();
    uint32_t s = seed * 2654435761u + 12345u;
    for (size_t i = 0; i < nelem; i++) {
        s = s * 1664525u + 1013904223u;               /* deterministic LCG */
        p[i] = (float)(int32_t)s / 2147483648.0f * 0.05f;
    }
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->v = realloc(a->v, a->cap * sizeof *a->v);
        if (a->v == NULL) abort();
    }
    a->v[a->n++] = p;
    return p;
}

static void arena_free(arena *a) {
    for (size_t i = 0; i < a->n; i++) free(a->v[i]);
    free(a->v);
    memset(a, 0, sizeof *a);
}

static int add_f32(ingot_gguf_writer *w, arena *a, const char *name,
                   uint32_t rank, const uint64_t *ne, uint32_t seed) {
    uint64_t nelem = 1;
    for (uint32_t d = 0; d < rank; d++) nelem *= ne[d];
    return ingot_gguf_add_tensor(w, name, INGOT_TYPE_F32, rank, ne,
                                 arr(a, nelem, seed));
}

/* One builder for both MoE fixtures. The wide one exists because K-quant
 * rows must be a whole number of 256-element blocks: the small fixture's
 * 32-wide rows cannot hold a Q4_K tensor at all, so anything that
 * re-encodes weights needs a model with realistic row widths. */
static int moe_build(const char *path, uint32_t seed, uint32_t blocks,
                     uint64_t embd, uint64_t ff, uint64_t nexp, uint32_t topk,
                     const char *label, char *err, size_t errsz) {
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (w == NULL) { snprintf(err, errsz, "writer alloc failed"); return -1; }
    arena a = { 0 };
    int rc = -1;
    char name[128];

    ingot_gguf_kv_string(w, "general.architecture", POE_FIX_ARCH);
    ingot_gguf_kv_string(w, "general.name", label);
    ingot_gguf_kv_u32(w, POE_FIX_ARCH ".block_count",                blocks);
    ingot_gguf_kv_u32(w, POE_FIX_ARCH ".embedding_length",           (uint32_t)embd);
    ingot_gguf_kv_u32(w, POE_FIX_ARCH ".expert_count",               (uint32_t)nexp);
    ingot_gguf_kv_u32(w, POE_FIX_ARCH ".expert_used_count",          topk);
    ingot_gguf_kv_u32(w, POE_FIX_ARCH ".expert_feed_forward_length", (uint32_t)ff);

    const uint64_t vocab = POE_FIX_VOCAB;
    uint32_t s = seed * 7919u;

    {
        uint64_t ne[2] = { embd, vocab };
        if (add_f32(w, &a, "token_embd.weight", 2, ne, s++) != 0) goto done;
    }
    {
        uint64_t ne[1] = { embd };
        if (add_f32(w, &a, "output_norm.weight", 1, ne, s++) != 0) goto done;
    }

    for (uint32_t b = 0; b < blocks; b++) {
        uint64_t ne1[1]  = { embd };
        uint64_t neqq[2] = { embd, embd };
        uint64_t nert[2] = { embd, nexp };
        uint64_t negu[3] = { embd, ff, nexp };   /* gate/up: [embd, ff, E]  */
        uint64_t nedn[3] = { ff, embd, nexp };   /* down:    [ff, embd, E]  */

        static const char *const attn[] = {
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"
        };
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", b);
        if (add_f32(w, &a, name, 1, ne1, s++) != 0) goto done;
        for (size_t i = 0; i < sizeof attn / sizeof attn[0]; i++) {
            snprintf(name, sizeof name, "blk.%u.%s", b, attn[i]);
            if (add_f32(w, &a, name, 2, neqq, s++) != 0) goto done;
        }
        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", b);
        if (add_f32(w, &a, name, 1, ne1, s++) != 0) goto done;

        snprintf(name, sizeof name, "blk.%u.ffn_gate_inp.weight", b);
        if (add_f32(w, &a, name, 2, nert, s++) != 0) goto done;
        snprintf(name, sizeof name, "blk.%u.ffn_gate_exps.weight", b);
        if (add_f32(w, &a, name, 3, negu, s++) != 0) goto done;
        snprintf(name, sizeof name, "blk.%u.ffn_up_exps.weight", b);
        if (add_f32(w, &a, name, 3, negu, s++) != 0) goto done;
        snprintf(name, sizeof name, "blk.%u.ffn_down_exps.weight", b);
        if (add_f32(w, &a, name, 3, nedn, s++) != 0) goto done;
    }

    rc = ingot_gguf_writer_save(w, path, err, errsz);
done:
    if (rc != 0 && err && errsz && err[0] == '\0')
        snprintf(err, errsz, "fixture tensor add failed");
    ingot_gguf_writer_free(w);
    arena_free(&a);
    return rc;
}

int poe_fixture_moe(const char *path, uint32_t seed, char *err, size_t errsz) {
    return moe_build(path, seed, POE_FIX_BLOCKS, POE_FIX_EMBD, POE_FIX_FF,
                     POE_FIX_EXPERTS, POE_FIX_TOPK, "poe synthetic moe",
                     err, errsz);
}

int poe_fixture_moe_wide(const char *path, uint32_t seed, char *err, size_t errsz) {
    return moe_build(path, seed, POE_FIX_WIDE_BLOCKS, POE_FIX_WIDE_EMBD,
                     POE_FIX_WIDE_FF, POE_FIX_WIDE_EXPERTS, POE_FIX_TOPK,
                     "poe synthetic moe (quantizable rows)", err, errsz);
}

int poe_fixture_dense(const char *path, char *err, size_t errsz) {
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (w == NULL) { snprintf(err, errsz, "writer alloc failed"); return -1; }
    arena a = { 0 };
    int rc = -1;
    char name[128];

    ingot_gguf_kv_string(w, "general.architecture", "llama");
    ingot_gguf_kv_string(w, "general.name", "poe synthetic dense");
    ingot_gguf_kv_u32(w, "llama.block_count", 2);
    ingot_gguf_kv_u32(w, "llama.embedding_length", POE_FIX_EMBD);

    const uint64_t embd = POE_FIX_EMBD, vocab = POE_FIX_VOCAB, ff = 64;
    uint32_t s = 42;

    {
        uint64_t ne[2] = { embd, vocab };
        if (add_f32(w, &a, "token_embd.weight", 2, ne, s++) != 0) goto done;
    }
    for (uint32_t b = 0; b < 2; b++) {
        uint64_t neqq[2] = { embd, embd };
        uint64_t neup[2] = { embd, ff };
        uint64_t nedn[2] = { ff, embd };
        snprintf(name, sizeof name, "blk.%u.attn_q.weight", b);
        if (add_f32(w, &a, name, 2, neqq, s++) != 0) goto done;
        snprintf(name, sizeof name, "blk.%u.ffn_up.weight", b);
        if (add_f32(w, &a, name, 2, neup, s++) != 0) goto done;
        snprintf(name, sizeof name, "blk.%u.ffn_down.weight", b);
        if (add_f32(w, &a, name, 2, nedn, s++) != 0) goto done;
    }

    rc = ingot_gguf_writer_save(w, path, err, errsz);
done:
    if (rc != 0 && err && errsz && err[0] == '\0')
        snprintf(err, errsz, "fixture tensor add failed");
    ingot_gguf_writer_free(w);
    arena_free(&a);
    return rc;
}
