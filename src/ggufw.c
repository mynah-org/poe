/* ggufw.c — GGUF rewriting primitives shared by apply and requant.
 * SPDX-License-Identifier: MIT */
#include "ggufw.h"

#include <string.h>

#include "ingot.h"

void poe_enc_le(uint8_t *dst, uint64_t v, size_t width) {
    for (size_t i = 0; i < width; i++) dst[i] = (uint8_t)(v >> (8 * i));
}

int poe_rd_u32(const uint8_t *b, uint64_t size, uint64_t *off, uint32_t *out) {
    if (*off + 4 > size) return -1;
    *out = (uint32_t)b[*off] | (uint32_t)b[*off + 1] << 8 |
           (uint32_t)b[*off + 2] << 16 | (uint32_t)b[*off + 3] << 24;
    *off += 4;
    return 0;
}

int poe_rd_u64(const uint8_t *b, uint64_t size, uint64_t *off, uint64_t *out) {
    if (*off + 8 > size) return -1;
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = v << 8 | b[*off + i];
    *out = v;
    *off += 8;
    return 0;
}

int poe_buf_put(poe_buf *b, const void *src, size_t n) {
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->n + n) cap *= 2;
        uint8_t *q = realloc(b->p, cap);
        if (q == NULL) return -1;
        b->p = q; b->cap = cap;
    }
    memcpy(b->p + b->n, src, n);
    b->n += n;
    return 0;
}

int poe_buf_u32(poe_buf *b, uint32_t v) {
    uint8_t t[4]; poe_enc_le(t, v, 4); return poe_buf_put(b, t, 4);
}

int poe_buf_u64(poe_buf *b, uint64_t v) {
    uint8_t t[8]; poe_enc_le(t, v, 8); return poe_buf_put(b, t, 8);
}

int poe_buf_str(poe_buf *b, const char *s) {
    size_t n = strlen(s);
    return poe_buf_u64(b, n) != 0 ? -1 : poe_buf_put(b, s, n);
}

/* string KV entry: key, type=STRING, value */
int poe_buf_kv_str(poe_buf *b, const char *key, const char *value) {
    if (poe_buf_str(b, key) != 0) return -1;
    if (poe_buf_u32(b, INGOT_KV_STRING) != 0) return -1;
    return poe_buf_str(b, value);
}

uint64_t poe_kv_scalar_size(uint32_t type) {
    switch (type) {
    case INGOT_KV_UINT8:  case INGOT_KV_INT8:  case INGOT_KV_BOOL:    return 1;
    case INGOT_KV_UINT16: case INGOT_KV_INT16:                        return 2;
    case INGOT_KV_UINT32: case INGOT_KV_INT32: case INGOT_KV_FLOAT32: return 4;
    case INGOT_KV_UINT64: case INGOT_KV_INT64: case INGOT_KV_FLOAT64: return 8;
    default:                                                          return 0;
    }
}

static int kv_skip_value(const uint8_t *b, uint64_t size, uint64_t *off,
                         uint32_t type) {
    uint64_t n;
    if (type == INGOT_KV_STRING) {
        if (poe_rd_u64(b, size, off, &n) != 0 || *off + n > size) return -1;
        *off += n;
        return 0;
    }
    if (type == INGOT_KV_ARRAY) {
        uint32_t elem;
        if (poe_rd_u32(b, size, off, &elem) != 0 ||
            poe_rd_u64(b, size, off, &n) != 0) return -1;
        if (elem == INGOT_KV_STRING) {
            for (uint64_t i = 0; i < n; i++)
                if (kv_skip_value(b, size, off, INGOT_KV_STRING) != 0) return -1;
            return 0;
        }
        uint64_t es = poe_kv_scalar_size(elem);
        if (es == 0 || n > (size - *off) / es) return -1;   /* nested/overflow */
        *off += n * es;
        return 0;
    }
    uint64_t es = poe_kv_scalar_size(type);
    if (es == 0 || *off + es > size) return -1;
    *off += es;
    return 0;
}

int poe_kv_walk(const uint8_t *b, uint64_t size, uint64_t nkv,
                poe_kv_span *spans, uint64_t *kv_end) {
    uint64_t off = 24;                            /* fixed GGUF v2/v3 header */
    for (uint64_t i = 0; i < nkv; i++) {
        poe_kv_span *s = &spans[i];
        s->start = off;
        if (poe_rd_u64(b, size, &off, &s->keylen) != 0 ||
            off + s->keylen > size) return -1;
        s->key = (const char *)b + off;
        off += s->keylen;
        if (poe_rd_u32(b, size, &off, &s->type) != 0) return -1;
        s->val_off = off;
        if (kv_skip_value(b, size, &off, s->type) != 0) return -1;
        s->end = off;
    }
    *kv_end = off;
    return 0;
}

int poe_kv_is(const poe_kv_span *s, const char *name) {
    return s->keylen == strlen(name) && memcmp(s->key, name, s->keylen) == 0;
}

uint64_t poe_align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

int poe_wput(FILE *f, const void *p, size_t n) {
    return n == 0 || fwrite(p, 1, n, f) == n ? 0 : -1;
}

int poe_wpad(FILE *f, uint64_t n) {
    static const uint8_t zero[512];
    while (n > 0) {
        size_t c = n < sizeof zero ? (size_t)n : sizeof zero;
        if (poe_wput(f, zero, c) != 0) return -1;
        n -= c;
    }
    return 0;
}
