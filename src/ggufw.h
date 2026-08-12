/* ggufw.h — the internals of writing a GGUF that is a rewrite of another one.
 *
 * Not a general writer (ingot has one, and it holds every payload in memory
 * until save). This is the streaming shape POE needs: copy the source's
 * metadata byte-for-byte except for the few values that change, re-serialize
 * the tensor table with new shapes or types, then stream payloads one tensor
 * at a time. A 20 GB checkpoint is rewritten with a slab-sized buffer.
 *
 * Internal to src/; not part of the public API.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_GGUFW_H
#define POE_GGUFW_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ── little-endian primitives (host-order independent) ─────────────────── */

void poe_enc_le(uint8_t *dst, uint64_t v, size_t width);
int  poe_rd_u32(const uint8_t *b, uint64_t size, uint64_t *off, uint32_t *out);
int  poe_rd_u64(const uint8_t *b, uint64_t size, uint64_t *off, uint64_t *out);

/* ── growable output buffer (for the re-serialized tensor table) ────────── */

typedef struct { uint8_t *p; size_t n, cap; } poe_buf;

int poe_buf_put(poe_buf *b, const void *src, size_t n);
int poe_buf_u32(poe_buf *b, uint32_t v);
int poe_buf_u64(poe_buf *b, uint64_t v);
int poe_buf_str(poe_buf *b, const char *s);
int poe_buf_kv_str(poe_buf *b, const char *key, const char *value);

/* ── raw metadata walk ──────────────────────────────────────────────────── */

/* One KV entry's byte span within the source file: [start, end), with the
 * value bytes beginning at val_off (right after key + type tag). */
typedef struct {
    uint64_t    start, val_off, end;
    uint32_t    type;
    const char *key;
    uint64_t    keylen;
} poe_kv_span;

uint64_t poe_kv_scalar_size(uint32_t type);
int      poe_kv_walk(const uint8_t *b, uint64_t size, uint64_t nkv,
                     poe_kv_span *spans, uint64_t *kv_end);
int      poe_kv_is(const poe_kv_span *s, const char *name);

/* ── output ─────────────────────────────────────────────────────────────── */

uint64_t poe_align_up(uint64_t v, uint64_t a);
int      poe_wput(FILE *f, const void *p, size_t n);
int      poe_wpad(FILE *f, uint64_t n);

#endif
