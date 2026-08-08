/* json.c — minimal JSON reader. Recursive descent, plain malloc DOM.
 * SPDX-License-Identifier: MIT */
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct poe_json {
    poe_json_type kind;
    /* NUM */   double num; uint64_t u64; int u64_exact;
    /* STR */   char *str;
    /* ARR/OBJ */ struct poe_json **items; char **keys; size_t n, cap;
    /* BOOL */  int boolean;
};

typedef struct {
    const char *p, *end, *base;
    char *err; size_t errsz;
    int depth;
} parser;

#define MAX_DEPTH 64

static void perr(parser *ps, const char *msg) {
    if (ps->err && ps->errsz)
        snprintf(ps->err, ps->errsz, "json: %s at byte %zu",
                 msg, (size_t)(ps->p - ps->base));
}

static void skip_ws(parser *ps) {
    while (ps->p < ps->end &&
           (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r'))
        ps->p++;
}

static poe_json *node_new(poe_json_type k) {
    poe_json *j = calloc(1, sizeof *j);
    if (j) j->kind = k;
    return j;
}

void poe_json_free(poe_json *j) {
    if (j == NULL) return;
    free(j->str);
    for (size_t i = 0; i < j->n; i++) {
        poe_json_free(j->items[i]);
        if (j->keys) free(j->keys[i]);
    }
    free(j->items);
    free(j->keys);
    free(j);
}

static int push(poe_json *j, char *key, poe_json *child) {
    if (j->n == j->cap) {
        size_t cap = j->cap ? j->cap * 2 : 8;
        void *ni = realloc(j->items, cap * sizeof *j->items);
        if (ni == NULL) return -1;
        j->items = ni;
        if (j->kind == POE_JSON_OBJ) {
            void *nk = realloc(j->keys, cap * sizeof *j->keys);
            if (nk == NULL) return -1;
            j->keys = nk;
        }
        j->cap = cap;
    }
    j->items[j->n] = child;
    if (j->keys) j->keys[j->n] = key;
    j->n++;
    return 0;
}

static poe_json *parse_value(parser *ps);

static char *parse_string_raw(parser *ps) {
    /* ps->p at opening quote */
    ps->p++;
    size_t cap = 16, o = 0;
    char *s = malloc(cap);
    if (s == NULL) return NULL;

    while (ps->p < ps->end && *ps->p != '"') {
        if (o + 4 >= cap) {
            cap *= 2;
            char *ns = realloc(s, cap);
            if (ns == NULL) { free(s); return NULL; }
            s = ns;
        }
        char c = *ps->p++;
        if (c == '\\' && ps->p < ps->end) {
            char e = *ps->p++;
            switch (e) {
                case 'n': s[o++] = '\n'; break;
                case 't': s[o++] = '\t'; break;
                case 'r': s[o++] = '\r'; break;
                case 'b': s[o++] = '\b'; break;
                case 'f': s[o++] = '\f'; break;
                case 'u': {
                    unsigned v = 0; int ok = 1;
                    for (int i = 0; i < 4; i++) {
                        if (ps->p >= ps->end) { ok = 0; break; }
                        char h = *ps->p++;
                        v <<= 4;
                        if      (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                        else ok = 0;
                    }
                    if (!ok) { s[o++] = '?'; break; }
                    /* UTF-8 encode the BMP code point (surrogates -> '?') */
                    if (v < 0x80) s[o++] = (char)v;
                    else if (v < 0x800) {
                        s[o++] = (char)(0xC0 | (v >> 6));
                        s[o++] = (char)(0x80 | (v & 0x3F));
                    } else if (v >= 0xD800 && v <= 0xDFFF) {
                        s[o++] = '?';
                    } else {
                        s[o++] = (char)(0xE0 | (v >> 12));
                        s[o++] = (char)(0x80 | ((v >> 6) & 0x3F));
                        s[o++] = (char)(0x80 | (v & 0x3F));
                    }
                    break;
                }
                default: s[o++] = e; break;   /* covers \" \\ \/ */
            }
        } else {
            s[o++] = c;
        }
    }
    if (ps->p >= ps->end) { free(s); return NULL; }
    ps->p++;                                   /* closing quote */
    s[o] = '\0';
    return s;
}

static poe_json *parse_number(parser *ps) {
    const char *start = ps->p;
    char *end;
    double d = strtod(start, &end);
    if (end == start || end > ps->end) { perr(ps, "bad number"); return NULL; }

    poe_json *j = node_new(POE_JSON_NUM);
    if (j == NULL) return NULL;
    j->num = d;

    /* exact u64 when the token is plain digits (counters) */
    if (*start != '-') {
        uint64_t v = 0; int exact = 1;
        for (const char *q = start; q < end; q++) {
            if (*q < '0' || *q > '9') { exact = 0; break; }
            if (v > (UINT64_MAX - (uint64_t)(*q - '0')) / 10) { exact = 0; break; }
            v = v * 10 + (uint64_t)(*q - '0');
        }
        j->u64 = v; j->u64_exact = exact;
    }
    ps->p = end;
    return j;
}

static poe_json *parse_value(parser *ps) {
    if (++ps->depth > MAX_DEPTH) { perr(ps, "nesting too deep"); return NULL; }
    skip_ws(ps);
    if (ps->p >= ps->end) { perr(ps, "unexpected end"); ps->depth--; return NULL; }

    poe_json *j = NULL;
    char c = *ps->p;

    if (c == '{' || c == '[') {
        int obj = c == '{';
        char close = obj ? '}' : ']';
        j = node_new(obj ? POE_JSON_OBJ : POE_JSON_ARR);
        if (j == NULL) { ps->depth--; return NULL; }
        ps->p++;
        skip_ws(ps);
        if (ps->p < ps->end && *ps->p == close) { ps->p++; ps->depth--; return j; }
        for (;;) {
            char *key = NULL;
            if (obj) {
                skip_ws(ps);
                if (ps->p >= ps->end || *ps->p != '"') {
                    perr(ps, "expected object key"); goto fail;
                }
                key = parse_string_raw(ps);
                if (key == NULL) { perr(ps, "bad key string"); goto fail; }
                skip_ws(ps);
                if (ps->p >= ps->end || *ps->p != ':') {
                    free(key); perr(ps, "expected ':'"); goto fail;
                }
                ps->p++;
            }
            poe_json *child = parse_value(ps);
            if (child == NULL) { free(key); goto fail; }
            if (push(j, key, child) != 0) {
                free(key); poe_json_free(child);
                perr(ps, "out of memory"); goto fail;
            }
            skip_ws(ps);
            if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
            if (ps->p < ps->end && *ps->p == close) { ps->p++; break; }
            perr(ps, obj ? "expected ',' or '}'" : "expected ',' or ']'");
            goto fail;
        }
        ps->depth--;
        return j;
    }

    if (c == '"') {
        j = node_new(POE_JSON_STR);
        if (j == NULL) { ps->depth--; return NULL; }
        j->str = parse_string_raw(ps);
        if (j->str == NULL) { perr(ps, "bad string"); goto fail; }
        ps->depth--;
        return j;
    }
    if (c == 't' && ps->end - ps->p >= 4 && strncmp(ps->p, "true", 4) == 0) {
        j = node_new(POE_JSON_BOOL); if (j) j->boolean = 1;
        ps->p += 4; ps->depth--; return j;
    }
    if (c == 'f' && ps->end - ps->p >= 5 && strncmp(ps->p, "false", 5) == 0) {
        j = node_new(POE_JSON_BOOL);
        ps->p += 5; ps->depth--; return j;
    }
    if (c == 'n' && ps->end - ps->p >= 4 && strncmp(ps->p, "null", 4) == 0) {
        j = node_new(POE_JSON_NULL);
        ps->p += 4; ps->depth--; return j;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        j = parse_number(ps);
        ps->depth--;
        return j;
    }
    perr(ps, "unexpected character");
fail:
    poe_json_free(j);
    ps->depth--;
    return NULL;
}

poe_json *poe_json_parse(const char *text, size_t len, char *err, size_t errsz) {
    parser ps = { text, text + len, text, err, errsz, 0 };
    if (err && errsz) err[0] = '\0';
    poe_json *j = parse_value(&ps);
    if (j == NULL) return NULL;
    skip_ws(&ps);
    if (ps.p != ps.end) {
        perr(&ps, "trailing content");
        poe_json_free(j);
        return NULL;
    }
    return j;
}

poe_json_type poe_json_kind(const poe_json *j) {
    return j ? j->kind : POE_JSON_NULL;
}

const poe_json *poe_json_get(const poe_json *obj, const char *key) {
    if (obj == NULL || obj->kind != POE_JSON_OBJ) return NULL;
    for (size_t i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
    return NULL;
}

const poe_json *poe_json_at(const poe_json *arr, size_t index) {
    if (arr == NULL || (arr->kind != POE_JSON_ARR && arr->kind != POE_JSON_OBJ))
        return NULL;
    return index < arr->n ? arr->items[index] : NULL;
}

size_t poe_json_len(const poe_json *j) {
    if (j == NULL || (j->kind != POE_JSON_ARR && j->kind != POE_JSON_OBJ)) return 0;
    return j->n;
}

double poe_json_num(const poe_json *j, double dflt) {
    return (j && j->kind == POE_JSON_NUM) ? j->num : dflt;
}

uint64_t poe_json_u64(const poe_json *j, uint64_t dflt) {
    if (j == NULL || j->kind != POE_JSON_NUM) return dflt;
    return j->u64_exact ? j->u64 : (uint64_t)j->num;
}

const char *poe_json_str(const poe_json *j, const char *dflt) {
    return (j && j->kind == POE_JSON_STR) ? j->str : dflt;
}
