/* poe_profile.c — MoE router profiler over llama.cpp (milestone M2).
 *
 *   poe-profile <model.gguf> --dataset text-or-jsonl [-o out.poeprofile]
 *               [--max-tokens N] [--batch N] [--ngl N] [--threads N]
 *
 * Observes, per token and per MoE layer, the router distribution, the
 * selected expert IDs and the applied gate weights, through llama.cpp's
 * public scheduler eval callback (llama_context_params.cb_eval) — the same
 * mechanism llama-imatrix uses; no llama.cpp patches. Statistics stream
 * into poe_accum (O(1) in tokens) and are written as a versioned JSON
 * .poeprofile. Design notes: docs/router-observer.md.
 *
 * The dataset is scored prefill-style in independent chunks of --batch
 * tokens (memory cleared between chunks), which amortizes the per-tensor
 * scheduler syncs over the whole batch.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include "poe/poe.h"
#include "../src/profiler/accum.h"
#include "../src/profiler/imatrix.h"
#include "../src/profiler/stability.h"

#define MAX_WINDOWS 4096

/* ── observer state, shared with the callback ───────────────────────────── */

typedef struct {
    poe_accum acc;
    uint32_t  n_layers, n_experts, top_k;
    int       probs_are_logits;      /* delayed-softmax arch (gpt-oss)      */
    int       capture_down;          /* --metric reap: expert-output norms  */
    const char *down_kind;           /* "down" or "down_biased" (gpt-oss)   */
    uint64_t  bad_ids;
    uint64_t  reap_skipped;          /* down seen without staged selection  */
    float    *norms;                 /* [top_k * batch] scratch             */

    /* per-layer staging for the current decode: ids and best weights.
     * probs are consumed immediately (entropy/mass need nothing else). */
    int32_t  *ids;                   /* [n_layers][top_k * batch]           */
    float    *wts;                   /* [n_layers][top_k * batch]           */
    int      *wts_prio;              /* best weights variant seen so far    */
    uint32_t *stage_T;               /* tokens staged per layer             */
    uint32_t  batch_cap;

    /* --metric imatrix: per-expert activation statistics. The expert matmul
     * inputs are a different tensor from anything the routing path reads, so
     * they get their own buffers rather than sharing the scratch. */
    poe_imatrix imat;
    int       capture_imat;
    uint64_t  imat_skipped;          /* nodes whose input could not be read */
    float    *imat_act;
    size_t    imat_act_cap;
    int32_t  *imat_ids;
    size_t    imat_ids_cap;
    void     *imat_scratch;          /* strided input copies                */
    size_t    imat_scratch_cap;

    /* scratch for strided device→host copies */
    void     *scratch;
    size_t    scratch_cap;

    uint64_t  tensors_seen;
} observer;

/* Returns the buffer to use, or NULL if it could not be grown — the caller
 * assigns it back, so a failed realloc leaves the old buffer intact. */
static void *grow(void *buf, size_t *cap, size_t n) {
    if (n <= *cap) return buf;
    void *p = realloc(buf, n);
    if (p == NULL) return NULL;
    *cap = n;
    return p;
}

static void *scratch_for(observer *ob, size_t n) {
    if (n > ob->scratch_cap) {
        ob->scratch = realloc(ob->scratch, n);
        ob->scratch_cap = ob->scratch ? n : 0;
    }
    return ob->scratch;
}

/* Copy a (possibly non-contiguous) 2-D view [ne0 × ne1] to a contiguous
 * host buffer of elem size `esz`. ggml views (ffn_moe_topk is a view of
 * argsort) have nb[1] larger than ne[0]*esz, so fetch the backing span once
 * and stride-walk on the host — one D2H copy instead of ne1. */
static int fetch_2d(observer *ob, const struct ggml_tensor *t,
                    void *dst, size_t esz) {
    const uint32_t ne0 = (uint32_t)t->ne[0], ne1 = (uint32_t)t->ne[1];
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dst, 0, (size_t)ne0 * ne1 * esz);
        return 0;
    }
    size_t span = (size_t)t->nb[1] * (ne1 - 1) + (size_t)ne0 * t->nb[0];
    unsigned char *raw = scratch_for(ob, span);
    if (raw == NULL) return -1;
    ggml_backend_tensor_get(t, raw, 0, span);
    for (uint32_t c = 0; c < ne1; c++)
        memcpy((unsigned char *)dst + (size_t)c * ne0 * esz,
               raw + (size_t)c * t->nb[1], (size_t)ne0 * esz);
    return 0;
}

/* Copy an [ne0 × ne1 × ne2] f32 tensor to a contiguous host buffer. The
 * expert-matmul input is usually contiguous; when it is a view, fetch the
 * backing span once and stride-walk — the same trick as fetch_2d, but with
 * its own scratch so fetching the ids afterwards cannot clobber it. */
static int fetch_act(observer *ob, const struct ggml_tensor *t, float *dst) {
    const size_t ne0 = (size_t)t->ne[0], ne1 = (size_t)t->ne[1],
                 ne2 = (size_t)t->ne[2];
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dst, 0, ne0 * ne1 * ne2 * sizeof(float));
        return 0;
    }
    if (t->nb[0] != sizeof(float)) return -1;      /* rows must be dense */
    const size_t span = (size_t)t->nb[2] * (ne2 - 1) +
                        (size_t)t->nb[1] * (ne1 - 1) + ne0 * sizeof(float);
    unsigned char *raw = grow(ob->imat_scratch, &ob->imat_scratch_cap, span);
    if (raw == NULL) return -1;
    ob->imat_scratch = raw;
    ggml_backend_tensor_get(t, raw, 0, span);
    for (size_t z = 0; z < ne2; z++)
        for (size_t y = 0; y < ne1; y++)
            memcpy(dst + (z * ne1 + y) * ne0,
                   raw + z * (size_t)t->nb[2] + y * (size_t)t->nb[1],
                   ne0 * sizeof(float));
    return 0;
}

/* Which routed projection a graph node belongs to, or -1. Only the three
 * MUL_MAT_ID nodes carry expert inputs worth an imatrix. */
static int imat_proj_of(const char *kind) {
    if (strcmp(kind, "gate") == 0) return POE_IMAT_GATE;
    if (strcmp(kind, "up")   == 0) return POE_IMAT_UP;
    if (strcmp(kind, "down") == 0) return POE_IMAT_DOWN;
    return -1;
}

/* "ffn_moe_<kind>-<layer>" -> kind + layer. Returns 0 on match. */
static int parse_moe_name(const char *name, char *kind, size_t kindsz,
                          uint32_t *layer) {
    if (strncmp(name, "ffn_moe_", 8) != 0) return -1;
    const char *dash = strrchr(name, '-');
    if (dash == NULL || dash <= name + 8) return -1;
    size_t n = (size_t)(dash - (name + 8));
    if (n + 1 > kindsz) return -1;
    memcpy(kind, name + 8, n);
    kind[n] = '\0';
    char *end;
    unsigned long v = strtoul(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0') return -1;
    *layer = (uint32_t)v;
    return 0;
}

static void flush_layer(observer *ob, uint32_t l) {
    if (ob->stage_T[l] == 0) return;
    poe_accum_observe_selection(&ob->acc, l, ob->stage_T[l],
                                ob->ids + (size_t)l * ob->top_k * ob->batch_cap,
                                ob->wts_prio[l] > 0
                                    ? ob->wts + (size_t)l * ob->top_k * ob->batch_cap
                                    : NULL,
                                &ob->bad_ids);
    ob->stage_T[l] = 0;
    ob->wts_prio[l] = 0;
}

static void flush_all(observer *ob) {
    for (uint32_t l = 0; l < ob->n_layers; l++) flush_layer(ob, l);
}

/* Applied-gates priority: take the last variant in graph order. */
static int weights_prio(const char *kind) {
    if (strcmp(kind, "weights_norm") == 0)    return 3;
    if (strcmp(kind, "weights_softmax") == 0) return 2;
    if (strcmp(kind, "weights_scaled") == 0)  return 2;
    if (strcmp(kind, "weights") == 0)         return 1;
    return 0;
}

static bool moe_cb(struct ggml_tensor *t, bool ask, void *ud) {
    observer *ob = ud;
    char kind[32];
    uint32_t layer;

    if (parse_moe_name(t->name, kind, sizeof kind, &layer) != 0)
        return !ask;                       /* not ours; never abort compute */
    if (layer >= ob->n_layers) return !ask;

    int is_probs = strcmp(kind, "probs") == 0;
    int is_topk  = strcmp(kind, "topk") == 0;
    int is_down  = ob->capture_down && strcmp(kind, ob->down_kind) == 0;
    int wprio    = weights_prio(kind);
    int iproj    = (ob->capture_imat && t->op == GGML_OP_MUL_MAT_ID)
                 ? imat_proj_of(kind) : -1;

    if (ask) return is_probs || is_topk || is_down || iproj >= 0 || wprio > 0;

    ob->tensors_seen++;

    /* The imatrix reads this node's INPUT, so it coexists with the REAP
     * capture of the down node's output: fold it in, then fall through. */
    if (iproj >= 0) {
        const struct ggml_tensor *src1 = t->src[1], *ids = t->src[2];
        if (src1 == NULL || ids == NULL || src1->type != GGML_TYPE_F32 ||
            ids->ne[1] != src1->ne[2]) {
            ob->imat_skipped++;
        } else {
            const size_t cols  = (size_t)src1->ne[0];
            const size_t rows  = (size_t)src1->ne[1];
            const size_t nt    = (size_t)src1->ne[2];
            const size_t nused = (size_t)ids->ne[0];
            float *act = grow(ob->imat_act, &ob->imat_act_cap,
                              cols * rows * nt * sizeof *ob->imat_act);
            if (act != NULL) ob->imat_act = act;
            int32_t *idb = grow(ob->imat_ids, &ob->imat_ids_cap,
                                nused * nt * sizeof *ob->imat_ids);
            if (idb != NULL) ob->imat_ids = idb;

            if (act == NULL || idb == NULL ||
                fetch_act(ob, src1, act) != 0 ||
                fetch_2d(ob, ids, idb, sizeof(int32_t)) != 0 ||
                poe_imatrix_observe(&ob->imat, layer, (poe_imat_proj)iproj,
                                    (uint32_t)cols, (uint32_t)rows,
                                    (uint32_t)nused, (uint32_t)nt,
                                    act, idb, &ob->bad_ids) != 0)
                ob->imat_skipped++;
        }
    }

    if (is_probs) {
        /* [n_experts × T] */
        uint32_t T = (uint32_t)t->ne[1];
        float *buf = scratch_for(ob, (size_t)ob->n_experts * T * sizeof(float));
        if (buf && fetch_2d(ob, t, buf, sizeof(float)) == 0)
            poe_accum_observe_probs(&ob->acc, layer, T, buf,
                                    ob->probs_are_logits);
        return true;
    }

    if (is_down) {
        /* [n_embd × top_k × T], contiguous product of the expert matmul.
         * The router tensors for this layer precede it in the graph, so
         * the staged selection is already present. */
        uint32_t ne0 = (uint32_t)t->ne[0];
        uint32_t k   = (uint32_t)t->ne[1];
        uint32_t T   = (uint32_t)t->ne[2];
        if (k != ob->top_k || T > ob->batch_cap ||
            ob->stage_T[layer] != T || ob->wts_prio[layer] == 0) {
            ob->reap_skipped++;
            return true;
        }
        float *buf = scratch_for(ob, (size_t)ne0 * k * T * sizeof(float));
        if (buf == NULL) { ob->reap_skipped++; return true; }
        ggml_backend_tensor_get(t, buf, 0, (size_t)ne0 * k * T * sizeof(float));
        for (size_t i = 0; i < (size_t)k * T; i++) {
            const float *v = buf + i * ne0;
            double s = 0.0;
            for (uint32_t d = 0; d < ne0; d++) s += (double)v[d] * (double)v[d];
            ob->norms[i] = (float)sqrt(s);
        }
        poe_accum_observe_reap(&ob->acc, layer, T,
                               ob->ids + (size_t)layer * ob->top_k * ob->batch_cap,
                               ob->wts + (size_t)layer * ob->top_k * ob->batch_cap,
                               ob->norms, &ob->bad_ids);
        return true;
    }

    /* topk and weights arrive as [top_k × T] (weights sometimes [1,k,T];
     * collapse leading unit dims by using nelem/top_k as T). */
    uint32_t T = (uint32_t)(ggml_nelements(t) / ob->top_k);
    if (T > ob->batch_cap) T = ob->batch_cap;

    if (is_topk) {
        /* a new selection for this layer: flush the previous one first */
        flush_layer(ob, layer);
        int32_t *dst = ob->ids + (size_t)layer * ob->top_k * ob->batch_cap;
        if (fetch_2d(ob, t, dst, sizeof(int32_t)) == 0)
            ob->stage_T[layer] = T;
        return true;
    }

    if (wprio > ob->wts_prio[layer]) {
        float *dst = ob->wts + (size_t)layer * ob->top_k * ob->batch_cap;
        struct ggml_tensor flat = *t;   /* view as 2-D [k × T] for fetch */
        if (t->ne[0] == 1 && t->ne[1] == (int64_t)ob->top_k) {
            flat.ne[0] = t->ne[1]; flat.nb[0] = t->nb[1];
            flat.ne[1] = t->ne[2]; flat.nb[1] = t->nb[2];
            flat.ne[2] = 1;
        }
        if (fetch_2d(ob, &flat, dst, sizeof(float)) == 0)
            ob->wts_prio[layer] = wprio;
    }
    return true;
}

/* ── dataset loading ────────────────────────────────────────────────────── */

/* Read a whole file; for .jsonl, concatenate the "text" string fields
 * (minimal JSON string unescaping: \" \\ \/ \n \t \r; \uXXXX -> '?'). */
static char *load_dataset(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *raw = malloc((size_t)sz + 1);
    if (raw == NULL || fread(raw, 1, (size_t)sz, f) != (size_t)sz) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);
    raw[sz] = '\0';

    size_t n = strlen(path);
    if (n < 6 || strcmp(path + n - 6, ".jsonl") != 0) {
        *out_len = (size_t)sz;
        return raw;                         /* plain text */
    }

    char *out = malloc((size_t)sz + 1);
    if (out == NULL) { free(raw); return NULL; }
    size_t o = 0;
    const char *p = raw;
    while ((p = strstr(p, "\"text\"")) != NULL) {
        p += 6;
        while (*p == ' ' || *p == ':') p++;
        if (*p != '"') continue;
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) {
                p++;
                switch (*p) {
                    case 'n': out[o++] = '\n'; break;
                    case 't': out[o++] = '\t'; break;
                    case 'r': out[o++] = '\r'; break;
                    case 'u': out[o++] = '?';
                              for (int i = 0; i < 4 && p[1]; i++) p++;
                              break;
                    default:  out[o++] = *p;   break;
                }
                p++;
            } else out[o++] = *p++;
        }
        out[o++] = '\n'; out[o++] = '\n';
    }
    free(raw);
    if (o == 0) { free(out); return NULL; }
    out[o] = '\0';
    *out_len = o;
    return out;
}

static uint64_t fnv1a64(const void *data, size_t n) {
    const unsigned char *b = data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x00000100000001b3ULL; }
    return h;
}

/* ── main ───────────────────────────────────────────────────────────────── */

static void usage(void) {
    fprintf(stderr,
        "usage: poe-profile <model.gguf> --dataset <file.txt|file.jsonl>\n"
        "                   [--metric routing|reap|imatrix] [-o out.poeprofile]\n"
        "                   [--imatrix-out out.imatrix.gguf]\n"
        "                   [--max-tokens N] [--batch N] [--ngl N] [--threads N]\n"
        "                   [--until-stable S] [--stable-windows K] [--bottom F]\n"
        "\n"
        "  --metric routing   selection counts, gate stats, entropy, mass-K (cheap)\n"
        "  --metric reap      the above plus expert-output norms for REAP saliency\n"
        "                     (captures ffn_moe_down: slower, ~n_embd*k*4 B/token/layer)\n"
        "  --metric imatrix   routing plus per-expert activation statistics, written\n"
        "                     as llama.cpp's GGUF imatrix (feeds llama-quantize\n"
        "                     --imatrix, and any bit allocation that weights by\n"
        "                     what the calibration actually exercised)\n"
        "\n"
        "convergence (checked once per batch of --batch tokens):\n"
        "  --until-stable S   stop early when the bottom-F prune set's Jaccard\n"
        "                     between consecutive checks stays >= S (e.g. 0.99)\n"
        "  --stable-windows K consecutive checks required (default 2)\n"
        "  --bottom F         prune-decision fraction (default 0.25)\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *dataset_path = NULL, *out_path = NULL;
    const char *metric = "routing", *imat_path = NULL;
    long max_tokens = 8192, n_batch = 2048, ngl = 999, n_threads = 0;
    double until_stable = 0.0, bottom_frac = 0.25;
    long stable_windows = 2;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) dataset_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)        out_path = argv[++i];
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc)  metric = argv[++i];
        else if (strcmp(argv[i], "--imatrix-out") == 0 && i + 1 < argc) imat_path = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atol(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)   n_batch = atol(argv[++i]);
        else if (strcmp(argv[i], "--ngl") == 0 && i + 1 < argc)     ngl = atol(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = atol(argv[++i]);
        else if (strcmp(argv[i], "--until-stable") == 0 && i + 1 < argc) until_stable = atof(argv[++i]);
        else if (strcmp(argv[i], "--stable-windows") == 0 && i + 1 < argc) stable_windows = atol(argv[++i]);
        else if (strcmp(argv[i], "--bottom") == 0 && i + 1 < argc)  bottom_frac = atof(argv[++i]);
        else if (argv[i][0] != '-' && model_path == NULL)           model_path = argv[i];
        else { usage(); return 2; }
    }
    if (until_stable < 0.0 || until_stable > 1.0 ||
        bottom_frac <= 0.0 || bottom_frac > 1.0 || stable_windows < 1) {
        fprintf(stderr, "poe-profile: bad convergence parameters\n");
        return 2;
    }
    if (model_path == NULL || dataset_path == NULL) { usage(); return 2; }
    if (strcmp(metric, "routing") != 0 && strcmp(metric, "gate") != 0 &&
        strcmp(metric, "reap") != 0 && strcmp(metric, "imatrix") != 0) {
        fprintf(stderr, "poe-profile: unknown metric '%s'\n", metric);
        return 2;
    }
    if (imat_path != NULL && strcmp(metric, "imatrix") != 0) {
        fprintf(stderr, "poe-profile: --imatrix-out needs --metric imatrix\n");
        return 2;
    }

    /* Static discovery through ingot first: layer/expert topology and the
     * model fingerprint the profile gets bound to. */
    char err[256];
    poe_model *pm;
    if (poe_model_open(&pm, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe-profile: %s\n", err);
        return 1;
    }
    if (pm->n_moe_blocks == 0 || pm->experts_per_token == 0) {
        fprintf(stderr, "poe-profile: %s is not a MoE model\n", model_path);
        poe_model_close(pm);
        return 1;
    }

    observer ob = { 0 };
    ob.n_layers  = pm->n_blocks;
    ob.n_experts = pm->expert_count;
    ob.top_k     = pm->experts_per_token;
    ob.probs_are_logits = strcmp(pm->arch, "gpt-oss") == 0;
    ob.capture_down = strcmp(metric, "reap") == 0;
    ob.capture_imat = strcmp(metric, "imatrix") == 0;
    if (ob.capture_imat &&
        poe_imatrix_init(&ob.imat, ob.n_layers, ob.n_experts) != 0) {
        fprintf(stderr, "poe-profile: cannot initialise the imatrix\n");
        return 1;
    }
    /* gpt-oss experts carry biases: the applied output is the biased one */
    ob.down_kind = ob.probs_are_logits ? "down_biased" : "down";
    ob.batch_cap = (uint32_t)n_batch;
    if (poe_accum_init(&ob.acc, ob.n_layers, ob.n_experts, ob.top_k) != 0) {
        fprintf(stderr, "poe-profile: out of memory\n");
        return 1;
    }
    ob.ids      = malloc((size_t)ob.n_layers * ob.top_k * ob.batch_cap * sizeof *ob.ids);
    ob.wts      = malloc((size_t)ob.n_layers * ob.top_k * ob.batch_cap * sizeof *ob.wts);
    ob.wts_prio = calloc(ob.n_layers, sizeof *ob.wts_prio);
    ob.stage_T  = calloc(ob.n_layers, sizeof *ob.stage_T);
    ob.norms    = malloc((size_t)ob.top_k * ob.batch_cap * sizeof *ob.norms);
    if (!ob.ids || !ob.wts || !ob.wts_prio || !ob.stage_T || !ob.norms) {
        fprintf(stderr, "poe-profile: out of memory\n");
        return 1;
    }

    size_t ds_len = 0;
    char *dataset = load_dataset(dataset_path, &ds_len);
    if (dataset == NULL) {
        fprintf(stderr, "poe-profile: cannot read dataset '%s'\n", dataset_path);
        return 1;
    }
    uint64_t ds_hash = fnv1a64(dataset, ds_len);

    /* llama.cpp setup */
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = (int)ngl;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (model == NULL) {
        fprintf(stderr, "poe-profile: llama.cpp failed to load '%s'\n", model_path);
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t)n_batch;
    cp.n_batch  = (uint32_t)n_batch;
    cp.n_ubatch = (uint32_t)n_batch;
    if (n_threads > 0) { cp.n_threads = (int)n_threads; cp.n_threads_batch = (int)n_threads; }
    cp.cb_eval           = moe_cb;
    cp.cb_eval_user_data = &ob;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) {
        fprintf(stderr, "poe-profile: llama.cpp context creation failed\n");
        return 1;
    }

    /* tokenize the whole corpus once */
    int cap = (int)(ds_len + 16);
    llama_token *toks = malloc((size_t)cap * sizeof *toks);
    int n_toks = llama_tokenize(vocab, dataset, (int32_t)ds_len, toks, cap, true, false);
    if (n_toks <= 0) {
        fprintf(stderr, "poe-profile: tokenization failed (%d)\n", n_toks);
        return 1;
    }
    if (max_tokens > 0 && n_toks > max_tokens) n_toks = (int)max_tokens;
    fprintf(stderr, "poe-profile: %d calibration tokens, batch %ld, %u layers x %u experts (top-%u)\n",
            n_toks, n_batch, ob.n_layers, ob.n_experts, ob.top_k);

    /* convergence tracking: one check per batch */
    poe_stability st;
    if (poe_stability_init(&st, ob.n_layers, ob.n_experts, bottom_frac) != 0) {
        fprintf(stderr, "poe-profile: out of memory\n");
        return 1;
    }
    double *scores = malloc((size_t)ob.n_layers * ob.n_experts * sizeof *scores);
    struct { int tokens; double jaccard, spearman; } hist[MAX_WINDOWS];
    int n_hist = 0, stable_streak = 0, stopped_early = 0;

    if (until_stable > 0.0)
        fprintf(stderr, "  convergence: stop at bottom-%.0f%% Jaccard >= %.3f "
                        "for %ld checks\n",
                bottom_frac * 100.0, until_stable, stable_windows);

    /* prefill-style scoring in independent chunks */
    time_t t0 = time(NULL);
    int done = 0;
    while (done < n_toks) {
        int n = n_toks - done < (int)n_batch ? n_toks - done : (int)n_batch;
        struct llama_batch batch = llama_batch_get_one(toks + done, n);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "poe-profile: llama_decode failed at token %d\n", done);
            return 1;
        }
        flush_all(&ob);
        llama_memory_clear(llama_get_memory(ctx), true);
        done += n;

        poe_accum_scores(&ob.acc, scores);
        poe_stability_step step;
        if (scores && poe_stability_update(&st, scores, &step) == 0) {
            if (n_hist < MAX_WINDOWS) {
                hist[n_hist].tokens   = done;
                hist[n_hist].jaccard  = step.bottom_jaccard;
                hist[n_hist].spearman = step.spearman;
                n_hist++;
            }
            fprintf(stderr, "\r  %d/%d tokens   bottom-set stability %.3f   "
                            "rank corr %.3f\n", done, n_toks,
                    step.bottom_jaccard, step.spearman);
            if (until_stable > 0.0) {
                stable_streak = step.bottom_jaccard >= until_stable
                              ? stable_streak + 1 : 0;
                if (stable_streak >= stable_windows) {
                    stopped_early = 1;
                    fprintf(stderr, "  converged at %d tokens\n", done);
                    break;
                }
            }
        } else {
            fprintf(stderr, "\r  %d/%d tokens", done, n_toks);
        }
    }
    long elapsed = (long)(time(NULL) - t0);
    fprintf(stderr, "\n  %lds, %llu moe tensors observed, %llu bad expert ids",
            elapsed, (unsigned long long)ob.tensors_seen,
            (unsigned long long)ob.bad_ids);
    if (ob.capture_down)
        fprintf(stderr, ", %llu reap captures skipped",
                (unsigned long long)ob.reap_skipped);
    if (ob.capture_imat)
        fprintf(stderr, ", %llu imatrix captures skipped",
                (unsigned long long)ob.imat_skipped);
    fprintf(stderr, "\n");

    if (ob.tensors_seen == 0) {
        fprintf(stderr, "poe-profile: no ffn_moe_* tensors observed — "
                        "llama.cpp graph names changed? (see docs/router-observer.md)\n");
        return 1;
    }

    /* write the profile */
    char defout[512];
    if (out_path == NULL) {
        snprintf(defout, sizeof defout, "%s.poeprofile", model_path);
        out_path = defout;
    }
    FILE *f = fopen(out_path, "w");
    if (f == NULL) {
        fprintf(stderr, "poe-profile: cannot write '%s'\n", out_path);
        return 1;
    }
    fprintf(f, "{\n  \"poeprofile\": 0,\n");
    fprintf(f, "  \"poe_version\": \"%s\",\n", POE_VERSION);
    fprintf(f, "  \"model_fingerprint\": \"%s\",\n", pm->fingerprint);
    fprintf(f, "  \"arch\": \"%s\",\n", pm->arch);
    fprintf(f, "  \"n_layers\": %u, \"n_experts\": %u, \"top_k\": %u,\n",
            ob.n_layers, ob.n_experts, ob.top_k);
    fprintf(f, "  \"dataset_hash\": \"fnv1a:%016llx\",\n",
            (unsigned long long)ds_hash);
    fprintf(f, "  \"dataset_bytes\": %zu,\n", ds_len);
    fprintf(f, "  \"tokens\": %d,\n", n_toks);
    fprintf(f, "  \"metrics\": [\"routing\", \"gate\"%s%s],\n",
            ob.capture_down ? ", \"reap\"" : "",
            ob.capture_imat ? ", \"imatrix\"" : "");
    fprintf(f, "  \"tokens_scored\": %d,\n", done);
    fprintf(f, "  \"convergence\": {\n");
    fprintf(f, "    \"bottom_fraction\": %.4f,\n", bottom_frac);
    fprintf(f, "    \"target\": %.4f,\n", until_stable);
    fprintf(f, "    \"stopped_early\": %s,\n", stopped_early ? "true" : "false");
    fprintf(f, "    \"history\": [");
    for (int i = 0; i < n_hist; i++)
        fprintf(f, "%s\n      {\"tokens\": %d, \"bottom_jaccard\": %.6f, "
                   "\"spearman\": %.6f}", i ? "," : "",
                hist[i].tokens, hist[i].jaccard, hist[i].spearman);
    fprintf(f, "\n    ]\n  },\n");
    fprintf(f, "  \"wall_seconds\": %ld,\n", elapsed);
    fprintf(f, "  \"layers\":\n");
    poe_accum_write_json(&ob.acc, f, "  ");
    fprintf(f, "\n}\n");
    fclose(f);
    fprintf(stderr, "poe-profile: wrote %s\n", out_path);

    if (ob.capture_imat) {
        char imdef[512];
        if (imat_path == NULL) {
            snprintf(imdef, sizeof imdef, "%s.imatrix.gguf", model_path);
            imat_path = imdef;
        }
        if (!poe_imatrix_has_data(&ob.imat)) {
            fprintf(stderr, "poe-profile: no expert activations were captured — "
                            "no imatrix written\n");
        } else {
            /* Chunks are this runner's batches, which is what chunk_size
             * means to a reader: the token count one accumulation spans. */
            const uint32_t chunks = (uint32_t)((done + n_batch - 1) / n_batch);
            if (poe_imatrix_write_gguf(&ob.imat, imat_path, dataset_path,
                                       chunks, (uint32_t)n_batch,
                                       err, sizeof err) != 0)
                fprintf(stderr, "poe-profile: cannot write '%s': %s\n",
                        imat_path, err);
            else
                fprintf(stderr, "poe-profile: wrote %s\n", imat_path);
        }
        poe_imatrix_free(&ob.imat);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    poe_model_close(pm);
    return 0;
}
