/* test_apply.c — structural pruning: slicing, remap, metadata, determinism.
 *
 * Applies a reap-25% plan (prune {1,3}, keep {0,2,4,5,6,7}) to the MoE
 * fixture and checks the output at three levels: poe_model_open structure
 * and exact byte accounting, ingot-level metadata (patched expert_count,
 * preserved keys, fresh provenance), and byte-for-byte payload remap of
 * expert slabs and router rows against the source.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fixture.h"
#include "poe/apply.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* same ranking as test_plan: bottom-2 by reap = {3, 1} in every layer */
static int write_profile(const char *path, const char *fingerprint) {
    FILE *f = fopen(path, "w");
    if (f == NULL) return -1;
    fprintf(f, "{\n\"poeprofile\": 0, \"poe_version\": \"test\",\n");
    fprintf(f, "\"model_fingerprint\": \"%s\", \"arch\": \"%s\",\n",
            fingerprint, POE_FIX_ARCH);
    fprintf(f, "\"n_layers\": %u, \"n_experts\": %u, \"top_k\": %u,\n",
            POE_FIX_BLOCKS, POE_FIX_EXPERTS, POE_FIX_TOPK);
    fprintf(f, "\"dataset_hash\": \"fnv1a:0\", \"tokens\": 16384,\n");
    fprintf(f, "\"layers\": [\n");
    for (uint32_t l = 0; l < POE_FIX_BLOCKS; l++) {
        fprintf(f, "{\"layer\": %u, \"tokens\": 16384, \"entropy_bits_mean\": 2.0,\n", l);
        fprintf(f, " \"mass_k_mean\": {\"0.80\": 4, \"0.90\": 5, \"0.95\": 6, \"0.99\": 7},\n");
        fprintf(f, " \"sel_count\": [80,10,70,5,60,50,40,30],\n");
        fprintf(f, " \"gate_mean\": [0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.3],\n");
        fprintf(f, " \"reap_mean\": [0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.3],\n");
        fprintf(f, " \"actnorm_mean\": [1,1,1,1,1,1,1,1]}%s\n",
                l + 1 < POE_FIX_BLOCKS ? "," : "");
    }
    fprintf(f, "]\n}\n");
    fclose(f);
    return 0;
}

static int same_file(const char *pa, const char *pb) {
    FILE *fa = fopen(pa, "rb"), *fb = fopen(pb, "rb");
    if (!fa || !fb) { if (fa) fclose(fa); if (fb) fclose(fb); return 0; }
    int same = 1, ca, cb;
    do {
        ca = fgetc(fa); cb = fgetc(fb);
        if (ca != cb) { same = 0; break; }
    } while (ca != EOF);
    fclose(fa); fclose(fb);
    return same;
}

/* out tensor's slab j must equal src tensor's slab kept[j], byte for byte */
static int remap_ok(const ingot_gguf *gs, const ingot_gguf *go,
                    const char *name, const uint32_t *kept, uint32_t nkept) {
    const ingot_tensor *ts = ingot_gguf_find(gs, name);
    const ingot_tensor *to = ingot_gguf_find(go, name);
    if (!ts || !to) return 0;
    const uint64_t slab = ts->nbytes / POE_FIX_EXPERTS;
    if (to->nbytes != slab * nkept) return 0;
    const uint8_t *ps = ingot_gguf_data(gs, ts);
    const uint8_t *po = ingot_gguf_data(go, to);
    if (!ps || !po) return 0;
    for (uint32_t j = 0; j < nkept; j++)
        if (memcmp(po + (uint64_t)j * slab,
                   ps + (uint64_t)kept[j] * slab, (size_t)slab) != 0) return 0;
    return 1;
}

int main(void) {
    char err[256];
    static const uint32_t kept[6] = { 0, 2, 4, 5, 6, 7 };

    CHECK(poe_fixture_moe("build/apply-fix.gguf", 1, err, sizeof err) == 0,
          "fixture written");
    poe_model *m;
    CHECK(poe_model_open(&m, "build/apply-fix.gguf", err, sizeof err) == 0,
          "model open");
    CHECK(write_profile("build/apply.poeprofile", m->fingerprint) == 0,
          "profile written");
    poe_profile *pr;
    CHECK(poe_profile_load(&pr, "build/apply.poeprofile", err, sizeof err) == 0,
          "profile load: %s", err[0] ? err : "ok");

    const poe_profile *profs[1] = { pr };
    double w[1] = { 1.0 };
    poe_plan *p;
    CHECK(poe_plan_build(&p, m, profs, w, 1, "reap", 0.25, 0,
                         err, sizeof err) == 0, "plan reap 25%%");
    CHECK(p->keep_per_layer == 6, "plan keeps 6/8");

    /* ── apply ──────────────────────────────────────────────────────────── */
    printf("apply\n");
    poe_apply_stats st;
    CHECK(poe_apply(m, p, "build/apply-out.gguf", 0, &st,
                    err, sizeof err) == 0, "apply: %s", err[0] ? err : "ok");
    /* per block: router + gate/up/down exps = 4 sliced tensors */
    CHECK(st.tensors_sliced == 4 * POE_FIX_BLOCKS,
          "sliced %u tensors (expect %u)", st.tensors_sliced, 4 * POE_FIX_BLOCKS);
    CHECK(st.expert_count_patched == 1, "expert_count patched");
    CHECK(st.payload_bytes == p->bytes_before - p->bytes_removed,
          "payload bytes match the plan exactly");

    /* ── structure of the output ────────────────────────────────────────── */
    printf("structure\n");
    poe_model *o;
    CHECK(poe_model_open(&o, "build/apply-out.gguf", err, sizeof err) == 0,
          "output reopens: %s", err[0] ? err : "ok");
    CHECK(o->n_blocks == m->n_blocks && o->n_moe_blocks == m->n_moe_blocks,
          "block structure preserved");
    CHECK(o->expert_count == 6, "expert_count is 6 (%u)", o->expert_count);
    CHECK(o->experts_per_token == POE_FIX_TOPK, "top_k untouched");
    CHECK(o->total_bytes == p->bytes_before - p->bytes_removed,
          "exact payload accounting (%llu)", (unsigned long long)o->total_bytes);

    /* ── metadata ───────────────────────────────────────────────────────── */
    printf("metadata\n");
    const ingot_kv *kv = ingot_gguf_kv_find(o->g, POE_FIX_ARCH ".expert_count");
    uint64_t u = 0;
    CHECK(kv != NULL && ingot_kv_u64(kv, &u) == 0 && u == 6,
          "metadata expert_count = 6");
    const char *s = NULL;
    kv = ingot_gguf_kv_find(o->g, "general.name");
    CHECK(kv != NULL && ingot_kv_str(kv, &s) == 0 &&
          strcmp(s, "poe synthetic moe") == 0, "unrelated metadata preserved");
    kv = ingot_gguf_kv_find(o->g, "poe.source_fingerprint");
    CHECK(kv != NULL && ingot_kv_str(kv, &s) == 0 &&
          strcmp(s, m->fingerprint) == 0, "provenance records the source");
    kv = ingot_gguf_kv_find(o->g, "poe.method");
    CHECK(kv != NULL && ingot_kv_str(kv, &s) == 0 && strcmp(s, "reap") == 0,
          "provenance records the method");

    /* ── payload remap ──────────────────────────────────────────────────── */
    printf("remap\n");
    char name[64];
    int slabs_ok = 1, rows_ok = 1;
    for (uint32_t b = 0; b < POE_FIX_BLOCKS; b++) {
        static const char *const exps[] = {
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
        };
        for (size_t i = 0; i < sizeof exps / sizeof exps[0]; i++) {
            snprintf(name, sizeof name, "blk.%u.%s", b, exps[i]);
            if (!remap_ok(m->g, o->g, name, kept, 6)) slabs_ok = 0;
        }
        snprintf(name, sizeof name, "blk.%u.ffn_gate_inp.weight", b);
        if (!remap_ok(m->g, o->g, name, kept, 6)) rows_ok = 0;
    }
    CHECK(slabs_ok, "kept expert slabs identical to the source, in keep order");
    CHECK(rows_ok, "router rows compacted in the same order");

    /* untouched tensors byte-identical */
    const ingot_tensor *ts = ingot_gguf_find(m->g, "token_embd.weight");
    const ingot_tensor *to = ingot_gguf_find(o->g, "token_embd.weight");
    CHECK(ts && to && ts->nbytes == to->nbytes &&
          memcmp(ingot_gguf_data(m->g, ts), ingot_gguf_data(o->g, to),
                 (size_t)ts->nbytes) == 0, "untouched tensors copied verbatim");
    poe_model_close(o);

    /* ── determinism ────────────────────────────────────────────────────── */
    printf("determinism\n");
    CHECK(poe_apply(m, p, "build/apply-out2.gguf", 0, &st,
                    err, sizeof err) == 0, "second apply");
    CHECK(same_file("build/apply-out.gguf", "build/apply-out2.gguf"),
          "same plan -> byte-identical output");

    /* ── guards ─────────────────────────────────────────────────────────── */
    printf("guards\n");
    char saved[24];
    memcpy(saved, p->model_fingerprint, sizeof saved);
    snprintf(p->model_fingerprint, sizeof p->model_fingerprint,
             "poe1:ffffffffffffffff");
    CHECK(poe_apply(m, p, "build/apply-bad.gguf", 0, &st,
                    err, sizeof err) != 0, "fingerprint mismatch rejected: %s", err);
    CHECK(poe_apply(m, p, "build/apply-bad.gguf", 1, &st,
                    err, sizeof err) == 0, "--force overrides");
    memcpy(p->model_fingerprint, saved, sizeof saved);

    CHECK(poe_fixture_dense("build/apply-dense.gguf", err, sizeof err) == 0,
          "dense fixture written");
    poe_model *d;
    CHECK(poe_model_open(&d, "build/apply-dense.gguf", err, sizeof err) == 0,
          "dense open");
    CHECK(poe_apply(d, p, "build/apply-bad.gguf", 1, &st,
                    err, sizeof err) != 0, "dense model rejected: %s", err);
    poe_model_close(d);

    poe_plan_free(p);
    poe_profile_free(pr);
    poe_model_close(m);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
