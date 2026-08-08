/* test_plan.c — .poeplan building, determinism, guards, and roundtrip.
 *
 * Uses the synthetic MoE fixture (4 blocks × 8 experts, top-2, F32) and a
 * matching hand-written profile whose REAP scores make the prune decision
 * obvious: experts 3 and 1 are the bottom-2 of every layer. All accounting
 * numbers are derived from the fixture geometry, never hardcoded blindly.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fixture.h"
#include "poe/plan.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* profile matching the MoE fixture: reap scores rank experts
 * 0 > 2 > 4 > 5 > 6 > 7 > 1 > 3, so bottom-2 = {3, 1} in every layer */
static int write_profile(const char *path, const char *fingerprint,
                         int with_reap, uint64_t tokens) {
    FILE *f = fopen(path, "w");
    if (f == NULL) return -1;
    fprintf(f, "{\n\"poeprofile\": 0, \"poe_version\": \"test\",\n");
    fprintf(f, "\"model_fingerprint\": \"%s\", \"arch\": \"%s\",\n",
            fingerprint, POE_FIX_ARCH);
    fprintf(f, "\"n_layers\": %u, \"n_experts\": %u, \"top_k\": %u,\n",
            POE_FIX_BLOCKS, POE_FIX_EXPERTS, POE_FIX_TOPK);
    fprintf(f, "\"dataset_hash\": \"fnv1a:0\", \"tokens\": %llu,\n",
            (unsigned long long)tokens);
    fprintf(f, "\"layers\": [\n");
    for (uint32_t l = 0; l < POE_FIX_BLOCKS; l++) {
        fprintf(f, "{\"layer\": %u, \"tokens\": %llu, \"entropy_bits_mean\": 2.0,\n",
                l, (unsigned long long)tokens);
        fprintf(f, " \"mass_k_mean\": {\"0.80\": 4, \"0.90\": 5, \"0.95\": 6, \"0.99\": 7},\n");
        fprintf(f, " \"sel_count\": [80,10,70,5,60,50,40,30],\n");
        fprintf(f, " \"gate_mean\": [0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.3]");
        if (with_reap)
            fprintf(f, ",\n \"reap_mean\": [0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.3],\n"
                       " \"actnorm_mean\": [1,1,1,1,1,1,1,1]");
        fprintf(f, "}%s\n", l + 1 < POE_FIX_BLOCKS ? "," : "");
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

int main(void) {
    char err[256];

    CHECK(poe_fixture_moe("build/plan-fix.gguf", 1, err, sizeof err) == 0,
          "fixture written");
    poe_model *m;
    CHECK(poe_model_open(&m, "build/plan-fix.gguf", err, sizeof err) == 0,
          "model open");

    CHECK(write_profile("build/plan.poeprofile", m->fingerprint, 1, 16384) == 0,
          "matching profile written");
    poe_profile *pr;
    CHECK(poe_profile_load(&pr, "build/plan.poeprofile", err, sizeof err) == 0,
          "profile load: %s", err[0] ? err : "ok");

    /* ── build: reap, 25% ───────────────────────────────────────────────── */
    printf("build\n");
    const poe_profile *profs[1] = { pr };
    double w[1] = { 1.0 };
    poe_plan *p;
    CHECK(poe_plan_build(&p, m, profs, w, 1, "reap", 0.25, 0,
                         err, sizeof err) == 0, "build reap 25%%: %s",
          err[0] ? err : "ok");
    CHECK(p->keep_per_layer == 6, "keep 6/8 per layer (%u)", p->keep_per_layer);
    int prune_right = 1;
    for (uint32_t l = 0; l < p->n_layers; l++) {
        const uint8_t *k = p->keep + (size_t)l * 8;
        if (k[1] || k[3]) prune_right = 0;                 /* pruned */
        if (!k[0] || !k[2] || !k[4] || !k[5] || !k[6] || !k[7]) prune_right = 0;
    }
    CHECK(prune_right, "bottom-2 by reap ({1,3}) pruned in every layer");

    /* exact accounting from the fixture geometry (F32 = 4 B/param):
     * expert slice/blk = 3*EMBD*FF*4, router row = EMBD*4 */
    const uint64_t slice_b = (uint64_t)3 * POE_FIX_EMBD * POE_FIX_FF * 4;
    const uint64_t row_b   = (uint64_t)POE_FIX_EMBD * 4;
    const uint64_t exp_rm  = 2ull * (slice_b + row_b) * POE_FIX_BLOCKS;
    CHECK(p->bytes_removed == exp_rm, "bytes removed %llu (expect %llu)",
          (unsigned long long)p->bytes_removed, (unsigned long long)exp_rm);
    CHECK(p->params_removed == exp_rm / 4, "params removed = bytes/4 for F32");
    CHECK(p->n_warnings == 0, "no warnings (%u)", p->n_warnings);

    /* ── determinism ────────────────────────────────────────────────────── */
    printf("determinism\n");
    CHECK(poe_plan_write(p, "build/a.poeplan", err, sizeof err) == 0, "write 1");
    CHECK(poe_plan_write(p, "build/b.poeplan", err, sizeof err) == 0, "write 2");
    CHECK(same_file("build/a.poeplan", "build/b.poeplan"),
          "same plan -> byte-identical files");

    /* ── roundtrip ──────────────────────────────────────────────────────── */
    printf("roundtrip\n");
    poe_plan *q;
    CHECK(poe_plan_load(&q, "build/a.poeplan", err, sizeof err) == 0,
          "load: %s", err[0] ? err : "ok");
    CHECK(q->keep_per_layer == p->keep_per_layer &&
          q->n_layers == p->n_layers && q->n_experts == p->n_experts &&
          strcmp(q->model_fingerprint, p->model_fingerprint) == 0 &&
          strcmp(q->method, "reap") == 0, "header fields survive");
    CHECK(memcmp(q->keep, p->keep, (size_t)4 * 8) == 0, "keep map survives");
    CHECK(q->bytes_removed == p->bytes_removed, "accounting survives");
    poe_plan_free(q);
    poe_plan_free(p);

    /* ── clamping and warnings ──────────────────────────────────────────── */
    printf("guards\n");
    CHECK(poe_plan_build(&p, m, profs, w, 1, "reap", 0.90, 0,
                         err, sizeof err) == 0, "build reap 90%%");
    CHECK(p->keep_per_layer == POE_FIX_TOPK,
          "keep clamped to top_k (%u)", p->keep_per_layer);
    CHECK(p->n_warnings >= 2, "clamp + beyond-validated warnings (%u)",
          p->n_warnings);
    poe_plan_free(p);

    CHECK(write_profile("build/badfp.poeprofile", "poe1:ffffffffffffffff",
                        1, 16384) == 0, "mismatched profile written");
    poe_profile *bad;
    CHECK(poe_profile_load(&bad, "build/badfp.poeprofile", err, sizeof err) == 0,
          "mismatched profile loads");
    const poe_profile *badp[1] = { bad };
    CHECK(poe_plan_build(&p, m, badp, w, 1, "reap", 0.25, 0,
                         err, sizeof err) != 0,
          "fingerprint mismatch rejected: %s", err);
    CHECK(poe_plan_build(&p, m, badp, w, 1, "reap", 0.25, 1,
                         err, sizeof err) == 0, "--force overrides");
    CHECK(p->n_warnings >= 1, "forced mismatch leaves a warning");
    poe_plan_free(p);
    poe_profile_free(bad);

    /* method guards */
    CHECK(write_profile("build/noreap.poeprofile", m->fingerprint, 0, 16384) == 0,
          "profile without reap written");
    poe_profile *nr;
    CHECK(poe_profile_load(&nr, "build/noreap.poeprofile", err, sizeof err) == 0,
          "no-reap profile loads");
    const poe_profile *nrp[1] = { nr };
    CHECK(poe_plan_build(&p, m, nrp, w, 1, "reap", 0.25, 0,
                         err, sizeof err) != 0, "method reap without reap data fails");
    CHECK(poe_plan_build(&p, m, nrp, w, 1, "frequency", 0.25, 0,
                         err, sizeof err) == 0, "method frequency works");
    /* frequency ranks the same way in this profile -> same prune set */
    prune_right = 1;
    for (uint32_t l = 0; l < p->n_layers; l++) {
        const uint8_t *k = p->keep + (size_t)l * 8;
        if (k[1] || k[3]) prune_right = 0;
    }
    CHECK(prune_right, "frequency prunes {1,3} too");
    poe_plan_free(p);
    poe_profile_free(nr);

    poe_profile_free(pr);
    poe_model_close(m);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
