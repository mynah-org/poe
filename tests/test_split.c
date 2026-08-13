/* test_split.c — hot and cold experts as two tensors, and the permutation
 * that makes `id < hot_count` a valid test at runtime.
 *
 * The properties a patched runtime will depend on are the ones checked here:
 * the two tensors exist with the right shapes and types, expert i of the
 * output really is expert order[i] of the source, and the router's rows moved
 * with them — because if the permutation and the router ever disagree, every
 * token is routed to the wrong expert and nothing else in the model can tell.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/split.h"
#include "fixture.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* Which source expert a rewritten one is closest to. Quantization moves the
 * values a little; it does not move them onto a different expert. */
static int nearest_source_expert(const float *out_expert, const float *src_all,
                                 uint64_t per_expert, uint32_t E) {
    int best = -1;
    double best_d = 0;
    for (uint32_t e = 0; e < E; e++) {
        double d = 0;
        for (uint64_t i = 0; i < per_expert; i++) {
            const double x = (double)out_expert[i] -
                             (double)src_all[e * per_expert + i];
            d += x * x;
        }
        if (best < 0 || d < best_d) { best = (int)e; best_d = d; }
    }
    return best;
}

int main(void) {
    const char *src_path = "build/fixture-wide.gguf";
    const char *out_path = "build/split.gguf";
    const char *inv_path = "build/split-inv.gguf";
    char err[256] = { 0 };

    CHECK(poe_fixture_moe_wide(src_path, 5, err, sizeof err) == 0,
          "wide fixture written (%s)", err[0] ? err : "ok");
    poe_model *src = NULL;
    CHECK(poe_model_open(&src, src_path, err, sizeof err) == 0,
          "open the fixture (%s)", err[0] ? err : "ok");
    if (src == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                       return 1; }
    const uint32_t E = src->expert_count, L = src->n_blocks;

    /* saliency e+1: expert E-1 is the hottest, expert 0 the coldest */
    poe_profile pr;
    memset(&pr, 0, sizeof pr);
    pr.n_layers = L;
    pr.n_experts = E;
    pr.tokens = 8192;
    snprintf(pr.fingerprint, sizeof pr.fingerprint, "%s", src->fingerprint);
    pr.reap_mean = calloc((size_t)L * E, sizeof *pr.reap_mean);
    if (pr.reap_mean == NULL) return 1;
    for (uint32_t l = 0; l < L; l++)
        for (uint32_t e = 0; e < E; e++)
            pr.reap_mean[(size_t)l * E + e] = (double)(e + 1);

    poe_split_opts o;
    memset(&o, 0, sizeof o);
    o.profile = &pr;
    o.hot_fraction = 0.5;
    o.hot_type = INGOT_TYPE_Q6_K;
    o.cold_type = INGOT_TYPE_Q2_K;

    poe_split_stats st;
    CHECK(poe_split(src, &o, out_path, &st, err, sizeof err) == 0,
          "split written (%s)", err[0] ? err : "ok");
    CHECK(st.hot_per_layer == E / 2 && st.cold_per_layer == E - E / 2,
          "%u hot and %u cold experts per layer", st.hot_per_layer,
          st.cold_per_layer);
    CHECK(st.slabs_split == L * 3, "%u slabs split", st.slabs_split);
    CHECK(st.routers_permuted >= L, "%u router tensors permuted",
          st.routers_permuted);
    CHECK(fabs(st.mean_bits - 0.5 * (6.5625 + 2.625)) < 1e-6,
          "mean %.4f bits/weight over the experts", st.mean_bits);

    /* ── the artifact ──────────────────────────────────────────────────── */
    ingot_gguf *g = NULL;
    CHECK(ingot_gguf_open(&g, out_path, err, sizeof err) == 0,
          "the split checkpoint reopens (%s)", err[0] ? err : "ok");
    if (g == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                     return 1; }

    const ingot_kv *hc = ingot_gguf_kv_find(g, "poe.split.hot_count");
    uint64_t hot_count = 0;
    CHECK(hc != NULL && ingot_kv_u64(hc, &hot_count) == 0 &&
          hot_count == E / 2,
          "poe.split.hot_count is an integer a runtime can read (%llu)",
          (unsigned long long)hot_count);

    const ingot_tensor *hot = ingot_gguf_find(g, "blk.0.ffn_gate_exps.weight");
    const ingot_tensor *cold =
        ingot_gguf_find(g, "blk.0.ffn_gate_exps_cold.weight");
    CHECK(hot != NULL && cold != NULL, "both halves of the slab are present");
    if (hot == NULL || cold == NULL) { ingot_gguf_close(g); return 1; }
    CHECK(hot->type == INGOT_TYPE_Q6_K && cold->type == INGOT_TYPE_Q2_K,
          "each half carries its own type (%s / %s)",
          ingot_type_name(hot->type), ingot_type_name(cold->type));
    CHECK(hot->ne[2] == E / 2 && cold->ne[2] == E - E / 2,
          "the expert dimension splits %llu / %llu",
          (unsigned long long)hot->ne[2], (unsigned long long)cold->ne[2]);
    CHECK(hot->ne[0] == src->blocks[0].gate_exps_w->ne[0] &&
          hot->ne[1] == src->blocks[0].gate_exps_w->ne[1],
          "the other dimensions are untouched");

    /* ── the permutation ───────────────────────────────────────────────── */
    const ingot_tensor *s_slab = src->blocks[0].gate_exps_w;
    const uint64_t per_expert = s_slab->nelem / E;
    float *s_all = malloc((size_t)s_slab->nelem * sizeof *s_all);
    float *h_all = malloc((size_t)hot->nelem * sizeof *h_all);
    float *c_all = malloc((size_t)cold->nelem * sizeof *c_all);
    if (s_all && h_all && c_all &&
        ingot_gguf_dequant(src->g, s_slab, s_all) == 0 &&
        ingot_gguf_dequant(g, hot, h_all) == 0 &&
        ingot_gguf_dequant(g, cold, c_all) == 0) {
        /* saliency e+1 means the order is E-1, E-2, ... so hot expert i is
         * source expert E-1-i and cold expert j is source expert E/2-1-j */
        int hot_ok = 1, cold_ok = 1;
        for (uint32_t i = 0; i < E / 2; i++)
            if (nearest_source_expert(h_all + (uint64_t)i * per_expert,
                                      s_all, per_expert, E) != (int)(E - 1 - i))
                hot_ok = 0;
        for (uint32_t j = 0; j < E - E / 2; j++)
            if (nearest_source_expert(c_all + (uint64_t)j * per_expert,
                                      s_all, per_expert, E) !=
                (int)(E - 1 - (E / 2 + j)))
                cold_ok = 0;
        CHECK(hot_ok, "hot expert i is source expert order[i]");
        CHECK(cold_ok, "cold expert j continues the same order");
    } else {
        printf("  FAIL: cannot dequantize the halves for comparison\n");
        failures++;
    }

    /* the router must have moved with them */
    const ingot_tensor *s_router = src->blocks[0].router_w;
    const ingot_tensor *o_router = ingot_gguf_find(g, s_router->name);
    CHECK(o_router != NULL && o_router->nbytes == s_router->nbytes,
          "the router keeps its shape");
    if (o_router != NULL) {
        const uint8_t *a = ingot_gguf_data(src->g, s_router);
        const uint8_t *b = ingot_gguf_data(g, o_router);
        const uint64_t unit = s_router->nbytes / E;
        int permuted = 1;
        for (uint32_t i = 0; i < E; i++)
            if (memcmp(b + (uint64_t)i * unit,
                       a + (uint64_t)(E - 1 - i) * unit, (size_t)unit) != 0)
                permuted = 0;
        CHECK(permuted, "router row i is the source row for expert order[i]");
    }
    ingot_gguf_close(g);
    free(s_all); free(h_all); free(c_all);

    /* ── the control ───────────────────────────────────────────────────── */
    poe_split_opts iv = o;
    iv.invert = 1;
    poe_split_stats si;
    CHECK(poe_split(src, &iv, inv_path, &si, err, sizeof err) == 0,
          "inverted split written (%s)", err[0] ? err : "ok");
    CHECK(si.bytes_written == st.bytes_written,
          "the control is the same size (%llu B)",
          (unsigned long long)si.bytes_written);
    if (ingot_gguf_open(&g, inv_path, err, sizeof err) == 0) {
        const ingot_tensor *ir = ingot_gguf_find(g, s_router->name);
        const uint8_t *a = ingot_gguf_data(src->g, s_router);
        const uint8_t *b = ir ? ingot_gguf_data(g, ir) : NULL;
        const uint64_t unit = s_router->nbytes / E;
        int reversed = b != NULL;
        for (uint32_t i = 0; i < E && reversed; i++)
            if (memcmp(b + (uint64_t)i * unit, a + (uint64_t)i * unit,
                       (size_t)unit) != 0)
                reversed = 0;
        CHECK(reversed, "inverting puts the least needed experts first");
        ingot_gguf_close(g);
    }

    /* ── guards ────────────────────────────────────────────────────────── */
    poe_split_opts bad = o;
    bad.profile = NULL;
    poe_split_stats sb;
    err[0] = '\0';
    CHECK(poe_split(src, &bad, "build/split-bad.gguf", &sb, err, sizeof err) != 0,
          "splitting without a profile is refused (%s)", err);
    bad = o;
    bad.hot_fraction = 1.0;
    err[0] = '\0';
    CHECK(poe_split(src, &bad, "build/split-bad.gguf", &sb, err, sizeof err) != 0,
          "a hot fraction leaving no cold set is refused (%s)", err);

    poe_model *narrow = NULL;
    if (poe_model_open(&narrow, "build/fixture-moe.gguf", err, sizeof err) == 0) {
        /* A profile's arrays are n_layers x n_experts by contract, so the
         * dimensions cannot be rewritten without resizing them: the shapes
         * here are the *other* fixture's. Copying the struct and only
         * relabelling it made poe_rank_experts read past the end of
         * reap_mean — silently, because this call has to fail for an
         * unrelated reason anyway. ASan is what noticed. */
        poe_profile npr = pr;
        npr.n_experts = narrow->expert_count;
        npr.n_layers = narrow->n_blocks;
        npr.reap_mean = calloc((size_t)npr.n_layers * npr.n_experts,
                               sizeof *npr.reap_mean);
        if (npr.reap_mean == NULL) { poe_model_close(narrow); return 1; }
        for (uint32_t l = 0; l < npr.n_layers; l++)
            for (uint32_t e = 0; e < npr.n_experts; e++)
                npr.reap_mean[(size_t)l * npr.n_experts + e] = (double)(e + 1);
        snprintf(npr.fingerprint, sizeof npr.fingerprint, "%s",
                 narrow->fingerprint);
        poe_split_opts no = o;
        no.profile = &npr;
        err[0] = '\0';
        CHECK(poe_split(narrow, &no, "build/split-bad.gguf", &sb,
                        err, sizeof err) != 0,
              "rows too narrow for the types are refused (%s)", err);
        free(npr.reap_mean);
        poe_model_close(narrow);
    }

    free(pr.reap_mean);
    poe_model_close(src);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
