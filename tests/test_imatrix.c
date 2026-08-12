/* test_imatrix.c — per-expert activation statistics against hand-computed
 * sums, and the GGUF artifact read back with ingot. No inference backend
 * involved; belongs in CI.
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/profiler/imatrix.h"
#include "ingot.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

static int feq(double a, double b) { return fabs(a - b) < 1e-6; }

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "build/test.imatrix.gguf";
    poe_imatrix m;

    CHECK(poe_imatrix_init(&m, 2, 3) == 0, "init 2 layers x 3 experts");
    CHECK(!poe_imatrix_has_data(&m), "a fresh accumulator has no data");

    /* ── one row per selected expert (act_rows == n_used) ───────────────
     * 2 tokens, top-2 of 3 experts, 4 columns. Token 0 picks experts 0,2;
     * token 1 picks 1,0. Activation rows are chosen so every expert's sum
     * of squares is a number that can be read off by hand. */
    const float act[2 * 2 * 4] = {
        1.0f, 2.0f, 0.0f, 0.0f,    /* t0 slot0 -> expert 0 */
        3.0f, 0.0f, 0.0f, 0.0f,    /* t0 slot1 -> expert 2 */
        0.0f, 0.0f, 4.0f, 0.0f,    /* t1 slot0 -> expert 1 */
        1.0f, 1.0f, 0.0f, 0.0f,    /* t1 slot1 -> expert 0 */
    };
    const int32_t ids[2 * 2] = { 0, 2, 1, 0 };
    uint64_t bad = 0;

    CHECK(poe_imatrix_observe(&m, 0, POE_IMAT_GATE, 4, 2, 2, 2,
                              act, ids, &bad) == 0, "observe returns 0");
    CHECK(bad == 0, "no out-of-range ids");
    CHECK(poe_imatrix_has_data(&m), "the accumulator now has data");
    CHECK(m.cols[POE_IMAT_GATE] == 4, "column count recorded");

    const double *g = m.sum2[POE_IMAT_GATE];
    /* expert 0 saw (1,2,0,0) and (1,1,0,0) -> squares add per column */
    CHECK(feq(g[0], 1.0 + 1.0) && feq(g[1], 4.0 + 1.0) && feq(g[2], 0.0),
          "expert 0 accumulates both of its rows (%.1f, %.1f)", g[0], g[1]);
    CHECK(feq(g[4 + 2], 16.0), "expert 1 sums the column it actually used");
    CHECK(feq(g[8 + 0], 9.0), "expert 2 sums its single row");
    CHECK(m.counts[POE_IMAT_GATE][0] == 2 && m.counts[POE_IMAT_GATE][1] == 1 &&
          m.counts[POE_IMAT_GATE][2] == 1, "counts follow the routing");

    /* Layer 1 stays untouched: the layer index must not alias. */
    CHECK(feq(g[3 * 4 + 0], 0.0) && m.counts[POE_IMAT_GATE][3] == 0,
          "layer 1 is untouched");

    /* ── one row broadcast to every selected expert (act_rows == 1) ──── */
    const float shared[4] = { 0.0f, 0.0f, 0.0f, 5.0f };
    const int32_t both[2] = { 1, 2 };
    CHECK(poe_imatrix_observe(&m, 1, POE_IMAT_DOWN, 4, 1, 2, 1,
                              shared, both, &bad) == 0, "broadcast observe");
    const double *d = m.sum2[POE_IMAT_DOWN];
    CHECK(feq(d[(3 + 1) * 4 + 3], 25.0) && feq(d[(3 + 2) * 4 + 3], 25.0),
          "a broadcast row lands on every expert that selected it");

    /* ── the guards ─────────────────────────────────────────────────── */
    const int32_t rogue[2] = { 9, -1 };
    CHECK(poe_imatrix_observe(&m, 1, POE_IMAT_DOWN, 4, 1, 2, 1,
                              shared, rogue, &bad) == 0, "rogue ids survive");
    CHECK(bad == 2, "both out-of-range ids were counted, not written");
    CHECK(poe_imatrix_observe(&m, 0, POE_IMAT_GATE, 8, 2, 2, 2,
                              act, ids, &bad) != 0,
          "a changed column count is refused, not averaged");
    CHECK(poe_imatrix_observe(&m, 99, POE_IMAT_GATE, 4, 2, 2, 2,
                              act, ids, &bad) != 0, "a layer past the end is refused");

    /* ── the artifact: write it, then read it back with ingot ───────── */
    char err[256] = { 0 };
    CHECK(poe_imatrix_write_gguf(&m, out, "test-calib.txt", 7, 512,
                                 err, sizeof err) == 0,
          "write_gguf returns 0 (%s)", err[0] ? err : "no error");

    ingot_gguf *g2 = NULL;
    CHECK(ingot_gguf_open(&g2, out, err, sizeof err) == 0,
          "ingot reopens the artifact (%s)", err[0] ? err : "no error");
    if (g2 != NULL) {
        const ingot_kv *type = ingot_gguf_kv_find(g2, "general.type");
        const char *sval = NULL;
        CHECK(type != NULL && ingot_kv_str(type, &sval) == 0 &&
              strcmp(sval, "imatrix") == 0, "general.type is imatrix");

        const ingot_kv *chunks = ingot_gguf_kv_find(g2, "imatrix.chunk_count");
        CHECK(chunks != NULL, "imatrix.chunk_count is present");
        CHECK(ingot_gguf_kv_find(g2, "imatrix.chunk_size") != NULL,
              "imatrix.chunk_size is present");
        CHECK(ingot_gguf_kv_find(g2, "imatrix.datasets") != NULL,
              "imatrix.datasets is present");

        /* Two projections had data, two tensors each per layer. */
        CHECK(ingot_gguf_count(g2) == 2 * 2 * 2,
              "8 tensors: in_sum2 + counts, 2 layers, 2 projections (%zu)",
              ingot_gguf_count(g2));

        const ingot_tensor *t =
            ingot_gguf_find(g2, "blk.0.ffn_gate_exps.weight.in_sum2");
        CHECK(t != NULL, "the gate in_sum2 tensor is named as llama.cpp expects");
        if (t != NULL) {
            uint64_t shape[4] = { 0 };
            ingot_gguf_shape_row_major(t, shape);
            const float *v = ingot_gguf_data(g2, t);
            CHECK(v != NULL && feq(v[1], 5.0),
                  "the payload survives the round trip (%.1f)", v ? v[1] : -1.0);
            CHECK(t->ne[0] == 4 && t->ne[1] == 3,
                  "in_sum2 is [cols, experts] = [4, 3]");
        }

        const ingot_tensor *c =
            ingot_gguf_find(g2, "blk.0.ffn_gate_exps.weight.counts");
        CHECK(c != NULL && c->ne[0] == 1 && c->ne[1] == 3,
              "counts is [1, experts]");
        if (c != NULL) {
            const float *v = ingot_gguf_data(g2, c);
            CHECK(v != NULL && feq(v[0], 2.0) && feq(v[1], 1.0),
                  "counts carry the routing totals");
        }
        ingot_gguf_close(g2);
    }

    /* Determinism: the same accumulator must produce the same bytes. */
    char again[512];
    snprintf(again, sizeof again, "%s.2", out);
    if (poe_imatrix_write_gguf(&m, again, "test-calib.txt", 7, 512,
                               err, sizeof err) == 0) {
        FILE *fa = fopen(out, "rb"), *fb = fopen(again, "rb");
        int same = fa != NULL && fb != NULL;
        while (same) {
            const int x = fgetc(fa), y = fgetc(fb);
            if (x != y) same = 0;
            if (x == EOF || y == EOF) break;
        }
        CHECK(same, "two writes of the same accumulator are byte-identical");
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        remove(again);
    }

    /* ── the dense path ─────────────────────────────────────────────────
     * An expert-only imatrix leaves a whole-model quantize on unweighted
     * fits everywhere else, so plain matmuls accumulate too — keyed by the
     * weight's own name, one row, counts[0] = tokens, which is the shape
     * llama.cpp writes for a non-MoE tensor. */
    const float dense[2 * 3] = {
        1.0f, 2.0f, 3.0f,      /* token 0 */
        0.0f, 4.0f, 0.0f,      /* token 1 */
    };
    CHECK(poe_imatrix_observe_plain(&m, "blk.0.attn_q.weight", 3, 2, dense) == 0,
          "a plain matmul folds in");
    CHECK(poe_imatrix_observe_plain(&m, "blk.0.attn_q.weight", 3, 1, dense) == 0,
          "the same weight accumulates rather than duplicating");
    CHECK(m.n_plain == 1, "one dense entry so far (%zu)", m.n_plain);
    CHECK(poe_imatrix_observe_plain(&m, "blk.0.attn_q.weight", 4, 1, dense) != 0,
          "a column-count change is refused, not averaged");
    CHECK(feq(m.plain[0].sum2[1], 2.0 * 2.0 + 4.0 * 4.0 + 2.0 * 2.0),
          "squares accumulate per column (%.1f)", m.plain[0].sum2[1]);
    CHECK(m.plain[0].count == 3, "every row counts (%llu)",
          (unsigned long long)m.plain[0].count);

    char withdense[512];
    snprintf(withdense, sizeof withdense, "%s.dense", out);
    if (poe_imatrix_write_gguf(&m, withdense, "test-calib.txt", 7, 512,
                               err, sizeof err) == 0) {
        ingot_gguf *g3 = NULL;
        if (ingot_gguf_open(&g3, withdense, err, sizeof err) == 0) {
            CHECK(ingot_gguf_count(g3) == 2 * 2 * 2 + 2,
                  "the dense entry joins the expert ones (%zu tensors)",
                  ingot_gguf_count(g3));
            const ingot_tensor *dsum =
                ingot_gguf_find(g3, "blk.0.attn_q.weight.in_sum2");
            CHECK(dsum != NULL && dsum->ne[0] == 3 && dsum->ne[1] == 1,
                  "dense in_sum2 is [cols, 1]");
            const ingot_tensor *dc =
                ingot_gguf_find(g3, "blk.0.attn_q.weight.counts");
            const float *cv = dc ? ingot_gguf_data(g3, dc) : NULL;
            CHECK(cv != NULL && feq(cv[0], 3.0),
                  "dense counts carry the token total");
            ingot_gguf_close(g3);
        }
        remove(withdense);
    } else {
        printf("  FAIL: cannot write the artifact with dense entries: %s\n", err);
        failures++;
    }

    poe_imatrix_free(&m);
    CHECK(m.sum2[POE_IMAT_GATE] == NULL, "free clears the accumulator");
    CHECK(m.plain == NULL && m.n_plain == 0, "and clears the dense entries");

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
