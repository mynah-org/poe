/* test_imatrix_stats.c — the imatrix read back as a per-layer ranking.
 *
 * The accumulator is fed activations whose per-layer shape is known by
 * construction: three layers of uniform columns with growing amplitude, and
 * one layer whose energy sits in a single column. Energy must then order by
 * amplitude and concentration must single out the peaky layer — the two
 * properties a bit allocation would be driven by, and the reason the second
 * statistic exists at all (a scale-free one cannot be an activation-norm
 * ramp in disguise).
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/imatrix_stats.h"
#include "poe/stats.h"
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

#define LAYERS  4
#define EXPERTS 8   /* matches the synthetic MoE fixture */
#define COLS    4
#define PEAKY   1        /* the layer whose energy sits in one column */

/* Feed one layer of every projection: 2 tokens, top-2 routing, one
 * activation row per selected expert. */
static void feed_layer(poe_imatrix *m, uint32_t layer, float amp, int peaky) {
    float act[2 * 2 * COLS];
    for (int slot = 0; slot < 4; slot++) {
        float *row = act + (size_t)slot * COLS;
        for (int j = 0; j < COLS; j++)
            row[j] = peaky ? (j == 0 ? amp * 2.0f : 0.0f) : amp;
    }
    const int32_t ids[4] = { 0, 1, 2, 0 };
    uint64_t bad = 0;
    for (int p = 0; p < POE_IMAT_NPROJ; p++)
        poe_imatrix_observe(m, layer, (poe_imat_proj)p, COLS, 2, 2, 2,
                            act, ids, &bad);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "build/test-stats.imatrix.gguf";
    const char *plain = argc > 2 ? argv[2] : "build/test-plain.gguf";

    poe_imatrix m;
    if (poe_imatrix_init(&m, LAYERS, EXPERTS) != 0) {
        printf("  FAIL: cannot init the accumulator\n");
        return 1;
    }
    for (uint32_t l = 0; l < LAYERS; l++)
        feed_layer(&m, l, (float)(l + 1), (int)l == PEAKY);

    char err[256] = {0};
    if (poe_imatrix_write_gguf(&m, path, "stats-calib.txt", 7, 512,
                               err, sizeof err) != 0) {
        printf("  FAIL: cannot write the fixture: %s\n", err);
        poe_imatrix_free(&m);
        return 1;
    }
    poe_imatrix_free(&m);

    printf("imatrix statistics\n");
    poe_imatrix_stats *s = NULL;
    CHECK(poe_imatrix_stats_load(&s, path, err, sizeof err) == 0,
          "load returns 0 (%s)", err[0] ? err : "no error");
    if (s == NULL) return 1;

    CHECK(s->n_layers == LAYERS && s->n_experts == EXPERTS,
          "shape discovered: %u layers x %u experts", s->n_layers, s->n_experts);
    CHECK(s->cols[0] == COLS && s->cols[1] == COLS && s->cols[2] == COLS,
          "column count per projection is %u", s->cols[0]);
    CHECK(s->n_entries == LAYERS * POE_IMSTAT_NPROJ,
          "%u in_sum2 entries read", s->n_entries);
    CHECK(s->chunk_count == 7 && s->chunk_size == 512,
          "chunk provenance round-trips (%u x %u)", s->chunk_count, s->chunk_size);
    CHECK(strcmp(s->dataset, "stats-calib.txt") == 0,
          "dataset name round-trips (%s)", s->dataset);

    /* Energy is the mean squared activation per element, so a layer fed
     * amplitude a has energy a^2 — for the uniform layers, 1, 9 and 16. */
    CHECK(fabs(s->energy[0] - 1.0) < 1e-4, "layer 0 energy %.4f = 1", s->energy[0]);
    CHECK(fabs(s->energy[2] - 9.0) < 1e-4, "layer 2 energy %.4f = 9", s->energy[2]);
    CHECK(fabs(s->energy[3] - 16.0) < 1e-4, "layer 3 energy %.4f = 16", s->energy[3]);
    CHECK(s->energy[0] < s->energy[2] && s->energy[2] < s->energy[3],
          "energy orders by amplitude");

    /* Concentration is 1 - participation ratio: 0 when every column carries
     * the same energy, 1 - 1/COLS when one column carries all of it. */
    CHECK(s->concentration[0] < 1e-9, "uniform layer has concentration ~0 (%.6f)",
          s->concentration[0]);
    CHECK(fabs(s->concentration[PEAKY] - (1.0 - 1.0 / COLS)) < 1e-4,
          "peaky layer concentration %.4f = %.4f", s->concentration[PEAKY],
          1.0 - 1.0 / COLS);
    CHECK(s->concentration[PEAKY] > s->concentration[3],
          "concentration ranks the peaky layer above the loudest one");

    /* Routing: 2 tokens x top-2 per projection, ids {0,1,2,0}, so expert 0
     * takes two slots, experts 1 and 2 one each, expert 3 none. */
    CHECK(s->counts[0] == 2 && s->counts[1] == 1 && s->counts[2] == 1 &&
          s->counts[3] == 0, "per-expert counts survive the round trip");
    CHECK(s->experts_unseen == LAYERS * (EXPERTS - 3),
          "%u experts never routed (%u per layer)", s->experts_unseen,
          EXPERTS - 3);
    CHECK(s->slots_total == (uint64_t)LAYERS * 4,
          "%llu routed slots", (unsigned long long)s->slots_total);

    /* The depth check the allocator warns on — and the reason there are two
     * statistics. Amplitude grows with the layer index here, so energy is a
     * perfect depth ramp even though layer PEAKY has a completely different
     * column structure: exactly the confound that made per-layer REAP look
     * informative. Concentration sees that structure and is not a ramp. */
    CHECK(fabs(poe_depth_rho(s->energy, s->n_layers) - 1.0) < 1e-9,
          "energy is a perfect depth ramp (%.3f) — the warned-about shape",
          poe_depth_rho(s->energy, s->n_layers));
    CHECK(fabs(poe_depth_rho(s->concentration, s->n_layers)) < 0.9,
          "concentration is not a depth ramp (%.3f)",
          poe_depth_rho(s->concentration, s->n_layers));
    const double ramp[4] = { 0.1, 0.4, 0.7, 1.0 };
    const double flat[4] = { 1.0, 1.0, 1.0, 1.0 };
    CHECK(fabs(poe_depth_rho(ramp, 4) - 1.0) < 1e-9,
          "a monotone ramp scores depth rho 1.0");
    CHECK(poe_depth_rho(flat, 4) == 0.0,
          "a constant vector has no ranking (rho 0)");
    CHECK(fabs(poe_spearman(ramp, ramp, 4) - 1.0) < 1e-9, "spearman(x,x) = 1");
    {
        double rev[4];
        for (int i = 0; i < 4; i++) rev[i] = ramp[3 - i];
        CHECK(fabs(poe_spearman(ramp, rev, 4) + 1.0) < 1e-9,
              "spearman(x, reverse x) = -1");
    }

    poe_imatrix_stats_free(s);

    /* A GGUF that is not an imatrix must be refused by name, not read as
     * one: silently returning empty statistics would produce a plan labelled
     * with a ranking it never had. */
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    const uint64_t ne[2] = { 2, 2 };
    const float payload[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    if (w != NULL &&
        ingot_gguf_kv_string(w, "general.architecture", "llama") == 0 &&
        ingot_gguf_add_tensor(w, "blk.0.ffn_gate_exps.weight.in_sum2",
                              INGOT_TYPE_F32, 2, ne, payload) == 0 &&
        ingot_gguf_writer_save(w, plain, err, sizeof err) == 0) {
        poe_imatrix_stats *bad = NULL;
        err[0] = '\0';
        CHECK(poe_imatrix_stats_load(&bad, plain, err, sizeof err) != 0 &&
              bad == NULL, "a GGUF without general.type=imatrix is refused (%s)",
              err);
    } else {
        printf("  FAIL: cannot write the negative fixture\n");
        failures++;
    }
    if (w) ingot_gguf_writer_free(w);

    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
