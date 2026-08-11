/* test_quantplan.c — mixed-precision allocation: exact accounting, the
 * budget is respected, and saliency actually moves bits.
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/quantplan.h"
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

/* Sum of the bytes the plan claims, recomputed from the type ladder — a
 * second opinion on the accounting rather than the plan's own total. */
static uint64_t recount(const poe_quantplan *p) {
    uint64_t n = 0;
    for (size_t i = 0; i < (size_t)p->n_layers * POE_QSLAB_NPROJ; i++) {
        if (p->type[i] < 0) continue;
        uint64_t b = 0;
        ingot_type_nbytes(p->type[i], p->nelem[i], &b);
        n += b;
    }
    return n;
}

static int count_type(const poe_quantplan *p, int type) {
    int n = 0;
    for (size_t i = 0; i < (size_t)p->n_layers * POE_QSLAB_NPROJ; i++)
        if (p->type[i] == type) n++;
    return n;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "build/fixture-moe.gguf";
    (void)argc;
    char err[256] = { 0 };

    poe_model *m = NULL;
    CHECK(poe_model_open(&m, model, err, sizeof err) == 0,
          "open %s (%s)", model, err[0] ? err : "ok");
    if (m == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                     return 1; }

    uint64_t slabs = 0;
    for (uint32_t l = 0; l < m->n_blocks; l++) {
        if (m->blocks[l].gate_exps_w) slabs += m->blocks[l].gate_exps_w->nbytes;
        if (m->blocks[l].up_exps_w)   slabs += m->blocks[l].up_exps_w->nbytes;
        if (m->blocks[l].down_exps_w) slabs += m->blocks[l].down_exps_w->nbytes;
    }
    CHECK(slabs > 0, "the fixture has packed expert slabs (%llu B)",
          (unsigned long long)slabs);

    /* ── a budget the ladder cannot spend: everything tops out ────────── */
    poe_quantplan *p = NULL;
    CHECK(poe_quantplan_build(&p, m, NULL, slabs, NULL, 0, 0,
                              err, sizeof err) == 0,
          "build at the source size (%s)", err[0] ? err : "ok");
    if (p != NULL) {
        CHECK(p->bytes_after_total == recount(p),
              "the total equals the sum of its slabs (%llu)",
              (unsigned long long)p->bytes_after_total);
        CHECK(p->bytes_after_total <= slabs, "the budget is respected");
        CHECK(p->n_warnings > 0 && strstr(p->warnings[0], "top of the") != NULL,
              "an unspendable budget is reported, not hidden");
        CHECK(strcmp(p->method, "uniform") == 0, "no profile means uniform");
        poe_quantplan_free(p);
    }

    /* ── a tight budget: the floor type, and never over ───────────────── */
    const int *ladder;
    const size_t nl = poe_quantplan_ladder(&ladder, NULL);
    p = NULL;
    CHECK(poe_quantplan_build(&p, m, NULL, 1, NULL, 0, 0, err, sizeof err) == 0,
          "build with an impossible budget still returns a plan");
    if (p != NULL) {
        CHECK(count_type(p, ladder[0]) ==
              (int)((size_t)p->n_layers * POE_QSLAB_NPROJ),
              "every slab sits on the cheapest rung");
        CHECK(p->n_warnings > 0 && strstr(p->warnings[0], "floor") != NULL,
              "the floor is reported as a warning");
        CHECK(p->bytes_after_total == recount(p), "accounting still exact");
        poe_quantplan_free(p);
    }

    /* ── the ladder can be restricted ─────────────────────────────────── */
    const int only[2] = { ladder[0], ladder[1] };
    p = NULL;
    CHECK(poe_quantplan_build(&p, m, NULL, slabs, only, 2, 0,
                              err, sizeof err) == 0, "build with 2 candidates");
    if (p != NULL) {
        int outside = 0;
        for (size_t i = 0; i < (size_t)p->n_layers * POE_QSLAB_NPROJ; i++)
            if (p->type[i] >= 0 && p->type[i] != only[0] && p->type[i] != only[1])
                outside++;
        CHECK(outside == 0, "no slab escapes the restricted ladder");
        poe_quantplan_free(p);
    }

    /* ── saliency moves bits ──────────────────────────────────────────
     * A hand-built profile, not the fixture's: the point is to force the
     * allocator to CHOOSE, so layer 0 is made overwhelmingly salient and
     * the last layer nearly worthless, and the budget is set just above the
     * floor so there is only enough for a few upgrades. With a real profile
     * on a 4-layer fixture the allocation saturates and the comparison
     * passes without ever exercising the ranking. */
    uint64_t floor_b = 0, ceil_b = 0;
    for (uint32_t l = 0; l < m->n_blocks; l++) {
        for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
            const ingot_tensor *t = pr == 0 ? m->blocks[l].gate_exps_w
                                  : pr == 1 ? m->blocks[l].up_exps_w
                                            : m->blocks[l].down_exps_w;
            if (t == NULL) continue;
            uint64_t a = 0, b = 0;
            ingot_type_nbytes(ladder[0], t->nelem, &a);
            ingot_type_nbytes(ladder[nl - 1], t->nelem, &b);
            floor_b += a; ceil_b += b;
        }
    }

    poe_profile skew;
    memset(&skew, 0, sizeof skew);
    snprintf(skew.fingerprint, sizeof skew.fingerprint, "%s", m->fingerprint);
    skew.n_layers = m->n_blocks;
    skew.n_experts = m->expert_count;
    skew.top_k = m->experts_per_token;
    const size_t le = (size_t)skew.n_layers * skew.n_experts;
    skew.reap_mean = calloc(le, sizeof *skew.reap_mean);
    skew.sel_count = calloc(le, sizeof *skew.sel_count);
    if (skew.reap_mean != NULL && skew.sel_count != NULL) {
        for (uint32_t l = 0; l < skew.n_layers; l++)
            for (uint32_t e = 0; e < skew.n_experts; e++)
                skew.reap_mean[(size_t)l * skew.n_experts + e] =
                    l == 0 ? 1000.0 : 1.0;

        p = NULL;
        const uint64_t tight = floor_b + (ceil_b - floor_b) / 8;
        CHECK(poe_quantplan_build(&p, m, &skew, tight, NULL, 0, 0,
                                  err, sizeof err) == 0,
              "build with a skewed profile at a tight budget (%s)",
              err[0] ? err : "ok");
        if (p != NULL) {
            CHECK(strcmp(p->method, "saliency") == 0, "method is saliency");
            CHECK(p->bytes_after_total <= tight, "the tight budget is respected");
            CHECK(p->bytes_after_total == recount(p), "accounting exact");
            CHECK(p->layer_score[0] == 1.0 &&
                  p->layer_score[p->n_layers - 1] == 0.0,
                  "the skew survives normalization (%.2f vs %.2f)",
                  p->layer_score[0], p->layer_score[p->n_layers - 1]);

            uint64_t hib = 0, lob = 0;
            for (int pr = 0; pr < POE_QSLAB_NPROJ; pr++) {
                hib += p->bytes_after[pr];
                lob += p->bytes_after[(size_t)(p->n_layers - 1) *
                                      POE_QSLAB_NPROJ + pr];
            }
            CHECK(hib > lob, "the salient layer is given strictly more bits "
                  "than the starved one (%llu vs %llu B)",
                  (unsigned long long)hib, (unsigned long long)lob);
            poe_quantplan_free(p);
        }
    }
    free(skew.reap_mean);
    free(skew.sel_count);

    /* ── the artifacts ────────────────────────────────────────────────── */
    p = NULL;
    if (poe_quantplan_build(&p, m, NULL, slabs / 2, NULL, 0, 0,
                            err, sizeof err) == 0) {
        CHECK(poe_quantplan_write(p, "build/test.poequant", err, sizeof err) == 0,
              "write .poequant (%s)", err[0] ? err : "ok");
        CHECK(poe_quantplan_write_tensor_types(p, "build/test.tensortypes",
                                               err, sizeof err) == 0,
              "write the tensor-type file (%s)", err[0] ? err : "ok");

        /* The regex must be anchored: an unescaped blk.1. also matches
         * blk.11, which would silently re-type layers nobody chose. */
        FILE *f = fopen("build/test.tensortypes", "r");
        char line[256];
        int anchored = 1, lines = 0;
        while (f != NULL && fgets(line, sizeof line, f)) {
            lines++;
            if (line[0] != '^' || strstr(line, "$=") == NULL) anchored = 0;
            if (strstr(line, "blk\\.") == NULL) anchored = 0;
        }
        if (f) fclose(f);
        CHECK(lines == (int)((size_t)p->n_layers * POE_QSLAB_NPROJ),
              "one line per slab (%d)", lines);
        CHECK(anchored, "every pattern is anchored and has escaped dots");
        poe_quantplan_free(p);
    }

    poe_model_close(m);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
