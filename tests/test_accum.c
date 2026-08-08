/* test_accum.c — profiler accumulators against hand-computed statistics.
 * No inference backend involved; belongs in CI.
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/profiler/accum.h"

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

int main(void) {
    poe_accum a;
    CHECK(poe_accum_init(&a, 2, 4, 2) == 0, "init 2 layers x 4 experts, top-2");

    /* ── probs: uniform distribution over 4 experts, layer 0, 1 token ──── */
    const float uni[4] = { 0.25f, 0.25f, 0.25f, 0.25f };
    poe_accum_observe_probs(&a, 0, 1, uni, 0);
    CHECK(feq(a.entropy_sum[0], 2.0), "uniform entropy = 2 bits (%f)",
          a.entropy_sum[0]);
    /* mass thresholds .80/.90/.95/.99 -> need 4/4/4/4 experts of 0.25 */
    CHECK(feq(a.mass_k_sum[0][0], 4.0) && feq(a.mass_k_sum[3][0], 4.0),
          "uniform mass-k = 4 at all thresholds");

    /* ── probs: concentrated distribution, layer 0, 1 token ────────────── */
    const float conc[4] = { 0.85f, 0.10f, 0.04f, 0.01f };
    poe_accum_observe_probs(&a, 0, 1, conc, 0);
    /* .80 -> k=1 (0.85); .90/.95 -> k=2 (0.95); .99 -> k=3 (0.99) */
    CHECK(feq(a.mass_k_sum[0][0], 4.0 + 1.0), "mass-k .80 += 1");
    CHECK(feq(a.mass_k_sum[1][0], 4.0 + 2.0), "mass-k .90 += 2");
    CHECK(feq(a.mass_k_sum[2][0], 4.0 + 2.0), "mass-k .95 += 2");
    CHECK(feq(a.mass_k_sum[3][0], 4.0 + 3.0), "mass-k .99 += 3");
    CHECK(a.tok_probs[0] == 2, "2 tokens observed for probs on layer 0");

    /* ── logits path: softmax(logits) == conc must give identical stats ── */
    float logits[4];
    for (int e = 0; e < 4; e++) logits[e] = logf(conc[e]) + 7.5f;
    double before = a.entropy_sum[1];
    poe_accum_observe_probs(&a, 1, 1, logits, 1);
    double h_conc = 0.0;
    for (int e = 0; e < 4; e++) h_conc -= conc[e] * log2(conc[e]);
    CHECK(feq(a.entropy_sum[1] - before, h_conc),
          "logits softmax entropy matches probs entropy (%f)", h_conc);

    /* ── selection: ids + gate weights, layer 0, 2 tokens, top-2 ───────── */
    const int32_t ids[4]  = { 0, 2,   2, 3 };      /* t0: {0,2}  t1: {2,3} */
    const float   wts[4]  = { 0.7f, 0.3f, 0.6f, 0.4f };
    uint64_t bad = 0;
    poe_accum_observe_selection(&a, 0, 2, ids, wts, &bad);
    CHECK(a.sel_count[0] == 1 && a.sel_count[2] == 2 && a.sel_count[3] == 1 &&
          a.sel_count[1] == 0, "selection counts 1/0/2/1");
    CHECK(feq(a.gate_sum[0], 0.7) && feq(a.gate_sum[2], 0.9) &&
          feq(a.gate_sum[3], 0.4), "gate sums 0.7/0.9/0.4");
    CHECK(a.tok_sel[0] == 2 && bad == 0, "2 tokens, no bad ids");

    /* ── corrupt ids are counted and skipped ───────────────────────────── */
    const int32_t badids[2] = { 7, -1 };
    const float   badwts[2] = { 1.0f, 1.0f };
    poe_accum_observe_selection(&a, 0, 1, badids, badwts, &bad);
    CHECK(bad == 2, "out-of-range ids counted (%llu)", (unsigned long long)bad);

    /* ── REAP accumulation: gate x norm, mean over routed tokens ───────── */
    const int32_t rids[4] = { 0, 2,   0, 1 };
    const float   rwts[4] = { 0.6f, 0.4f, 0.5f, 0.5f };
    const float   rnrm[4] = { 2.0f, 1.0f, 4.0f, 3.0f };
    poe_accum_observe_reap(&a, 1, 2, rids, rwts, rnrm, &bad);
    /* expert0: (0.6*2 + 0.5*4)/2 = 1.6 ; expert1: 0.5*3 = 1.5 ; expert2: 0.4 */
    CHECK(a.reap_count[4 + 0] == 2 && a.reap_count[4 + 1] == 1 &&
          a.reap_count[4 + 2] == 1, "reap counts 2/1/1");
    CHECK(feq(a.reap_sum[4 + 0] / 2.0, 1.6), "reap mean expert0 = 1.6");
    CHECK(feq(a.reap_sum[4 + 1], 1.5) && feq(a.reap_sum[4 + 2], 0.4),
          "reap sums expert1/2");
    CHECK(feq(a.norm_sum[4 + 0], 6.0), "activation-norm sum expert0 = 6");
    CHECK(bad == 2, "reap left bad_ids untouched");

    /* ── JSON body is well-formed enough to be embedded ────────────────── */
    FILE *f = fopen("build/accum.json", "w");
    fprintf(f, "{\"layers\":\n");
    poe_accum_write_json(&a, f, "");
    fprintf(f, "\n}\n");
    fclose(f);
    CHECK(1, "json body written");

    poe_accum_free(&a);
    CHECK(a.sel_count == NULL, "free resets");

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
