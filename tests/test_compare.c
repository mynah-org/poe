/* test_compare.c — .poeprofile parsing and comparison metrics against
 * hand-computed values. Two tiny synthetic profiles (2 layers × 4 experts,
 * top-2) chosen so every metric has a closed-form expected value:
 *
 *   layer 0: identical selections            -> Jaccard 1, Spearman 1, JSD 0
 *   layer 1: exactly disjoint selections     -> Jaccard 0, Spearman -1, JSD 1 bit
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/json.h"
#include "poe/profile.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

static int feq(double a, double b) { return fabs(a - b) < 1e-9; }

static const char *PROFILE_FMT =
"{\n"
"  \"poeprofile\": 0,\n"
"  \"poe_version\": \"test\",\n"
"  \"model_fingerprint\": \"poe1:00000000deadbeef\",\n"
"  \"arch\": \"qwen3moe\",\n"
"  \"n_layers\": 2, \"n_experts\": 4, \"top_k\": 2,\n"
"  \"dataset_hash\": \"fnv1a:0000000000000000\",\n"
"  \"tokens\": 16,\n"
"  \"layers\":\n"
"  [\n"
"    {\"layer\": 0, \"tokens\": 16,\n"
"     \"entropy_bits_mean\": 1.500000,\n"
"     \"mass_k_mean\": {\"0.80\": 2.0, \"0.90\": 3.0, \"0.95\": 3.5, \"0.99\": 4.0},\n"
"     \"mass_k_hist\": {\"0.80\": [8,8,0,0], \"0.90\": [0,16,0,0],\n"
"                       \"0.95\": [0,8,8,0], \"0.99\": [0,0,0,16]},\n"
"     \"sel_count\": [10,0,5,1],\n"
"     \"gate_mean\": [0.200000,0.000000,0.150000,0.050000]},\n"
"    {\"layer\": 1, \"tokens\": 16,\n"
"     \"entropy_bits_mean\": 1.000000,\n"
"     \"mass_k_mean\": {\"0.80\": 2.0, \"0.90\": 2.0, \"0.95\": 2.0, \"0.99\": 2.0},\n"
"     \"sel_count\": [%s],\n"
"     \"gate_mean\": [0.5,0.5,0.0,0.0],\n"
"     \"reap_mean\": [%s],\n"
"     \"actnorm_mean\": [1.0,1.0,1.0,1.0]}\n"
"  ]\n"
"}\n";

static int write_profile(const char *path, const char *l1_sel,
                         const char *l1_reap) {
    FILE *f = fopen(path, "w");
    if (f == NULL) return -1;
    fprintf(f, PROFILE_FMT, l1_sel, l1_reap);
    fclose(f);
    return 0;
}

int main(void) {
    char err[256];

    /* ── json parser basics ─────────────────────────────────────────────── */
    printf("json parser\n");
    const char *doc =
        "{\"a\": [1, 2.5, -3], \"s\": \"x\\n\\\"y\\u0041\", \"b\": true, "
        "\"n\": null, \"big\": 18446744073709551615}";
    poe_json *j = poe_json_parse(doc, strlen(doc), err, sizeof err);
    CHECK(j != NULL, "parse: %s", j ? "ok" : err);
    CHECK(poe_json_len(poe_json_get(j, "a")) == 3, "array length");
    CHECK(feq(poe_json_num(poe_json_at(poe_json_get(j, "a"), 1), 0), 2.5),
          "number 2.5");
    CHECK(strcmp(poe_json_str(poe_json_get(j, "s"), ""), "x\n\"yA") == 0,
          "string escapes incl. \\u0041");
    CHECK(poe_json_u64(poe_json_get(j, "big"), 0) == 18446744073709551615ull,
          "u64 counters stay exact");
    poe_json_free(j);

    CHECK(poe_json_parse("{\"x\": }", 8, err, sizeof err) == NULL,
          "malformed rejected: %s", err);
    CHECK(poe_json_parse("[1,2] junk", 10, err, sizeof err) == NULL,
          "trailing content rejected");

    /* ── profile load ───────────────────────────────────────────────────── */
    printf("profile load\n");
    /* layer-1 REAP scores: A ranks 3>2>1>0 by value, B identical ->
     * Spearman +1 on layer 1; layer 0 has no reap array (mixed profiles
     * still compare, reap only when both carry it — here only layer 1
     * carries it, which the loader treats as "profile has reap"). */
    CHECK(write_profile("build/pa.poeprofile", "8,8,0,0", "0.1,0.2,0.3,0.4") == 0,
          "write A");
    CHECK(write_profile("build/pb.poeprofile", "0,0,8,8", "0.1,0.2,0.3,0.4") == 0,
          "write B");

    poe_profile *a, *b;
    CHECK(poe_profile_load(&a, "build/pa.poeprofile", err, sizeof err) == 0,
          "load A: %s", err[0] ? err : "ok");
    CHECK(poe_profile_load(&b, "build/pb.poeprofile", err, sizeof err) == 0,
          "load B");
    CHECK(a->n_layers == 2 && a->n_experts == 4 && a->top_k == 2,
          "shape 2x4 top-2");
    CHECK(a->sel_count[0] == 10 && a->sel_count[4 + 1] == 8,
          "sel counts positioned per layer");
    CHECK(feq(a->gate_mean[2], 0.15), "gate mean parsed");
    CHECK(feq(a->entropy_bits[0], 1.5) && feq(a->mass_k[1][0], 3.0),
          "entropy and mass-k parsed");
    CHECK(a->mass_k_hist[0] != NULL &&
          a->mass_k_hist[0][0 * POE_PROFILE_KHIST + 0] == 8 &&
          a->mass_k_hist[0][0 * POE_PROFILE_KHIST + 1] == 8,
          "the mass-k histogram parses, bins in order");
    CHECK(a->mass_k_hist[3][0 * POE_PROFILE_KHIST + 3] == 16,
          "the .99 threshold keeps its own bins");
    CHECK(a->mass_k_hist[0][1 * POE_PROFILE_KHIST + 0] == 0,
          "a layer with no histogram stays zero rather than borrowing one");

    /* ── comparison metrics, hand-computed ──────────────────────────────── */
    printf("compare\n");
    poe_profile_cmp c;
    double jl[2];
    CHECK(poe_profile_compare(a, b, 0.5, &c, jl) == 0, "compare runs");
    CHECK(feq(jl[0], 1.0) && feq(jl[1], 0.0), "per-layer Jaccard 1.0 / 0.0");
    CHECK(feq(c.jaccard_mean, 0.5), "Jaccard mean 0.5");
    CHECK(c.jaccard_min_layer == 1 && c.jaccard_max_layer == 0,
          "min layer 1, max layer 0");
    CHECK(feq(c.wjaccard_mean, 0.5), "weighted Jaccard mean 0.5 (1 + 0)");
    CHECK(feq(c.spearman_mean, 0.0), "Spearman mean 0 (+1 and -1)");
    CHECK(feq(c.jsd_bits_mean, 0.5), "JSD mean 0.5 bits (0 + 1)");
    CHECK(c.exclusive_a == 2 && c.exclusive_b == 2,
          "2 exclusive experts each (layer 1)");
    CHECK(c.cold_both == 1, "1 slot cold in both (layer 0, expert 1)");
    CHECK(a->reap_mean != NULL && feq(a->reap_mean[4 + 3], 0.4),
          "reap array parsed");
    CHECK(c.has_reap == 1, "reap comparison available");
    /* layer 0 reap is all zeros (ties -> spearman 0), layer 1 identical
     * ranking -> +1; mean 0.5. Bottom sets: layer 0 all-tied -> same
     * deterministic pick -> 1.0; layer 1 identical -> 1.0. */
    CHECK(feq(c.reap_spearman_mean, 0.5), "reap Spearman 0.5 (%f)",
          c.reap_spearman_mean);
    CHECK(feq(c.reap_bottom_jaccard, 1.0), "reap bottom-set Jaccard 1.0");

    /* identity comparison */
    CHECK(poe_profile_compare(a, a, 0.5, &c, NULL) == 0, "self-compare");
    CHECK(feq(c.jaccard_mean, 1.0) && feq(c.spearman_mean, 1.0) &&
          feq(c.jsd_bits_mean, 0.0) && feq(c.wjaccard_mean, 1.0),
          "self: Jaccard 1, Spearman 1, JSD 0");

    poe_profile_free(a);
    poe_profile_free(b);

    /* ── errors ─────────────────────────────────────────────────────────── */
    printf("errors\n");
    poe_profile *x;
    CHECK(poe_profile_load(&x, "build/missing.poeprofile", err, sizeof err) != 0,
          "missing file rejected");
    FILE *f = fopen("build/notprofile.json", "w");
    fprintf(f, "{\"hello\": 1}");
    fclose(f);
    CHECK(poe_profile_load(&x, "build/notprofile.json", err, sizeof err) != 0,
          "non-profile rejected: %s", err);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
