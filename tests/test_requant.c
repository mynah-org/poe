/* test_requant.c — per-expert precision emulated inside one carrier type.
 *
 * The properties that make the M9b experiment readable are the ones checked
 * here: two arms at the same average bits are byte-identical in size, the
 * cold set is the one the profile says is cold, the inverted control really
 * is the complement, and the degraded experts are measurably further from
 * the source than the untouched ones.
 *
 * SPDX-License-Identifier: MIT */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/requant.h"
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

static uint64_t file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n > 0 ? (uint64_t)n : 0;
}

/* Mean squared difference between a rewritten slab and the source's, per
 * expert — the direct question "was this expert damaged more than that one". */
static int expert_error(const poe_model *src, const poe_model *out,
                        const char *name, uint32_t E, double *mse) {
    const ingot_tensor *a = ingot_gguf_find(src->g, name);
    const ingot_tensor *b = ingot_gguf_find(out->g, name);
    if (a == NULL || b == NULL || a->nelem != b->nelem) return -1;
    float *fa = malloc((size_t)a->nelem * sizeof *fa);
    float *fb = malloc((size_t)b->nelem * sizeof *fb);
    if (fa == NULL || fb == NULL ||
        ingot_gguf_dequant(src->g, a, fa) != 0 ||
        ingot_gguf_dequant(out->g, b, fb) != 0) {
        free(fa); free(fb);
        return -1;
    }
    const uint64_t per = a->nelem / E;
    for (uint32_t e = 0; e < E; e++) {
        double s = 0.0;
        for (uint64_t i = 0; i < per; i++) {
            const double d = (double)fa[e * per + i] - (double)fb[e * per + i];
            s += d * d;
        }
        mse[e] = s / (double)per;
    }
    free(fa); free(fb);
    return 0;
}

/* The assumption the whole emulation rests on: once values sit on the
 * carrier's grid, encoding them at the carrier again costs (almost) nothing.
 * If it did not, every arm would be paying an unknown second tax and the
 * comparison between them would mean nothing. Measured here rather than
 * asserted, on the bell-shaped distribution a weight matrix actually has. */
static double rel_l2(const float *a, const float *b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)a[i] * (double)a[i];
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

static void check_carrier_idempotence(void) {
    enum { N = 4096 };
    float *x  = malloc(N * sizeof *x);
    float *r1 = malloc(N * sizeof *r1);
    float *r2 = malloc(N * sizeof *r2);
    uint8_t *q = malloc(N);            /* Q4_K is 4.5 bits: N bytes is ample */
    if (!x || !r1 || !r2 || !q) { free(x); free(r1); free(r2); free(q); return; }

    uint32_t s = 12345u;
    for (size_t i = 0; i < N; i++) {   /* sum of uniforms ~ bell shaped */
        double v = 0;
        for (int k = 0; k < 6; k++) {
            s = s * 1664525u + 1013904223u;
            v += (double)(int32_t)s / 2147483648.0;
        }
        x[i] = (float)(v / 6.0);
    }

    if (ingot_quantize(INGOT_TYPE_Q4_K, x, N, q) == 0 &&
        ingot_dequant(INGOT_TYPE_Q4_K, q, N, r1) == 0 &&
        ingot_quantize(INGOT_TYPE_Q4_K, r1, N, q) == 0 &&
        ingot_dequant(INGOT_TYPE_Q4_K, q, N, r2) == 0) {
        const double first  = rel_l2(x, r1, N);
        const double second = rel_l2(r1, r2, N);
        CHECK(first > 0.01, "one Q4_K round trip costs rel L2 %.4f", first);
        CHECK(second < first / 10.0,
              "a second round trip adds %.5f, an order of magnitude less "
              "— the carrier pass is near-idempotent", second);
    } else {
        printf("  FAIL: cannot run the carrier round trip\n");
        failures++;
    }

    /* And the tax the emulation actually charges: storing a hard-quantized
     * signal at the carrier instead of at the hard type. If that were large,
     * an emulated arm would not stand in for the real allocation. */
    const int hard[2] = { INGOT_TYPE_Q3_K, INGOT_TYPE_Q2_K };
    for (int h = 0; h < 2; h++) {
        if (ingot_quantize(hard[h], x, N, q) != 0 ||
            ingot_dequant(hard[h], q, N, r1) != 0 ||
            ingot_quantize(INGOT_TYPE_Q4_K, r1, N, q) != 0 ||
            ingot_dequant(INGOT_TYPE_Q4_K, q, N, r2) != 0) {
            printf("  FAIL: cannot run the %s emulation path\n",
                   ingot_type_name(hard[h]));
            failures++;
            continue;
        }
        const double direct   = rel_l2(x, r1, N);
        const double emulated = rel_l2(x, r2, N);
        /* The tax is small but NOT equal across types — measured here at
         * +7% for Q3_K and +2% for Q2_K — so an arm that degrades everything
         * to Q3_K pays more of it than one that degrades some experts to
         * Q2_K. That is why the inverted control, which carries exactly the
         * same tax as the arm under test, is the comparison that decides,
         * and the uniform arm is a practical baseline rather than a clean
         * one. Bounded, reported, and never assumed to be zero. */
        CHECK(emulated <= direct * 1.10,
              "%s emulated through the carrier costs %.4f against %.4f "
              "stored directly (+%.1f%%)", ingot_type_name(hard[h]),
              emulated, direct, 100.0 * (emulated / direct - 1.0));
    }
    free(x); free(r1); free(r2); free(q);
}

int main(void) {
    check_carrier_idempotence();

    const char *src_path = "build/fixture-wide.gguf";
    const char *uni_path = "build/rq-uniform.gguf";
    const char *pe_path  = "build/rq-perexpert.gguf";
    const char *inv_path = "build/rq-inverted.gguf";
    char err[256] = { 0 };

    CHECK(poe_fixture_moe_wide(src_path, 3, err, sizeof err) == 0,
          "wide fixture written (%s)", err[0] ? err : "ok");

    poe_model *src = NULL;
    CHECK(poe_model_open(&src, src_path, err, sizeof err) == 0,
          "open the fixture (%s)", err[0] ? err : "ok");
    if (src == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                       return 1; }
    const uint32_t E = src->expert_count, L = src->n_blocks;

    /* A profile whose REAP saliency makes the ordering unambiguous: expert e
     * is exactly e+1 salient, so the coldest half is {0,1} in every layer. */
    poe_profile pr;
    memset(&pr, 0, sizeof pr);
    pr.n_layers = L;
    pr.n_experts = E;
    pr.tokens = 8192;
    snprintf(pr.fingerprint, sizeof pr.fingerprint, "%s", src->fingerprint);
    pr.reap_mean = calloc((size_t)L * E, sizeof *pr.reap_mean);
    pr.sel_count = calloc((size_t)L * E, sizeof *pr.sel_count);
    if (pr.reap_mean == NULL || pr.sel_count == NULL) return 1;
    for (uint32_t l = 0; l < L; l++)
        for (uint32_t e = 0; e < E; e++) {
            pr.reap_mean[(size_t)l * E + e] = (double)(e + 1);
            pr.sel_count[(size_t)l * E + e] = e + 1;
        }

    /* ── the control: every expert degraded through the same type ─────── */
    poe_requant_opts uni;
    memset(&uni, 0, sizeof uni);
    uni.carrier_type = INGOT_TYPE_Q4_K;
    uni.degrade_type = INGOT_TYPE_Q3_K;
    uni.degrade_fraction = 1.0;
    poe_requant_stats su;
    CHECK(poe_requant(src, &uni, uni_path, &su, err, sizeof err) == 0,
          "uniform arm written (%s)", err[0] ? err : "ok");
    CHECK(su.slabs_rewritten == L * 3,
          "%u routed slabs rewritten", su.slabs_rewritten);
    CHECK(su.experts_degraded == (uint64_t)L * 3 * E,
          "every expert degraded (%llu)",
          (unsigned long long)su.experts_degraded);
    CHECK(fabs(su.emulated_bits - 3.4375) < 1e-9,
          "the control emulates %.4f bits/weight", su.emulated_bits);

    /* ── the arm under test: half the experts, hard ────────────────────── */
    poe_requant_opts pe;
    memset(&pe, 0, sizeof pe);
    pe.carrier_type = INGOT_TYPE_Q4_K;
    pe.degrade_type = INGOT_TYPE_Q2_K;
    pe.degrade_fraction = 0.5;
    pe.profile = &pr;
    poe_requant_stats sp;
    CHECK(poe_requant(src, &pe, pe_path, &sp, err, sizeof err) == 0,
          "per-expert arm written (%s)", err[0] ? err : "ok");
    CHECK(sp.degraded_per_layer == E / 2,
          "%u of %u experts degraded per layer", sp.degraded_per_layer, E);
    CHECK(sp.ranked_by_reap, "ranked by REAP saliency, not selection counts");
    CHECK(fabs(sp.emulated_bits - (0.5 * 2.625 + 0.5 * 4.5)) < 1e-9,
          "emulates %.4f bits/weight", sp.emulated_bits);

    /* The property the whole experiment rests on: same carrier, same size. */
    CHECK(file_size(uni_path) == file_size(pe_path),
          "both arms are the same size (%llu B)",
          (unsigned long long)file_size(pe_path));
    CHECK(su.bytes_written == sp.bytes_written,
          "and both agree on what they wrote");

    /* ── the control that says the aiming worked ───────────────────────── */
    poe_requant_opts iv = pe;
    iv.invert = 1;
    poe_requant_stats si;
    CHECK(poe_requant(src, &iv, inv_path, &si, err, sizeof err) == 0,
          "inverted arm written (%s)", err[0] ? err : "ok");
    CHECK(file_size(inv_path) == file_size(pe_path),
          "the control is the same size too");

    /* Which experts actually took the damage. */
    poe_model *mo = NULL, *mi = NULL;
    double *e_pe = malloc(E * sizeof *e_pe), *e_iv = malloc(E * sizeof *e_iv);
    if (poe_model_open(&mo, pe_path, err, sizeof err) == 0 &&
        poe_model_open(&mi, inv_path, err, sizeof err) == 0 &&
        e_pe != NULL && e_iv != NULL &&
        expert_error(src, mo, "blk.0.ffn_gate_exps.weight", E, e_pe) == 0 &&
        expert_error(src, mi, "blk.0.ffn_gate_exps.weight", E, e_iv) == 0) {
        /* saliency is e+1, so experts 0..E/2-1 are the cold half */
        double cold_max = 0, hot_min = 1e30;
        for (uint32_t e = 0; e < E; e++) {
            if (e < E / 2) { if (e_pe[e] > cold_max) cold_max = e_pe[e]; }
            else           { if (e_pe[e] < hot_min)  hot_min  = e_pe[e]; }
        }
        CHECK(cold_max > hot_min,
              "the cold half is damaged more than the hot half "
              "(%.3e vs %.3e)", cold_max, hot_min);

        int mirrored = 1;
        for (uint32_t e = 0; e < E; e++) {
            const int pe_cold = e_pe[e] > 0.5 * (cold_max + hot_min);
            const int iv_cold = e_iv[e] > 0.5 * (cold_max + hot_min);
            if (pe_cold == iv_cold) mirrored = 0;
        }
        CHECK(mirrored, "the inverted arm damaged exactly the complement");
    } else {
        printf("  FAIL: cannot reopen the arms to compare per-expert error\n");
        failures++;
    }
    free(e_pe); free(e_iv);
    poe_model_close(mo);
    poe_model_close(mi);

    /* ── the two rankings are not the same ordering ────────────────────── */
    {
        /* saliency says expert e is worth e+1; frequency says the opposite,
         * so the two cold sets must be complements of each other */
        for (uint32_t l = 0; l < L; l++)
            for (uint32_t e = 0; e < E; e++)
                pr.sel_count[(size_t)l * E + e] = E - e;

        poe_requant_opts byc = pe;
        byc.rank_by_counts = 1;
        poe_requant_stats sc;
        const char *pc = "build/rq-counts.gguf";
        CHECK(poe_requant(src, &byc, pc, &sc, err, sizeof err) == 0,
              "frequency-ranked arm written (%s)", err[0] ? err : "ok");
        CHECK(!sc.ranked_by_reap,
              "--rank counts uses frequency even when REAP data is present");

        poe_model *mc = NULL;
        double *e_c = malloc(E * sizeof *e_c);
        if (e_c != NULL && poe_model_open(&mc, pc, err, sizeof err) == 0 &&
            expert_error(src, mc, "blk.0.ffn_gate_exps.weight", E, e_c) == 0) {
            /* frequency is E-e, so the cold half is now the TOP half */
            double hot_half = 0, cold_half = 1e30;
            for (uint32_t e = 0; e < E; e++) {
                if (e < E / 2) { if (e_c[e] > hot_half)   hot_half  = e_c[e]; }
                else           { if (e_c[e] < cold_half)  cold_half = e_c[e]; }
            }
            CHECK(cold_half > hot_half,
                  "the frequency ranking picked the other half (%.3e vs %.3e)",
                  cold_half, hot_half);
        } else {
            printf("  FAIL: cannot reopen the frequency-ranked arm\n");
            failures++;
        }
        free(e_c);
        poe_model_close(mc);
        remove(pc);

        /* a profile with no counts cannot be ranked by them */
        poe_profile nocounts = pr;
        nocounts.sel_count = NULL;
        poe_requant_opts bad_rank = byc;
        bad_rank.profile = &nocounts;
        poe_requant_stats sbr;
        err[0] = '\0';
        CHECK(poe_requant(src, &bad_rank, "build/rq-bad.gguf", &sbr,
                          err, sizeof err) != 0,
              "ranking by counts without counts is refused (%s)", err);
    }

    /* Threading must not change a single byte: experts are encoded
     * independently either way, so one thread and four have to agree. */
    {
        poe_requant_opts one = pe, four = pe;
        one.threads = 1;
        four.threads = 4;
        poe_requant_stats s1, s4;
        const char *p1 = "build/rq-t1.gguf", *p4 = "build/rq-t4.gguf";
        if (poe_requant(src, &one, p1, &s1, err, sizeof err) == 0 &&
            poe_requant(src, &four, p4, &s4, err, sizeof err) == 0) {
            FILE *fa = fopen(p1, "rb"), *fb = fopen(p4, "rb");
            int same = fa != NULL && fb != NULL;
            while (same) {
                const int x = fgetc(fa), y = fgetc(fb);
                if (x != y) same = 0;
                if (x == EOF || y == EOF) break;
            }
            if (fa) fclose(fa);
            if (fb) fclose(fb);
            CHECK(same, "1 thread and 4 produce byte-identical output");
            CHECK(s1.threads == 1 && s4.threads == 4,
                  "the thread count is reported (%u vs %u)", s1.threads,
                  s4.threads);
            CHECK(s1.experts_degraded == s4.experts_degraded,
                  "and both degraded the same experts (%llu)",
                  (unsigned long long)s4.experts_degraded);
            remove(p1);
            remove(p4);
        } else {
            printf("  FAIL: cannot write the threading arms: %s\n", err);
            failures++;
        }
    }

    /* ── guards ────────────────────────────────────────────────────────── */
    poe_requant_opts bad = pe;
    bad.profile = NULL;
    poe_requant_stats sb;
    err[0] = '\0';
    CHECK(poe_requant(src, &bad, "build/rq-bad.gguf", &sb, err, sizeof err) != 0,
          "a partial degrade without a profile is refused (%s)", err);
    bad = pe;
    bad.degrade_type = INGOT_TYPE_Q6_K;      /* wider than the carrier */
    err[0] = '\0';
    CHECK(poe_requant(src, &bad, "build/rq-bad.gguf", &sb, err, sizeof err) != 0,
          "a degrade type wider than the carrier is refused (%s)", err);

    /* The narrow fixture's rows are 32 elements: no K-quant can live there,
     * and saying so beats writing a tensor no runtime can read. */
    poe_model *narrow = NULL;
    if (poe_model_open(&narrow, "build/fixture-moe.gguf", err, sizeof err) == 0) {
        err[0] = '\0';
        CHECK(poe_requant(narrow, &uni, "build/rq-bad.gguf", &sb,
                          err, sizeof err) != 0,
              "rows too narrow for the carrier are refused (%s)", err);
        poe_model_close(narrow);
    }

    free(pr.reap_mean);
    free(pr.sel_count);
    poe_model_close(src);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
