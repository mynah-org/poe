/* test_super.c — super-expert detection, and the guard it feeds.
 *
 * The failure this defends against is silent: a model that lost three super
 * experts still writes fluent text, so nothing downstream will complain. The
 * assertions therefore pin the properties that make the guard trustworthy —
 * that a planted outlier is found and an ordinary spread is not, that a flat
 * layer is reported as undecidable rather than as clean, that protection
 * actually changes which experts a cut removes, and that it does so without
 * touching the per-layer keep count, which `poe apply` requires to be
 * uniform.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/plan.h"
#include "poe/super.h"
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

/* One expert (7) is a planted super expert: rarely selected, activation norm
 * two orders of magnitude above its neighbours, and low REAP saliency so
 * that every ranking would delete it. Expert 3 is merely rare. */
static const char *SEL     = "[80,10,70,5,60,50,40,3]";
static const char *ACT_SUP = "[1.0,1.1,0.9,1.05,0.95,1.02,0.98,50.0]";
static const char *ACT_FLAT= "[1,1,1,1,1,1,1,1]";
static const char *REAP    = "[0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.02]";

static int write_profile(const char *path, const char *fingerprint,
                         const char *actnorm) {
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
        fprintf(f, " \"sel_count\": %s,\n", SEL);
        fprintf(f, " \"gate_mean\": [0.8,0.1,0.7,0.05,0.6,0.5,0.4,0.3],\n");
        fprintf(f, " \"reap_mean\": %s,\n \"actnorm_mean\": %s}%s\n",
                REAP, actnorm, l + 1 < POE_FIX_BLOCKS ? "," : "");
    }
    fprintf(f, "]\n}\n");
    fclose(f);
    return 0;
}

/* how many experts this layer keeps */
static uint32_t kept(const poe_plan *p, uint32_t l) {
    uint32_t n = 0;
    for (uint32_t e = 0; e < p->n_experts; e++)
        n += p->keep[(size_t)l * p->n_experts + e] != 0;
    return n;
}

int main(void) {
    char err[256] = { 0 };
    const char *model = "build/fixture-super.gguf";

    CHECK(poe_fixture_moe(model, 7, err, sizeof err) == 0,
          "fixture written (%s)", err[0] ? err : "ok");
    poe_model *m = NULL;
    CHECK(poe_model_open(&m, model, err, sizeof err) == 0, "open the fixture");
    if (m == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                     return 1; }

    CHECK(write_profile("build/super.poeprofile", m->fingerprint, ACT_SUP) == 0,
          "profile with a planted super expert");
    CHECK(write_profile("build/flat.poeprofile", m->fingerprint, ACT_FLAT) == 0,
          "profile with a flat activation spread");

    poe_profile *pr = NULL, *flat = NULL;
    CHECK(poe_profile_load(&pr, "build/super.poeprofile", err, sizeof err) == 0,
          "load the planted profile (%s)", err[0] ? err : "ok");
    CHECK(poe_profile_load(&flat, "build/flat.poeprofile", err, sizeof err) == 0,
          "load the flat profile");
    if (pr == NULL || flat == NULL) {
        printf("\n%d checks, %d failures\n", checks, ++failures); return 1;
    }

    /* ── detection ──────────────────────────────────────────────────────── */
    {
        poe_super *s = NULL;
        CHECK(poe_super_detect(&s, pr, 0, err, sizeof err) == 0,
              "detect over the planted profile (%s)", err[0] ? err : "ok");
        if (s) {
            CHECK(s->n_outliers == POE_FIX_BLOCKS,
                  "exactly one outlier per layer, %u total", (unsigned)s->n_outliers);
            CHECK(s->is_outlier[7] == 1, "the planted expert 7 is flagged");
            CHECK(s->is_rare_outlier[7] == 1,
                  "and it carries the published signature: rare as well as huge");
            int others = 0;
            for (uint32_t e = 0; e < 7; e++) others += s->is_outlier[e];
            CHECK(others == 0, "an ordinary +-10%% spread flags nobody (%d)", others);
            CHECK(s->z_max > 100.0, "its robust z is enormous (%.0f)", s->z_max);
            CHECK(s->n_layers_undecidable == 0, "no layer was undecidable");
            poe_super_free(s);
        }
    }

    /* A flat layer has MAD 0: no z-score exists, and saying "no outliers"
     * there would be a claim the data cannot support. */
    {
        poe_super *s = NULL;
        CHECK(poe_super_detect(&s, flat, 0, err, sizeof err) == 0,
              "detect over the flat profile");
        if (s) {
            CHECK(s->n_outliers == 0, "nothing flagged");
            CHECK(s->n_layers_undecidable == POE_FIX_BLOCKS,
                  "every layer reported undecidable, not clean (%u)",
                  s->n_layers_undecidable);
            poe_super_free(s);
        }
    }

    /* Raising the threshold past the planted z must silence it: the
     * parameter has to be doing the work the name says. */
    {
        poe_super *s = NULL;
        poe_super_detect(&s, pr, 1e9, err, sizeof err);
        CHECK(s && s->n_outliers == 0, "a threshold above the planted z flags nobody");
        poe_super_free(s);
    }

    /* ── protection inside a plan ───────────────────────────────────────── */
    const poe_profile *one[1] = { pr };
    const double w[1] = { 1.0 };

    poe_plan *unprot = NULL, *prot = NULL;
    poe_plan_opts o = { .profiles = one, .weights = w, .n_profiles = 1,
                        .method = "frequency", .prune_frac = 0.25,
                        .force = 0, .protect_super = 0, .super_z = 0 };
    CHECK(poe_plan_build_opts(&unprot, m, &o, err, sizeof err) == 0,
          "plan without protection (%s)", err[0] ? err : "ok");
    o.protect_super = 1;
    CHECK(poe_plan_build_opts(&prot, m, &o, err, sizeof err) == 0,
          "plan with protection");

    if (unprot && prot) {
        const uint32_t E = prot->n_experts;
        CHECK(unprot->keep[7] == 0,
              "unprotected, frequency deletes the super expert");
        CHECK(prot->keep[7] == 1, "protected, it survives");
        CHECK(prot->n_super_flagged == POE_FIX_BLOCKS,
              "the plan records what was flagged (%u)", prot->n_super_flagged);
        CHECK(prot->n_super_rescued == POE_FIX_BLOCKS,
              "and what protection actually bought: %u rescued",
              prot->n_super_rescued);
        CHECK(prot->n_super_still_pruned == 0, "nothing had to be pruned anyway");

        /* The invariant poe apply depends on. */
        int uniform = 1;
        for (uint32_t l = 0; l < prot->n_layers; l++)
            uniform &= kept(prot, l) == prot->keep_per_layer;
        CHECK(uniform, "every layer still keeps exactly keep_per_layer (%u)",
              prot->keep_per_layer);
        CHECK(prot->keep_per_layer == unprot->keep_per_layer,
              "protection did not change the size target");
        CHECK(prot->bytes_removed == unprot->bytes_removed,
              "nor the exact byte accounting (%llu)",
              (unsigned long long)prot->bytes_removed);

        /* Someone else took the place. */
        uint32_t differ = 0;
        for (uint32_t e = 0; e < E; e++)
            differ += prot->keep[e] != unprot->keep[e];
        CHECK(differ == 2, "one expert swapped for another in the cut (%u changed)",
              differ);
    }

    /* Round-trip: the record of what the guard did must survive the file. */
    if (prot) {
        CHECK(poe_plan_write(prot, "build/super.poeplan", err, sizeof err) == 0,
              "write the plan");
        poe_plan *back = NULL;
        CHECK(poe_plan_load(&back, "build/super.poeplan", err, sizeof err) == 0,
              "load it back (%s)", err[0] ? err : "ok");
        if (back) {
            CHECK(back->protect_super == 1 &&
                  back->n_super_rescued == prot->n_super_rescued,
                  "protection and rescue count round-trip (%d, %u)",
                  back->protect_super, back->n_super_rescued);
            poe_plan_free(back);
        }
    }

    /* Asking for protection a profile cannot support must say so. */
    {
        poe_profile *nore = NULL;
        write_profile("build/super-noact.poeprofile", m->fingerprint, ACT_FLAT);
        if (poe_profile_load(&nore, "build/super-noact.poeprofile", err, sizeof err) == 0) {
            const poe_profile *p1[1] = { nore };
            poe_plan_opts o2 = { .profiles = p1, .weights = w, .n_profiles = 1,
                                 .method = "frequency", .prune_frac = 0.25,
                                 .protect_super = 1 };
            poe_plan *pl = NULL;
            if (poe_plan_build_opts(&pl, m, &o2, err, sizeof err) == 0 && pl) {
                CHECK(pl->n_warnings > 0,
                      "a flat profile yields a warning, not silent confidence");
                poe_plan_free(pl);
            }
            poe_profile_free(nore);
        }
    }

    poe_plan_free(unprot);
    poe_plan_free(prot);
    poe_profile_free(pr);
    poe_profile_free(flat);
    poe_model_close(m);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
