/* cmd_diff.c — `poe diff <a> <b>` for plans, profiles, or models.
 *
 * Dispatches on file extension:
 *   .poeplan     per-layer prune-set agreement between two plans
 *   .poeprofile  condensed fingerprint comparison (poe compare's core)
 *   anything else is treated as GGUF: structural model diff
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/plan.h"
#include "poe/profile.h"
#include "poe/quantplan.h"
#include "poe/stats.h"
#include "cli.h"

static int has_suffix(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && strcmp(s + n - m, suf) == 0;
}

static int diff_plans(const char *pa, const char *pb) {
    char err[256];
    poe_plan *a, *b;
    if (poe_plan_load(&a, pa, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pa, err);
        return 1;
    }
    if (poe_plan_load(&b, pb, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pb, err);
        poe_plan_free(a);
        return 1;
    }

    printf("A  %s   %s %.1f%%   keep %u/%u   %s\n", pa, a->method,
           a->prune_fraction * 100.0, a->keep_per_layer, a->n_experts,
           a->model_fingerprint);
    printf("B  %s   %s %.1f%%   keep %u/%u   %s\n", pb, b->method,
           b->prune_fraction * 100.0, b->keep_per_layer, b->n_experts,
           b->model_fingerprint);
    if (strcmp(a->model_fingerprint, b->model_fingerprint) != 0)
        printf("warning: plans target different models\n");

    int rc = 0;
    if (a->n_layers != b->n_layers || a->n_experts != b->n_experts) {
        printf("shapes differ (%ux%u vs %ux%u) — nothing further to compare\n",
               a->n_layers, a->n_experts, b->n_layers, b->n_experts);
    } else {
        const uint32_t L = a->n_layers, E = a->n_experts;
        uint32_t identical = 0;
        double jsum = 0, jmin = 2.0;
        uint32_t jmin_l = 0;
        uint64_t only_a = 0, only_b = 0;
        for (uint32_t l = 0; l < L; l++) {
            uint32_t inter = 0, uni = 0;
            for (uint32_t e = 0; e < E; e++) {
                int ka = !a->keep[(size_t)l * E + e];   /* pruned sets */
                int kb = !b->keep[(size_t)l * E + e];
                if (ka && kb) inter++;
                if (ka || kb) uni++;
                if (ka && !kb) only_a++;
                if (!ka && kb) only_b++;
            }
            double jac = uni ? (double)inter / (double)uni : 1.0;
            jsum += jac;
            if (jac < jmin) { jmin = jac; jmin_l = l; }
            if (jac == 1.0) identical++;
        }
        printf("\nprune-set agreement\n");
        printf("  identical layers    %u / %u\n", identical, L);
        printf("  Jaccard             mean %.3f   min %.3f (layer %u)\n",
               jsum / L, jmin, jmin_l);
        printf("  pruned only by A    %llu expert slots\n",
               (unsigned long long)only_a);
        printf("  pruned only by B    %llu expert slots\n",
               (unsigned long long)only_b);
    }
    poe_plan_free(a);
    poe_plan_free(b);
    return rc;
}

static int diff_profiles(const char *pa, const char *pb) {
    char err[256];
    poe_profile *a, *b;
    if (poe_profile_load(&a, pa, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pa, err);
        return 1;
    }
    if (poe_profile_load(&b, pb, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pb, err);
        poe_profile_free(a);
        return 1;
    }
    int rc = 0;
    poe_profile_cmp c;
    if (poe_profile_compare(a, b, 0.25, &c, NULL) != 0) {
        fprintf(stderr, "poe diff: profiles have different shapes\n");
        rc = 1;
    } else {
        printf("A  %s   %llu tok   %s\n", pa,
               (unsigned long long)a->tokens, a->fingerprint);
        printf("B  %s   %llu tok   %s\n", pb,
               (unsigned long long)b->tokens, b->fingerprint);
        printf("top-25%% Jaccard %.3f   weighted %.3f   Spearman %.3f   "
               "JSD %.3f bits\n",
               c.jaccard_mean, c.wjaccard_mean, c.spearman_mean,
               c.jsd_bits_mean);
        if (c.has_reap)
            printf("REAP Spearman %.3f   bottom-set Jaccard %.3f\n",
                   c.reap_spearman_mean, c.reap_bottom_jaccard);
        printf("(full detail: poe compare)\n");
    }
    poe_profile_free(a);
    poe_profile_free(b);
    return rc;
}

/* Two bit maps over the same slabs. The question a reader has is how far
 * apart two allocations actually fall — whether swapping the ranking moved
 * anything, and by how many bytes — so that is what this prints. */
static int diff_quantplans(const char *pa, const char *pb) {
    char err[256];
    poe_quantplan *a, *b;
    if (poe_quantplan_load(&a, pa, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pa, err);
        return 1;
    }
    if (poe_quantplan_load(&b, pb, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pb, err);
        poe_quantplan_free(a);
        return 1;
    }

    char s1[32], s2[32];
    poe_format_bytes(a->bytes_after_total, s1, sizeof s1);
    poe_format_bytes(b->bytes_after_total, s2, sizeof s2);
    printf("A  %s   %s   %s slabs   %s\n", pa, a->method, s1, a->model_fingerprint);
    printf("B  %s   %s   %s slabs   %s\n", pb, b->method, s2, b->model_fingerprint);
    if (strcmp(a->model_fingerprint, b->model_fingerprint) != 0)
        printf("warning: plans target different checkpoints\n");

    int rc = 0;
    if (a->n_layers != b->n_layers) {
        printf("layer counts differ (%u vs %u) — nothing further to compare\n",
               a->n_layers, b->n_layers);
    } else {
        const size_t n = (size_t)a->n_layers * POE_QSLAB_NPROJ;
        uint32_t same = 0, present = 0, a_wider = 0, b_wider = 0;
        int64_t byte_delta = 0;
        for (size_t i = 0; i < n; i++) {
            if (a->type[i] < 0 || b->type[i] < 0) continue;
            present++;
            if (a->type[i] == b->type[i]) same++;
            else if (a->bytes_after[i] > b->bytes_after[i]) a_wider++;
            else b_wider++;
            byte_delta += (int64_t)a->bytes_after[i] - (int64_t)b->bytes_after[i];
        }
        printf("\ntype map\n");
        printf("  slabs with the same type   %u / %u\n", same, present);
        printf("  wider in A                 %u\n", a_wider);
        printf("  wider in B                 %u\n", b_wider);
        poe_format_bytes(byte_delta < 0 ? (uint64_t)-byte_delta
                                        : (uint64_t)byte_delta, s1, sizeof s1);
        printf("  A - B                      %s%s\n",
               byte_delta < 0 ? "-" : "+", s1);

        const int *ladder;
        const size_t nl = poe_quantplan_ladder(&ladder, NULL);
        printf("\n%-8s %8s %8s\n", "type", "A", "B");
        for (size_t t = 0; t < nl; t++) {
            uint32_t ca = 0, cb = 0;
            for (size_t i = 0; i < n; i++) {
                if (a->type[i] == ladder[t]) ca++;
                if (b->type[i] == ladder[t]) cb++;
            }
            if (ca || cb)
                printf("%-8s %8u %8u\n", ingot_type_name(ladder[t]), ca, cb);
        }
        printf("\nlayer scores   Spearman %+.3f   (depth %+.3f vs %+.3f)\n",
               poe_spearman(a->layer_score, b->layer_score, a->n_layers),
               a->depth_rho, b->depth_rho);
    }
    poe_quantplan_free(a);
    poe_quantplan_free(b);
    return rc;
}

static int diff_models(const char *pa, const char *pb) {
    char err[256];
    poe_model *a, *b;
    if (poe_model_open(&a, pa, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pa, err);
        return 1;
    }
    if (poe_model_open(&b, pb, err, sizeof err) != 0) {
        fprintf(stderr, "poe diff: %s: %s\n", pb, err);
        poe_model_close(a);
        return 1;
    }

    char s1[32], s2[32];
    poe_format_bytes(a->total_bytes, s1, sizeof s1);
    poe_format_bytes(b->total_bytes, s2, sizeof s2);
    printf("A  %s   %s   %u blocks   %u experts   %s\n", pa,
           a->arch, a->n_blocks, a->expert_count, a->fingerprint);
    printf("B  %s   %s   %u blocks   %u experts   %s\n", pb,
           b->arch, b->n_blocks, b->expert_count, b->fingerprint);
    if (strcmp(a->fingerprint, b->fingerprint) == 0) {
        printf("models are identical (same fingerprint)\n");
        poe_model_close(a); poe_model_close(b);
        return 0;
    }

    printf("\n%-22s %14s %14s\n", "", "A", "B");
    printf("%-22s %14s %14s\n", "total size", s1, s2);
    poe_format_bytes(a->expert_bytes, s1, sizeof s1);
    poe_format_bytes(b->expert_bytes, s2, sizeof s2);
    printf("%-22s %14s %14s\n", "expert bytes", s1, s2);
    poe_format_params(a->total_params, s1, sizeof s1);
    poe_format_params(b->total_params, s2, sizeof s2);
    printf("%-22s %14s %14s\n", "params", s1, s2);
    printf("%-22s %14u %14u\n", "experts/block",
           a->expert_count, b->expert_count);
    printf("%-22s %14u %14u\n", "active/token",
           a->experts_per_token, b->experts_per_token);

    /* tensors present in only one of the two */
    size_t only_a = 0, only_b = 0;
    for (size_t i = 0; i < ingot_gguf_count(a->g); i++) {
        const ingot_tensor *t = ingot_gguf_at(a->g, i);
        if (ingot_gguf_find(b->g, t->name) == NULL) only_a++;
    }
    for (size_t i = 0; i < ingot_gguf_count(b->g); i++) {
        const ingot_tensor *t = ingot_gguf_at(b->g, i);
        if (ingot_gguf_find(a->g, t->name) == NULL) only_b++;
    }
    printf("%-22s %14zu %14zu\n", "tensors only here", only_a, only_b);

    poe_model_close(a);
    poe_model_close(b);
    return 0;
}

int poe_cmd_diff(int argc, char **argv) {
    const char *pa = NULL, *pb = NULL;
    for (int i = 0; i < argc; i++) {
        if (argv[i][0] == '-') {
            fprintf(stderr, "poe diff: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (pa == NULL) pa = argv[i];
        else if (pb == NULL) pb = argv[i];
        else { fprintf(stderr, "poe diff: too many arguments\n"); return 2; }
    }
    if (pa == NULL || pb == NULL) {
        fprintf(stderr, "usage: poe diff <a> <b>   "
                        "(two .poeplan, .poeprofile, .poequant, or .gguf files)\n");
        return 2;
    }

    /* Kind is decided by extension, and mixing kinds is refused rather than
     * silently falling through to the GGUF path. */
    enum { KIND_GGUF, KIND_PLAN, KIND_PROFILE, KIND_QUANT };
    const char *paths[2] = { pa, pb };
    int kind[2];
    for (int i = 0; i < 2; i++)
        kind[i] = has_suffix(paths[i], ".poeplan")    ? KIND_PLAN
                : has_suffix(paths[i], ".poeprofile") ? KIND_PROFILE
                : has_suffix(paths[i], ".poequant")   ? KIND_QUANT
                                                      : KIND_GGUF;
    if (kind[0] != kind[1]) {
        fprintf(stderr, "poe diff: cannot compare artifacts of different kinds\n");
        return 2;
    }
    switch (kind[0]) {
    case KIND_PLAN:    return diff_plans(pa, pb);
    case KIND_PROFILE: return diff_profiles(pa, pb);
    case KIND_QUANT:   return diff_quantplans(pa, pb);
    default:           return diff_models(pa, pb);
    }
}
