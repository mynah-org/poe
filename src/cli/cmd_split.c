/* cmd_split.c — `poe split <model.gguf> -o out.gguf --profile p
 *                 --hot-frac F --hot-type T --cold-type T`
 *
 * The checkpoint half of M9b: experts reordered by need, each slab stored as
 * a hot tensor and a cold tensor at different precisions. See
 * include/poe/split.h for the layout and why it needs no id remap.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/split.h"
#include "cli.h"

static int type_by_name(const char *s) {
    for (int t = 0; t < 64; t++) {
        const char *n = ingot_type_name(t);
        if (n == NULL || !ingot_can_quantize(t)) continue;
        size_t i = 0;
        for (; n[i] && s[i]; i++) {
            const char a = n[i] >= 'A' && n[i] <= 'Z' ? (char)(n[i] + 32) : n[i];
            const char b = s[i] >= 'A' && s[i] <= 'Z' ? (char)(s[i] + 32) : s[i];
            if (a != b) break;
        }
        if (n[i] == '\0' && s[i] == '\0') return t;
    }
    return -1;
}

int poe_cmd_split(int argc, char **argv) {
    const char *model_path = NULL, *out_path = NULL, *profile_path = NULL;
    const char *hot_arg = NULL, *cold_arg = NULL;
    double frac = 0.5;
    int invert = 0, force = 0, threads = 0, by_counts = 0;

    for (int i = 0; i < argc; i++) {
        if      (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_path = argv[++i];
        else if (strcmp(argv[i], "--hot-type") == 0 && i + 1 < argc) hot_arg = argv[++i];
        else if (strcmp(argv[i], "--cold-type") == 0 && i + 1 < argc) cold_arg = argv[++i];
        else if (strcmp(argv[i], "--hot-frac") == 0 && i + 1 < argc) {
            frac = atof(argv[++i]);
            if (frac > 1.0) frac /= 100.0;
        }
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if      (strcmp(v, "counts") == 0) by_counts = 1;
            else if (strcmp(v, "reap")   == 0) by_counts = 0;
            else { fprintf(stderr, "poe split: --rank must be 'reap' or "
                                   "'counts'\n"); return 2; }
        }
        else if (strcmp(argv[i], "--invert") == 0) invert = 1;
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe split: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (model_path == NULL) model_path = argv[i];
        else { fprintf(stderr, "poe split: more than one model given\n"); return 2; }
    }
    if (model_path == NULL || out_path == NULL || profile_path == NULL ||
        hot_arg == NULL || cold_arg == NULL) {
        fprintf(stderr,
            "usage: poe split <model.gguf> -o <out.gguf> --profile p.poeprofile\n"
            "                 --hot-type TYPE --cold-type TYPE [--hot-frac F]\n"
            "                 [--rank reap|counts] [--invert] [--threads N]\n"
            "\n"
            "  --hot-frac       share of experts kept at the hot type (default 0.5)\n"
            "  --rank           reap (default) or counts: contribution to the layer\n"
            "                   output, or how often the workload routes there\n"
            "  --invert         keep the LEAST needed experts wide: the control\n"
            "\n"
            "Experts are reordered so the hot ones come first and the router rows\n"
            "follow, so a runtime tests `id < poe.split.hot_count` and needs no\n"
            "remap. The output does NOT load in a stock llama.cpp: consuming it\n"
            "takes two mul_mat_id passes and a merge in build_moe_ffn.\n");
        return 2;
    }

    poe_split_opts o;
    memset(&o, 0, sizeof o);
    o.hot_fraction = frac;
    o.rank_by = by_counts ? POE_RANK_COUNTS : POE_RANK_REAP;
    o.invert = invert;
    o.force = force;
    o.threads = threads;
    o.hot_type = type_by_name(hot_arg);
    o.cold_type = type_by_name(cold_arg);
    if (o.hot_type < 0 || o.cold_type < 0) {
        fprintf(stderr, "poe split: unknown or unsupported type '%s'\n",
                o.hot_type < 0 ? hot_arg : cold_arg);
        return 2;
    }

    char err[256];
    poe_model *m = NULL;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe split: %s\n", err);
        return 1;
    }
    poe_profile *prof = NULL;
    if (poe_profile_load(&prof, profile_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe split: %s\n", err);
        poe_model_close(m);
        return 1;
    }
    o.profile = prof;

    poe_split_stats st;
    if (poe_split(m, &o, out_path, &st, err, sizeof err) != 0) {
        fprintf(stderr, "poe split: %s\n", err);
        poe_profile_free(prof);
        poe_model_close(m);
        return 1;
    }

    char b1[32], b2[32];
    poe_format_bytes(st.bytes_before, b1, sizeof b1);
    poe_format_bytes(st.bytes_written, b2, sizeof b2);
    printf("model     %s   %s   (%s, %u layers x %u experts)\n",
           model_path, m->fingerprint, m->arch, m->n_blocks, m->expert_count);
    printf("hot       %-6s  %.4f bits/weight   %u experts per layer%s\n",
           ingot_type_name(o.hot_type), st.hot_bits, st.hot_per_layer,
           invert ? "   (INVERTED: the least needed)" : "");
    printf("cold      %-6s  %.4f bits/weight   %u experts per layer\n",
           ingot_type_name(o.cold_type), st.cold_bits, st.cold_per_layer);
    printf("ranked by %s\n", st.ranked_by_reap ? "REAP saliency"
                                               : "selection frequency");
    printf("mean      %.4f bits/weight over the routed experts\n", st.mean_bits);
    printf("\n%u slabs split, %u router tensors permuted\n",
           st.slabs_split, st.routers_permuted);
    printf("wrote %s   %s  (source %s, %.1f%%)\n", out_path, b2, b1,
           st.bytes_before ? 100.0 * (double)st.bytes_written /
                             (double)st.bytes_before : 0.0);
    printf("note: this checkpoint needs a runtime that reads "
           "poe.split.hot_count and\n      runs the hot and cold experts as two "
           "mul_mat_id passes. Stock\n      llama.cpp will not load it.\n");

    poe_profile_free(prof);
    poe_model_close(m);
    return 0;
}
