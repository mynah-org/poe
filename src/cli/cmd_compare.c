/* cmd_compare.c — `poe compare a.poeprofile b.poeprofile [...] [--top P]`
 *
 * Milestone M3: are the same experts used across workloads? All pairwise
 * comparisons of the given profiles — top-X% expert-set Jaccard, weighted
 * Jaccard of selection frequencies, Spearman rank correlation, JS
 * divergence, exclusive/cold expert counts, plus per-profile routing
 * summaries. Static: reads profiles, never touches the model.
 *
 * SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/profile.h"
#include "cli.h"

#define MAX_PROFILES 8

static const char *base_name(const char *path) {
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

static void profile_summary(const poe_profile *p, char tag, const char *path) {
    printf("  %c  %-36s %-10s %8llu tok   %s\n", tag, base_name(path),
           p->arch[0] ? p->arch : "?", (unsigned long long)p->tokens,
           p->fingerprint);
}

static void pair_detail(char ta, char tb, const poe_profile_cmp *c,
                        uint32_t n_layers) {
    printf("\n%c <-> %c\n", ta, tb);
    printf("  top-set Jaccard     mean %.3f   min %.3f (layer %u)   max %.3f (layer %u)\n",
           c->jaccard_mean, c->jaccard_min, c->jaccard_min_layer,
           c->jaccard_max, c->jaccard_max_layer);
    printf("  weighted Jaccard    %.3f\n", c->wjaccard_mean);
    printf("  Spearman            %.3f\n", c->spearman_mean);
    printf("  JS divergence       %.3f bits\n", c->jsd_bits_mean);
    printf("  exclusive experts   %llu in %c only, %llu in %c only (top sets, all layers)\n",
           (unsigned long long)c->exclusive_a, ta,
           (unsigned long long)c->exclusive_b, tb);
    printf("  cold in both        %llu expert slots of %u layers\n",
           (unsigned long long)c->cold_both, n_layers);
}

int poe_cmd_compare(int argc, char **argv) {
    const char *paths[MAX_PROFILES];
    int n = 0, json = 0;
    double top = 0.25;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json = 1;
        else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc) {
            top = atof(argv[++i]);
            if (top > 1.0) top /= 100.0;               /* accept 25 or 0.25 */
            if (top <= 0.0 || top > 1.0) {
                fprintf(stderr, "poe compare: bad --top value\n");
                return 2;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "poe compare: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (n < MAX_PROFILES) paths[n++] = argv[i];
        else {
            fprintf(stderr, "poe compare: at most %d profiles\n", MAX_PROFILES);
            return 2;
        }
    }
    if (n < 2) {
        fprintf(stderr,
                "usage: poe compare <a.poeprofile> <b.poeprofile> [...] "
                "[--top P] [--json]\n");
        return 2;
    }

    poe_profile *p[MAX_PROFILES] = { 0 };
    char err[256];
    for (int i = 0; i < n; i++) {
        if (poe_profile_load(&p[i], paths[i], err, sizeof err) != 0) {
            fprintf(stderr, "poe compare: %s: %s\n", paths[i], err);
            for (int k = 0; k < i; k++) poe_profile_free(p[k]);
            return 1;
        }
    }

    int rc = 0;
    for (int i = 1; i < n; i++) {
        if (p[i]->n_layers != p[0]->n_layers ||
            p[i]->n_experts != p[0]->n_experts) {
            fprintf(stderr, "poe compare: %s has a different shape "
                            "(%ux%u vs %ux%u)\n",
                    paths[i], p[i]->n_layers, p[i]->n_experts,
                    p[0]->n_layers, p[0]->n_experts);
            rc = 1;
            goto done;
        }
        if (strcmp(p[i]->fingerprint, p[0]->fingerprint) != 0)
            fprintf(stderr, "poe compare: warning: %s comes from a different "
                            "model/quantization than %s — comparison is "
                            "cross-model\n", paths[i], paths[0]);
    }

    if (json) {
        printf("{\n  \"top_fraction\": %.4f,\n  \"pairs\": [\n", top);
        int first = 1;
        for (int i = 0; i < n; i++) for (int k = i + 1; k < n; k++) {
            poe_profile_cmp c;
            if (poe_profile_compare(p[i], p[k], top, &c, NULL) != 0) continue;
            char ea[512], eb[512];
            poe_json_escape(paths[i], ea, sizeof ea);
            poe_json_escape(paths[k], eb, sizeof eb);
            printf("%s    {\"a\": \"%s\", \"b\": \"%s\",\n"
                   "     \"jaccard_mean\": %.6f, \"jaccard_min\": %.6f, "
                   "\"jaccard_min_layer\": %u,\n"
                   "     \"jaccard_max\": %.6f, \"jaccard_max_layer\": %u,\n"
                   "     \"weighted_jaccard\": %.6f, \"spearman\": %.6f, "
                   "\"jsd_bits\": %.6f,\n"
                   "     \"exclusive_a\": %" PRIu64 ", \"exclusive_b\": %" PRIu64
                   ", \"cold_both\": %" PRIu64 "}",
                   first ? "" : ",\n", ea, eb,
                   c.jaccard_mean, c.jaccard_min, c.jaccard_min_layer,
                   c.jaccard_max, c.jaccard_max_layer,
                   c.wjaccard_mean, c.spearman_mean, c.jsd_bits_mean,
                   c.exclusive_a, c.exclusive_b, c.cold_both);
            first = 0;
        }
        printf("\n  ]\n}\n");
        goto done;
    }

    printf("profiles   (top set = top %.0f%% experts by selection count)\n",
           top * 100.0);
    for (int i = 0; i < n; i++) profile_summary(p[i], (char)('A' + i), paths[i]);

    if (n > 2) {
        printf("\npairwise task overlap   (weighted Jaccard of selection frequencies)\n     ");
        for (int i = 0; i < n; i++) printf("      %c", 'A' + i);
        printf("\n");
        for (int i = 0; i < n; i++) {
            printf("  %c  ", 'A' + i);
            for (int k = 0; k < n; k++) {
                if (k == i) { printf("      -"); continue; }
                poe_profile_cmp c;
                poe_profile_compare(p[i], p[k], top, &c, NULL);
                printf("  %4.1f%%", c.wjaccard_mean * 100.0);
            }
            printf("\n");
        }
    }

    for (int i = 0; i < n; i++) for (int k = i + 1; k < n; k++) {
        poe_profile_cmp c;
        if (poe_profile_compare(p[i], p[k], top, &c, NULL) == 0)
            pair_detail((char)('A' + i), (char)('A' + k), &c, p[0]->n_layers);
    }

    /* per-profile routing character */
    printf("\nrouting summary\n");
    for (int i = 0; i < n; i++) {
        const poe_profile *q = p[i];
        uint32_t L = q->n_layers;
        double h0 = q->entropy_bits[0], hm = q->entropy_bits[L / 2],
               hl = q->entropy_bits[L - 1];
        uint64_t cold = 0;
        for (size_t s = 0; s < (size_t)L * q->n_experts; s++)
            if (q->sel_count[s] == 0) cold++;
        printf("  %c  entropy bits L0/mid/last  %.2f / %.2f / %.2f   "
               "mass-K@0.90 mid %.1f   cold %llu/%u\n",
               'A' + i, h0, hm, hl, q->mass_k[1][L / 2],
               (unsigned long long)cold, L * q->n_experts);
    }

done:
    for (int i = 0; i < n; i++) poe_profile_free(p[i]);
    return rc;
}
