/* cmd_plan.c — `poe plan model.gguf --profile p.poeprofile[:W] ...
 *               [--method reap|frequency|gate] [--prune P] [-o out.poeplan]
 *               [--force]`
 *
 * Milestone M6: turn profiles + parameters into a deterministic,
 * persistent expert-removal plan with exact byte accounting. No checkpoint
 * mutation — `poe apply` (M7) consumes the plan.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/plan.h"
#include "cli.h"

#define MAX_PROFILES 8

int poe_cmd_plan(int argc, char **argv) {
    const char *model_path = NULL, *out_path = NULL;
    const char *method = "reap";
    const char *ppaths[MAX_PROFILES];
    double      pweights[MAX_PROFILES];
    size_t      n_prof = 0;
    double      prune = 0.25;
    int         force = 0;
    int         protect_super = 1;    /* M10: on by default, see docs */
    double      super_z = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            if (n_prof == MAX_PROFILES) {
                fprintf(stderr, "poe plan: at most %d profiles\n", MAX_PROFILES);
                return 2;
            }
            char *arg = argv[++i];
            char *colon = strrchr(arg, ':');
            double w = 1.0;
            if (colon && colon[1] && strspn(colon + 1, "0123456789.") ==
                strlen(colon + 1)) {
                w = atof(colon + 1);
                *colon = '\0';
            }
            ppaths[n_prof] = arg;
            pweights[n_prof++] = w > 0.0 ? w : 1.0;
        }
        else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) method = argv[++i];
        else if (strcmp(argv[i], "--prune") == 0 && i + 1 < argc) {
            prune = atof(argv[++i]);
            if (prune > 1.0) prune /= 100.0;          /* accept 25% or 0.25 */
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (strcmp(argv[i], "--protect-super-experts") == 0) protect_super = 1;
        else if (strcmp(argv[i], "--no-protect-super-experts") == 0) protect_super = 0;
        else if (strcmp(argv[i], "--super-z") == 0 && i + 1 < argc)
            super_z = atof(argv[++i]);
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe plan: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (model_path == NULL) model_path = argv[i];
        else {
            fprintf(stderr, "poe plan: more than one model given\n");
            return 2;
        }
    }
    if (model_path == NULL || n_prof == 0) {
        fprintf(stderr, "usage: poe plan <model.gguf> --profile <p.poeprofile[:W]> ...\n"
                        "               [--method reap|frequency|gate] [--prune P]\n"
                        "               [-o out.poeplan] [--force]\n"
                        "               [--no-protect-super-experts] [--super-z Z]\n"
                        "\n"
                        "  activation-magnitude outliers are lifted out of the cut by\n"
                        "  default and the next candidates take their place, so the\n"
                        "  size target is unchanged (M10, docs/super-experts.md)\n");
        return 2;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe plan: %s\n", err);
        return 1;
    }

    poe_profile *profs[MAX_PROFILES] = { 0 };
    int rc = 1;
    size_t loaded = 0;
    for (; loaded < n_prof; loaded++) {
        if (poe_profile_load(&profs[loaded], ppaths[loaded], err, sizeof err) != 0) {
            fprintf(stderr, "poe plan: %s: %s\n", ppaths[loaded], err);
            goto done;
        }
    }

    poe_plan *plan;
    const poe_plan_opts opts = {
        .profiles = (const poe_profile *const *)profs, .weights = pweights,
        .n_profiles = n_prof, .method = method, .prune_frac = prune,
        .force = force, .protect_super = protect_super, .super_z = super_z
    };
    if (poe_plan_build_opts(&plan, m, &opts, err, sizeof err) != 0) {
        fprintf(stderr, "poe plan: %s\n", err);
        goto done;
    }

    char defout[512];
    if (out_path == NULL) {
        snprintf(defout, sizeof defout, "%s-%.0f.poeplan",
                 method, plan->prune_fraction * 100.0);
        out_path = defout;
    }
    if (poe_plan_write(plan, out_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe plan: %s\n", err);
        poe_plan_free(plan);
        goto done;
    }

    char b1[32], b2[32], p1[32];
    printf("plan      %s\n", out_path);
    printf("method    %s   prune %.1f%%   (%u profile%s)\n",
           plan->method, plan->prune_fraction * 100.0,
           (unsigned)n_prof, n_prof == 1 ? "" : "s");
    printf("experts   keep %u / %u per layer   (%u layers, top-k %u)\n",
           plan->keep_per_layer, plan->n_experts, plan->n_layers, plan->top_k);
    poe_format_bytes(plan->bytes_removed, b1, sizeof b1);
    poe_format_bytes(plan->bytes_before - plan->bytes_removed, b2, sizeof b2);
    poe_format_params(plan->params_before - plan->params_removed, p1, sizeof p1);
    printf("exact     remove %s  ->  %s on disk, %s params\n", b1, b2, p1);
    if (plan->protect_super) {
        printf("super     %u activation outliers flagged (%u of them rarely "
               "selected)\n", plan->n_super_flagged, plan->n_super_rare);
        printf("          %u rescued from this cut%s\n", plan->n_super_rescued,
               plan->n_super_rescued ? "" :
               "  — the ranking was already keeping them");
    }
    for (uint32_t i = 0; i < plan->n_warnings; i++)
        printf("warning   %s\n", plan->warnings[i]);

    poe_plan_free(plan);
    rc = 0;
done:
    for (size_t i = 0; i < loaded; i++) poe_profile_free(profs[i]);
    poe_model_close(m);
    return rc;
}
