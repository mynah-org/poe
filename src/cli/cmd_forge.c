/* cmd_forge.c — `poe forge <model.gguf> -o <out.gguf> ...`: the whole
 * training-free pipeline in one command.
 *
 * profile (optional, via the poe-profile subprocess) -> plan -> exact
 * estimate -> apply -> verify. Every intermediate artifact is written next
 * to the output (<out>.poeprofile / <out>.poeplan) so each stage stays
 * individually inspectable and reproducible; forge adds orchestration, not
 * new semantics.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "poe/apply.h"
#include "cli.h"

#define MAX_PROFILES 8

static int same_inode(const char *a, const char *b) {
    struct stat sa, sb;
    if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return 0;
    return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

/* Run poe-profile as a subprocess. The binary is found via --profiler,
 * $POE_PROFILER, or ./build/poe-profile, in that order. */
static int run_profiler(const char *profiler, const char *model,
                        const char *dataset, const char *method,
                        const char *out_profile) {
    if (profiler == NULL) profiler = getenv("POE_PROFILER");
    if (profiler == NULL) {
        struct stat st;
        if (stat("build/poe-profile", &st) == 0) profiler = "build/poe-profile";
    }
    if (profiler == NULL) {
        fprintf(stderr,
                "poe forge: no profiler available for --dataset.\n"
                "  Build it with:  make profiler LLAMA_DIR=<llama.cpp>\n"
                "  then pass --profiler <path> or set $POE_PROFILER,\n"
                "  or capture a profile yourself and pass --profile instead.\n");
        return -1;
    }
    const char *metric = strcmp(method, "reap") == 0 ? "reap" : "routing";
    char cmd[2048];
    int n = snprintf(cmd, sizeof cmd,
                     "'%s' '%s' --dataset '%s' --metric %s "
                     "--until-stable 0.99 -o '%s'",
                     profiler, model, dataset, metric, out_profile);
    if (n < 0 || (size_t)n >= sizeof cmd ||
        strchr(profiler, '\'') || strchr(model, '\'') ||
        strchr(dataset, '\'') || strchr(out_profile, '\'')) {
        fprintf(stderr, "poe forge: paths too long or contain quotes\n");
        return -1;
    }
    printf("── profile ──  %s\n", cmd);
    fflush(stdout);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "poe forge: profiling failed (exit %d)\n", rc);
        return -1;
    }
    return 0;
}

int poe_cmd_forge(int argc, char **argv) {
    const char *model_path = NULL, *out_path = NULL, *dataset = NULL;
    const char *profiler = NULL, *method = "reap";
    const char *ppaths[MAX_PROFILES];
    double      pweights[MAX_PROFILES];
    size_t      n_prof = 0;
    double      prune = 0.25;
    int         force = 0;
    long        top_k = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            if (n_prof == MAX_PROFILES) {
                fprintf(stderr, "poe forge: at most %d profiles\n", MAX_PROFILES);
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
        else if (strcmp(argv[i], "--dataset")  == 0 && i + 1 < argc) dataset  = argv[++i];
        else if (strcmp(argv[i], "--profiler") == 0 && i + 1 < argc) profiler = argv[++i];
        else if (strcmp(argv[i], "--method")   == 0 && i + 1 < argc) method   = argv[++i];
        else if (strcmp(argv[i], "--prune")    == 0 && i + 1 < argc) {
            prune = atof(argv[++i]);
            if (prune > 1.0) prune /= 100.0;          /* accept 25% or 0.25 */
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i == argc) {
                fprintf(stderr, "poe forge: %s needs a path\n", argv[i - 1]);
                return 2;
            }
            out_path = argv[i];
        }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atol(argv[++i]);
            if (top_k < 1) {
                fprintf(stderr, "poe forge: --top-k needs a positive count\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe forge: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (model_path == NULL) model_path = argv[i];
        else {
            fprintf(stderr, "poe forge: more than one model given\n");
            return 2;
        }
    }
    if (model_path == NULL || out_path == NULL ||
        (n_prof == 0 && dataset == NULL)) {
        fprintf(stderr,
            "usage: poe forge <model.gguf> -o <out.gguf>\n"
            "                 (--profile <p.poeprofile[:W]> ... | --dataset <text>)\n"
            "                 [--method reap|frequency|gate] [--prune P]\n"
            "                 [--top-k K] [--profiler <poe-profile>] [--force]\n");
        return 2;
    }
    if (same_inode(model_path, out_path)) {
        fprintf(stderr, "poe forge: refusing to overwrite the source model\n");
        return 2;
    }

    char prof_art[512], plan_art[512];
    snprintf(prof_art, sizeof prof_art, "%s.poeprofile", out_path);
    snprintf(plan_art, sizeof plan_art, "%s.poeplan", out_path);

    /* ── profile ────────────────────────────────────────────────────────── */
    if (n_prof == 0) {
        if (run_profiler(profiler, model_path, dataset, method, prof_art) != 0)
            return 1;
        ppaths[0] = prof_art;
        pweights[0] = 1.0;
        n_prof = 1;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe forge: %s\n", err);
        return 1;
    }

    int rc = 1;
    poe_plan *plan = NULL;
    poe_profile *profs[MAX_PROFILES] = { 0 };
    size_t loaded = 0;
    for (; loaded < n_prof; loaded++) {
        if (poe_profile_load(&profs[loaded], ppaths[loaded], err, sizeof err) != 0) {
            fprintf(stderr, "poe forge: %s: %s\n", ppaths[loaded], err);
            goto done;
        }
    }

    /* ── plan + exact estimate ──────────────────────────────────────────── */
    /* Same default as `poe plan`: forge is that pipeline in one command, so
     * it must not quietly ship a weaker guard. */
    const poe_plan_opts plan_opts = {
        .profiles = (const poe_profile *const *)profs, .weights = pweights,
        .n_profiles = n_prof, .method = method, .prune_frac = prune,
        .force = force, .protect_super = 1, .super_z = 0
    };
    if (poe_plan_build_opts(&plan, m, &plan_opts, err, sizeof err) != 0) {
        fprintf(stderr, "poe forge: %s\n", err);
        goto done;
    }
    if (poe_plan_write(plan, plan_art, err, sizeof err) != 0) {
        fprintf(stderr, "poe forge: %s\n", err);
        goto done;
    }

    char b1[32], b2[32];
    printf("── plan ──  %s\n", plan_art);
    printf("method    %s   prune %.1f%%   (%u profile%s)\n",
           plan->method, plan->prune_fraction * 100.0,
           (unsigned)n_prof, n_prof == 1 ? "" : "s");
    printf("experts   %u -> %u per layer across %u MoE layers\n",
           plan->n_experts, plan->keep_per_layer, m->n_moe_blocks);
    poe_format_bytes(plan->bytes_removed, b1, sizeof b1);
    poe_format_bytes(plan->bytes_before - plan->bytes_removed, b2, sizeof b2);
    printf("exact     remove %s  ->  %s on disk\n", b1, b2);
    for (uint32_t i = 0; i < plan->n_warnings; i++)
        printf("warning   %s\n", plan->warnings[i]);

    /* ── apply + verify ─────────────────────────────────────────────────── */
    printf("── apply ──  %s\n", out_path);
    fflush(stdout);
    poe_apply_stats st;
    if (poe_apply(m, plan, out_path, (uint32_t)top_k, force, &st,
                  err, sizeof err) != 0) {
        fprintf(stderr, "poe forge: %s\n", err);
        goto done;
    }
    printf("sliced    %u of %u tensors%s\n", st.tensors_sliced,
           st.tensors_total,
           st.expert_count_patched ? ", expert_count patched" : "");

    rc = poe_cli_verify_pruned(m, plan, out_path, force);
    if (rc == 0)
        printf("\nforged: %s  (%s, artifacts: %s%s%s)\n", out_path, b2,
               plan_art, dataset ? ", " : "", dataset ? prof_art : "");

done:
    poe_plan_free(plan);
    for (size_t i = 0; i < loaded; i++) poe_profile_free(profs[i]);
    poe_model_close(m);
    return rc;
}
