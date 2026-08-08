/* cmd_apply.c — `poe apply <plan.poeplan> <model.gguf> -o <out.gguf>`
 *
 * Materializes a plan: slices the expert tensors, compacts the router,
 * patches expert_count, streams the result to a new GGUF, then reopens the
 * output and verifies structure and exact byte accounting against the plan.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "poe/apply.h"
#include "cli.h"

static int same_inode(const char *a, const char *b) {
    struct stat sa, sb;
    if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return 0;
    return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

/* Reopen a freshly written pruned checkpoint and verify structure, expert
 * count and exact byte accounting against the plan. Shared with forge.
 * Prints its findings; returns 0 when everything checks out. */
int poe_cli_verify_pruned(const poe_model *m, const poe_plan *p,
                          const char *out_path, int force) {
    char err[256];
    const uint64_t want = p->bytes_before - p->bytes_removed;
    poe_model *o;
    if (poe_model_open(&o, out_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe: output does not reopen: %s\n", err);
        return 1;
    }
    int rc = 0;
    printf("\nVerification\n");
    if (o->n_blocks != m->n_blocks || o->n_moe_blocks != m->n_moe_blocks) {
        printf("  BLOCK MISMATCH: %u blocks (%u MoE), expected %u (%u)\n",
               o->n_blocks, o->n_moe_blocks, m->n_blocks, m->n_moe_blocks);
        rc = 1;
    }
    if (o->expert_count != p->keep_per_layer) {
        printf("  EXPERT COUNT MISMATCH: output has %u, plan keeps %u\n",
               o->expert_count, p->keep_per_layer);
        rc = 1;
    }
    if (!force && o->total_bytes != want) {
        printf("  SIZE MISMATCH: output payload %llu bytes, plan expects %llu\n",
               (unsigned long long)o->total_bytes, (unsigned long long)want);
        rc = 1;
    }
    if (rc == 0)
        printf("  ok: %u blocks, %u experts per layer, exact bytes as planned\n",
               o->n_blocks, o->expert_count);
    printf("  output fingerprint %s\n", o->fingerprint);
    poe_model_close(o);
    return rc;
}

int poe_cmd_apply(int argc, char **argv) {
    const char *plan_path = NULL, *model_path = NULL, *out_path = NULL;
    int force = 0;
    long top_k = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atol(argv[++i]);
            if (top_k < 1) {
                fprintf(stderr, "poe apply: --top-k needs a positive count\n");
                return 2;
            }
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i == argc) {
                fprintf(stderr, "poe apply: %s needs a path\n", argv[i - 1]);
                return 2;
            }
            out_path = argv[i];
        } else if (strcmp(argv[i], "--force") == 0) {
            force = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "poe apply: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (plan_path == NULL) plan_path = argv[i];
        else if (model_path == NULL) model_path = argv[i];
        else {
            fprintf(stderr, "poe apply: too many arguments\n");
            return 2;
        }
    }
    if (plan_path == NULL || model_path == NULL || out_path == NULL) {
        fprintf(stderr, "usage: poe apply <plan.poeplan> <model.gguf> "
                        "-o <out.gguf> [--top-k K] [--force]\n");
        return 2;
    }
    if (same_inode(model_path, out_path)) {
        fprintf(stderr, "poe apply: refusing to overwrite the source model\n");
        return 2;
    }

    char err[256];
    poe_plan *p;
    if (poe_plan_load(&p, plan_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe apply: %s\n", err);
        return 1;
    }
    poe_model *m;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe apply: %s\n", err);
        poe_plan_free(p);
        return 1;
    }

    printf("plan      %s   (%s, prune %.1f%%)\n",
           plan_path, p->method, p->prune_fraction * 100.0);
    printf("source    %s   %s   (%s)\n", model_path, m->fingerprint, m->arch);
    printf("experts   %u -> %u per layer across %u MoE layers\n",
           p->n_experts, p->keep_per_layer, m->n_moe_blocks);
    if (top_k)
        printf("top-k     %u -> %ld active per token\n",
               m->experts_per_token, top_k);

    poe_apply_stats st;
    if (poe_apply(m, p, out_path, (uint32_t)top_k, force, &st,
                  err, sizeof err) != 0) {
        fprintf(stderr, "poe apply: %s\n", err);
        poe_model_close(m);
        poe_plan_free(p);
        return 1;
    }

    char b1[32], b2[32];
    poe_format_bytes(m->total_bytes, b1, sizeof b1);
    poe_format_bytes(st.payload_bytes, b2, sizeof b2);
    printf("output    %s\n", out_path);
    printf("  sliced  %u of %u tensors%s%s%s\n", st.tensors_sliced,
           st.tensors_total,
           st.expert_count_patched ? ", expert_count patched" : "",
           st.top_k_patched ? ", expert_used_count patched" : "",
           st.kv_dropped ? ", stale poe.* metadata replaced" : "");
    printf("  bytes   %s -> %s   %.1f%% of original\n", b1, b2,
           m->total_bytes
               ? 100.0 * (double)st.payload_bytes / (double)m->total_bytes
               : 0.0);

    int rc = poe_cli_verify_pruned(m, p, out_path, force);
    poe_model_close(m);
    poe_plan_free(p);
    return rc;
}
