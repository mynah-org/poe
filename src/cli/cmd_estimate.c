/* cmd_estimate.c — `poe estimate plan.poeplan [model.gguf]`
 *
 * Reports a plan's exact accounting; with the source model given, verifies
 * the fingerprint and recomputes the removal from the checkpoint so plan
 * and model cannot silently drift apart. Exact numbers only — runtime
 * VRAM/RAM estimation is a later milestone and will be labeled as such.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <string.h>

#include "poe/plan.h"
#include "cli.h"

int poe_cmd_estimate(int argc, char **argv) {
    const char *plan_path = NULL, *model_path = NULL;

    for (int i = 0; i < argc; i++) {
        if (argv[i][0] == '-') {
            fprintf(stderr, "poe estimate: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (plan_path == NULL) plan_path = argv[i];
        else if (model_path == NULL) model_path = argv[i];
        else {
            fprintf(stderr, "poe estimate: too many arguments\n");
            return 2;
        }
    }
    if (plan_path == NULL) {
        fprintf(stderr, "usage: poe estimate <plan.poeplan> [model.gguf]\n");
        return 2;
    }

    char err[256];
    poe_plan *p;
    if (poe_plan_load(&p, plan_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe estimate: %s\n", err);
        return 1;
    }

    char b1[32], b2[32], b3[32], p1[32], p2[32];
    poe_format_bytes(p->bytes_before, b1, sizeof b1);
    poe_format_bytes(p->bytes_removed, b2, sizeof b2);
    poe_format_bytes(p->bytes_before - p->bytes_removed, b3, sizeof b3);
    poe_format_params(p->params_before, p1, sizeof p1);
    poe_format_params(p->params_before - p->params_removed, p2, sizeof p2);

    printf("plan      %s\n", plan_path);
    printf("source    %s   (%s)\n", p->model_fingerprint, p->arch);
    printf("method    %s   prune %.1f%%   keep %u/%u per layer\n",
           p->method, p->prune_fraction * 100.0, p->keep_per_layer, p->n_experts);
    printf("\nExact accounting (checkpoint bytes)\n");
    printf("  before    %s   (%s params)\n", b1, p1);
    printf("  removed   %s\n", b2);
    printf("  after     %s   (%s params)   %.1f%% of original\n", b3, p2,
           p->bytes_before
               ? 100.0 * (double)(p->bytes_before - p->bytes_removed) /
                 (double)p->bytes_before : 0.0);

    int rc = 0;
    if (model_path) {
        poe_model *m;
        if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
            fprintf(stderr, "poe estimate: %s\n", err);
            poe_plan_free(p);
            return 1;
        }
        printf("\nVerification against %s\n", model_path);
        if (strcmp(m->fingerprint, p->model_fingerprint) != 0) {
            printf("  FINGERPRINT MISMATCH: model is %s\n", m->fingerprint);
            rc = 1;
        } else if (m->total_bytes != p->bytes_before) {
            printf("  SIZE MISMATCH: model has %llu bytes, plan says %llu\n",
                   (unsigned long long)m->total_bytes,
                   (unsigned long long)p->bytes_before);
            rc = 1;
        } else {
            /* recompute the removal straight from the checkpoint */
            uint64_t removed = 0;
            const uint32_t E = p->n_experts;
            for (uint32_t l = 0; l < p->n_layers && l < m->n_blocks; l++) {
                const poe_block *blk = &m->blocks[l];
                if (!blk->is_moe || blk->expert_count != E) continue;
                uint64_t pruned = 0;
                for (uint32_t e = 0; e < E; e++)
                    if (!p->keep[(size_t)l * E + e]) pruned++;
                uint64_t row_b = blk->router_w ? blk->router_w->nbytes / E : 0;
                removed += pruned * (blk->expert_bytes / E + row_b);
            }
            if (removed == p->bytes_removed)
                printf("  ok: fingerprint and byte accounting verified\n");
            else {
                printf("  ACCOUNTING MISMATCH: recomputed %llu, plan says %llu\n",
                       (unsigned long long)removed,
                       (unsigned long long)p->bytes_removed);
                rc = 1;
            }
        }
        poe_model_close(m);
    }

    printf("\nnote: runtime VRAM/RAM are not estimated yet; the numbers above\n");
    printf("      are exact on-disk accounting from the tensor table.\n");
    poe_plan_free(p);
    return rc;
}
