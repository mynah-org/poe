/* cmd_experts.c — `poe experts model.gguf [--layer N] [--json]`
 *
 * Per-block expert tensor mapping: which tensors hold the routed experts,
 * their type, shape, exact size, and the per-expert slice size. This is the
 * ground truth `poe apply` will rewrite, so it is worth being able to stare
 * at it.
 *
 * SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli.h"

static void print_tensor_line(const poe_block *blk, const ingot_tensor *t) {
    if (t == NULL) return;

    char size[32], per[32], shape[128];
    poe_format_bytes(t->nbytes, size, sizeof size);

    uint64_t rm[INGOT_MAX_RANK];
    ingot_gguf_shape_row_major(t, rm);
    size_t o = 0;
    for (uint32_t d = 0; d < t->rank && o + 24 < sizeof shape; d++)
        o += (size_t)snprintf(shape + o, sizeof shape - o, "%s%" PRIu64,
                              d ? " x " : "", rm[d]);

    /* The name after "blk.N." reads better in a table. */
    const char *short_name = strchr(t->name + 4, '.');
    short_name = short_name ? short_name + 1 : t->name;

    printf("  %-24s %-6s [%s]   %s", short_name,
           ingot_type_name(t->type), shape, size);
    if (blk->expert_count > 1) {
        poe_format_bytes(t->nbytes / blk->expert_count, per, sizeof per);
        printf("   (%s/expert)", per);
    }
    printf("\n");
}

static void print_block_human(const poe_model *m, const poe_block *blk) {
    char b1[32];

    printf("blk.%-3u  %u experts", blk->block, blk->expert_count);
    if (m->experts_per_token) printf("   top-k %u", m->experts_per_token);
    if (blk->has_shared_expert) printf("   +shared");
    if (blk->is_legacy_split) printf("   (legacy split tensors)");
    printf("\n");

    print_tensor_line(blk, blk->router_w);
    print_tensor_line(blk, blk->router_b);
    print_tensor_line(blk, blk->gate_exps_w);
    print_tensor_line(blk, blk->gate_exps_b);
    print_tensor_line(blk, blk->up_exps_w);
    print_tensor_line(blk, blk->up_exps_b);
    print_tensor_line(blk, blk->down_exps_w);
    print_tensor_line(blk, blk->down_exps_b);

    poe_format_bytes(blk->expert_bytes, b1, sizeof b1);
    printf("  %-24s %s\n\n", "expert bytes", b1);
}

static void print_block_json(const poe_model *m, const poe_block *blk, int first) {
    (void)m;
    printf("%s    {\"block\": %u, \"experts\": %u, \"legacy_split\": %s,\n",
           first ? "" : ",\n", blk->block, blk->expert_count,
           blk->is_legacy_split ? "true" : "false");
    printf("     \"shared_expert\": %s,\n", blk->has_shared_expert ? "true" : "false");
    printf("     \"expert_bytes\": %" PRIu64 ", \"router_bytes\": %" PRIu64
           ", \"shared_bytes\": %" PRIu64 ",\n",
           blk->expert_bytes, blk->router_bytes, blk->shared_bytes);
    printf("     \"tensors\": [");
    const ingot_tensor *ts[] = {
        blk->router_w, blk->router_b,
        blk->gate_exps_w, blk->gate_exps_b,
        blk->up_exps_w, blk->up_exps_b,
        blk->down_exps_w, blk->down_exps_b,
    };
    int emitted = 0;
    for (size_t i = 0; i < sizeof ts / sizeof ts[0]; i++) {
        if (ts[i] == NULL) continue;
        char esc[256];
        poe_json_escape(ts[i]->name, esc, sizeof esc);
        printf("%s\n      {\"name\": \"%s\", \"type\": \"%s\", \"bytes\": %" PRIu64 "}",
               emitted ? "," : "", esc, ingot_type_name(ts[i]->type), ts[i]->nbytes);
        emitted = 1;
    }
    printf("\n     ]}");
}

int poe_cmd_experts(int argc, char **argv) {
    const char *path = NULL;
    long layer = -1;
    int json = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json = 1;
        else if (strcmp(argv[i], "--layer") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "poe experts: --layer needs a value\n");
                return 2;
            }
            layer = strtol(argv[++i], NULL, 10);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "poe experts: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (path == NULL) path = argv[i];
        else {
            fprintf(stderr, "poe experts: more than one model given\n");
            return 2;
        }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: poe experts <model.gguf> [--layer N] [--json]\n");
        return 2;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, path, err, sizeof err) != 0) {
        fprintf(stderr, "poe experts: %s\n", err);
        return 1;
    }

    if (layer >= 0 && (uint64_t)layer >= m->n_blocks) {
        fprintf(stderr, "poe experts: layer %ld out of range (model has %u blocks)\n",
                layer, m->n_blocks);
        poe_model_close(m);
        return 1;
    }
    if (m->n_moe_blocks == 0) {
        fprintf(stderr, "poe experts: no MoE structure in this model\n");
        poe_model_close(m);
        return 1;
    }

    if (json) printf("{\n  \"blocks\": [\n");
    int first = 1;
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (!blk->is_moe) continue;
        if (layer >= 0 && blk->block != (uint32_t)layer) continue;
        if (json) print_block_json(m, blk, first);
        else      print_block_human(m, blk);
        first = 0;
    }
    if (json) printf("\n  ]\n}\n");

    poe_model_close(m);
    return 0;
}
