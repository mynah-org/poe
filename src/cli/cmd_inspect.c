/* cmd_inspect.c — `poe inspect model.gguf [--json]`
 *
 * Static MoE inspection: architecture, expert topology, exact storage split,
 * and the theoretical size of the checkpoint after removing 25/50/75% of the
 * routed experts. Everything here is exact byte accounting from the tensor
 * table — no estimates, no inference.
 *
 * SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "cli.h"

static void print_human(const poe_model *m, const char *path) {
    char b1[32], b2[32], b3[32];

    printf("model    %s\n", path);
    if (m->name[0]) printf("name     %s\n", m->name);
    printf("arch     %s   GGUF v%u, %u shard%s, %zu tensors\n",
           m->arch[0] ? m->arch : "(unknown)",
           ingot_gguf_version(m->g), ingot_gguf_shard_count(m->g),
           ingot_gguf_shard_count(m->g) == 1 ? "" : "s",
           ingot_gguf_count(m->g));
    printf("id       %s\n\n", m->fingerprint);

    if (m->n_moe_blocks == 0) {
        poe_format_bytes(m->total_bytes, b1, sizeof b1);
        printf("No MoE structure detected: no routed expert tensors in %u blocks.\n",
               m->n_blocks);
        printf("\nStorage\n  total                 %s\n", b1);
        return;
    }

    printf("MoE architecture\n");
    printf("  blocks                %u   (%u MoE)\n", m->n_blocks, m->n_moe_blocks);
    printf("  experts/block         %u\n", m->expert_count);
    if (m->experts_per_token)
        printf("  active/token          %u\n", m->experts_per_token);
    if (m->shared_expert_count)
        printf("  shared experts        %u\n", m->shared_expert_count);
    if (m->embedding_length)
        printf("  embedding length      %u\n", m->embedding_length);
    if (m->expert_ff_length)
        printf("  expert FF length      %u\n", m->expert_ff_length);

    /* Expert weight quantization: report the packed tensor's type from the
     * first MoE block (mixed-type models exist; the per-tensor truth is in
     * `poe experts`). */
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (blk->gate_exps_w) {
            printf("  expert quantization   %s\n",
                   ingot_type_name(blk->gate_exps_w->type));
            /* a split checkpoint carries two precisions per layer; saying so
             * here keeps "expert quantization" from being a half-truth */
            if (blk->is_split && blk->gate_exps_cold_w)
                printf("  split experts         %u hot at %s, %u cold at %s\n",
                       blk->hot_expert_count,
                       ingot_type_name(blk->gate_exps_w->type),
                       blk->cold_expert_count,
                       ingot_type_name(blk->gate_exps_cold_w->type));
            break;
        }
    }

    printf("\nStorage\n");
    poe_format_bytes(m->expert_bytes, b1, sizeof b1);
    printf("  experts               %-11s (%.1f%%)\n", b1,
           m->total_bytes ? 100.0 * (double)m->expert_bytes / (double)m->total_bytes : 0.0);
    poe_format_bytes(m->router_bytes, b1, sizeof b1);
    printf("  routers               %s\n", b1);
    if (m->shared_bytes) {
        poe_format_bytes(m->shared_bytes, b1, sizeof b1);
        printf("  shared experts        %s\n", b1);
    }
    poe_format_bytes(m->other_bytes, b1, sizeof b1);
    printf("  other                 %s\n", b1);
    poe_format_bytes(m->total_bytes, b1, sizeof b1);
    printf("  total                 %s\n", b1);

    printf("\nTheoretical expert removal   (routed expert weights only, exact)\n");
    static const unsigned pct[] = { 25, 50, 75 };
    for (size_t i = 0; i < sizeof pct / sizeof pct[0]; i++) {
        uint64_t removed = m->expert_bytes * pct[i] / 100;
        uint32_t n_rm    = m->expert_count * pct[i] / 100;
        poe_format_bytes(removed, b2, sizeof b2);
        poe_format_bytes(m->total_bytes - removed, b3, sizeof b3);
        printf("  -%u%%   remove %3u/%u per block   save %-11s ->  %s\n",
               pct[i], n_rm, m->expert_count, b2, b3);
    }
}

static void print_json(const poe_model *m, const char *path) {
    char esc[512];

    printf("{\n");
    poe_json_escape(path, esc, sizeof esc);
    printf("  \"model\": \"%s\",\n", esc);
    poe_json_escape(m->name, esc, sizeof esc);
    printf("  \"name\": \"%s\",\n", esc);
    poe_json_escape(m->arch, esc, sizeof esc);
    printf("  \"arch\": \"%s\",\n", esc);
    printf("  \"fingerprint\": \"%s\",\n", m->fingerprint);
    printf("  \"gguf_version\": %u,\n", ingot_gguf_version(m->g));
    printf("  \"shards\": %u,\n", ingot_gguf_shard_count(m->g));
    printf("  \"tensors\": %zu,\n", ingot_gguf_count(m->g));
    printf("  \"blocks\": %u,\n", m->n_blocks);
    printf("  \"moe_blocks\": %u,\n", m->n_moe_blocks);
    printf("  \"expert_count\": %u,\n", m->expert_count);
    printf("  \"experts_per_token\": %u,\n", m->experts_per_token);
    printf("  \"shared_expert_count\": %u,\n", m->shared_expert_count);
    printf("  \"embedding_length\": %u,\n", m->embedding_length);
    printf("  \"expert_ff_length\": %u,\n", m->expert_ff_length);
    printf("  \"bytes\": {\n");
    printf("    \"experts\": %" PRIu64 ",\n", m->expert_bytes);
    printf("    \"routers\": %" PRIu64 ",\n", m->router_bytes);
    printf("    \"shared_experts\": %" PRIu64 ",\n", m->shared_bytes);
    printf("    \"other\": %" PRIu64 ",\n", m->other_bytes);
    printf("    \"total\": %" PRIu64 "\n", m->total_bytes);
    printf("  }\n");
    printf("}\n");
}

int poe_cmd_inspect(int argc, char **argv) {
    const char *path = NULL;
    int json = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe inspect: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (path == NULL) path = argv[i];
        else {
            fprintf(stderr, "poe inspect: more than one model given\n");
            return 2;
        }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: poe inspect <model.gguf> [--json]\n");
        return 2;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, path, err, sizeof err) != 0) {
        fprintf(stderr, "poe inspect: %s\n", err);
        return 1;
    }

    if (json) print_json(m, path);
    else      print_human(m, path);

    poe_model_close(m);
    return 0;
}
