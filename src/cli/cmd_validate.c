/* cmd_validate.c — `poe validate <model.gguf> [--plan <p.poeplan>]`
 *
 * Structural validation, no inference: MoE metadata vs tensor shapes,
 * per-block consistency, slab divisibility (exact accounting possible),
 * quantization geometry known to ingot. With --plan it also checks the
 * provenance chain: given the source model it verifies the plan binds to
 * it; given a pruned output it verifies poe.source_fingerprint, the kept
 * expert count and the exact byte accounting all match the plan.
 *
 * The smoke/behavioral levels need an inference backend and live outside
 * this binary: llama.cpp for generation, tools/coding_eval.py for the
 * pruned-vs-full win/fail comparison.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <string.h>

#include "poe/plan.h"
#include "cli.h"

static int fails;

static void res(int ok, const char *what) {
    printf("  %s %s\n", ok ? "ok:  " : "FAIL:", what);
    if (!ok) fails++;
}

static int last_dim_is(const ingot_tensor *t, uint64_t v) {
    return t == NULL || (t->rank > 0 && t->ne[t->rank - 1] == v);
}

static int divides(const ingot_tensor *t, uint64_t e) {
    return t == NULL || (e && t->nbytes % e == 0);
}

int poe_cmd_validate(int argc, char **argv) {
    const char *model_path = NULL, *plan_path = NULL;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--plan") == 0 && i + 1 < argc)
            plan_path = argv[++i];
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe validate: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (model_path == NULL) model_path = argv[i];
        else {
            fprintf(stderr, "poe validate: too many arguments\n");
            return 2;
        }
    }
    if (model_path == NULL) {
        fprintf(stderr, "usage: poe validate <model.gguf> [--plan <p.poeplan>]\n");
        return 2;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe validate: %s\n", err);
        return 1;
    }

    char b1[32];
    poe_format_bytes(m->total_bytes, b1, sizeof b1);
    printf("model     %s   %s   (%s, %s)\n", model_path, m->fingerprint,
           m->arch, b1);
    fails = 0;

    /* ── structural ─────────────────────────────────────────────────────── */
    printf("\nStructural\n");
    const uint32_t E = m->expert_count;
    res(m->n_moe_blocks > 0, "model has MoE structure");
    if (m->n_moe_blocks == 0) goto plan_check;   /* nothing else applies */

    int uniform = 1, packed = 1, shapes = 1, router = 1, divis = 1, legacy = 0;
    for (uint32_t l = 0; l < m->n_blocks; l++) {
        const poe_block *blk = &m->blocks[l];
        if (!blk->is_moe) continue;
        if (blk->is_legacy_split) { legacy = 1; continue; }
        if (blk->expert_count != E) uniform = 0;
        if (!blk->gate_exps_w || !blk->up_exps_w || !blk->down_exps_w)
            packed = 0;
        if (!blk->router_w) router = 0;

        const ingot_tensor *slabs[8] = {
            blk->router_w, blk->router_b,
            blk->gate_exps_w, blk->up_exps_w, blk->down_exps_w,
            blk->gate_exps_b, blk->up_exps_b, blk->down_exps_b
        };
        for (size_t i = 0; i < 8; i++) {
            if (!last_dim_is(slabs[i], E)) shapes = 0;
            if (!divides(slabs[i], E)) divis = 0;
        }
        if (m->embedding_length) {
            if (blk->gate_exps_w &&
                blk->gate_exps_w->ne[0] != m->embedding_length) shapes = 0;
            if (blk->router_w &&
                blk->router_w->ne[0] != m->embedding_length) shapes = 0;
            if (blk->down_exps_w && blk->down_exps_w->rank == 3 &&
                blk->down_exps_w->ne[1] != m->embedding_length) shapes = 0;
        }
        if (m->expert_ff_length) {
            if (blk->gate_exps_w && blk->gate_exps_w->rank == 3 &&
                blk->gate_exps_w->ne[1] != m->expert_ff_length) shapes = 0;
            if (blk->down_exps_w &&
                blk->down_exps_w->ne[0] != m->expert_ff_length) shapes = 0;
        }
    }
    char line[128];
    snprintf(line, sizeof line, "uniform expert_count %u across %u MoE blocks",
             E, m->n_moe_blocks);
    res(uniform, line);
    res(!legacy, "packed expert layout (no legacy per-expert tensors)");
    res(packed, "gate/up/down packed expert tensors present in every MoE block");
    res(router, "router weights present in every MoE block");
    res(shapes, "tensor shapes consistent with embd/ff/expert metadata");
    res(divis, "expert slabs slice evenly (exact byte accounting possible)");
    snprintf(line, sizeof line, "expert_used_count %u within [1, %u]",
             m->experts_per_token, E);
    res(m->experts_per_token >= 1 && m->experts_per_token <= E, line);

    {
        int geom_ok = 1;
        const char *tname = "?";
        for (uint32_t l = 0; l < m->n_blocks && geom_ok; l++) {
            const poe_block *blk = &m->blocks[l];
            if (!blk->is_moe || !blk->gate_exps_w) continue;
            uint64_t be, bb;
            tname = ingot_type_name(blk->gate_exps_w->type);
            if (ingot_type_geometry(blk->gate_exps_w->type, &be, &bb) != 0)
                geom_ok = 0;
        }
        snprintf(line, sizeof line,
                 "expert quantization %s has known geometry", tname);
        res(geom_ok, line);
    }

    /* ── plan cross-check ───────────────────────────────────────────────── */
plan_check:
    if (plan_path) {
        printf("\nPlan cross-check (%s)\n", plan_path);
        poe_plan *p;
        if (poe_plan_load(&p, plan_path, err, sizeof err) != 0) {
            fprintf(stderr, "poe validate: %s\n", err);
            poe_model_close(m);
            return 1;
        }
        const char *srcfp = NULL;
        const ingot_kv *kv = ingot_gguf_kv_find(m->g, "poe.source_fingerprint");
        if (kv) ingot_kv_str(kv, &srcfp);

        if (strcmp(m->fingerprint, p->model_fingerprint) == 0) {
            printf("  role: source model of this plan\n");
            res(m->total_bytes == p->bytes_before,
                "plan bytes_before matches the checkpoint exactly");
            res(m->n_blocks == p->n_layers && E == p->n_experts,
                "plan topology matches the checkpoint");
        } else if (srcfp && strcmp(srcfp, p->model_fingerprint) == 0) {
            printf("  role: pruned output of this plan (provenance verified)\n");
            snprintf(line, sizeof line, "expert_count %u equals plan keep %u",
                     E, p->keep_per_layer);
            res(E == p->keep_per_layer, line);
            res(m->total_bytes == p->bytes_before - p->bytes_removed,
                "payload bytes equal the plan's exact bytes_after");
            const char *method = NULL;
            kv = ingot_gguf_kv_find(m->g, "poe.method");
            if (kv) ingot_kv_str(kv, &method);
            res(method && strcmp(method, p->method) == 0,
                "provenance method matches the plan");
        } else {
            res(0, "model is neither the plan's source nor its pruned output");
        }
        poe_plan_free(p);
    }

    printf("\n%s\n", fails ? "INVALID" : "valid");
    poe_model_close(m);
    return fails ? 1 : 0;
}
