/* test_model.c — poe_model_open() against synthetic fixtures.
 *
 * Checks, in order: the MoE fixture is discovered correctly (topology,
 * per-block tensor mapping, exact byte accounting derived from the fixture
 * geometry, never hardcoded); the fingerprint is deterministic, ignores the
 * path, and moves when the weights move; a dense model reports no MoE
 * structure; a missing file fails with a message instead of a crash.
 *
 * Runs with no model on disk, so it belongs in CI.
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <string.h>

#include "fixture.h"
#include "poe/poe.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

#define F32 4u

int main(void) {
    char err[256];

    const char *moe_path   = "build/fixture-moe.gguf";
    const char *moe_path2  = "build/fixture-moe-copy.gguf";
    const char *moe_path3  = "build/fixture-moe-seed9.gguf";
    const char *dense_path = "build/fixture-dense.gguf";

    printf("fixtures\n");
    CHECK(poe_fixture_moe(moe_path, 1, err, sizeof err) == 0, "write moe fixture");
    CHECK(poe_fixture_moe(moe_path2, 1, err, sizeof err) == 0, "write moe fixture copy");
    CHECK(poe_fixture_moe(moe_path3, 9, err, sizeof err) == 0, "write moe fixture seed 9");
    CHECK(poe_fixture_dense(dense_path, err, sizeof err) == 0, "write dense fixture");

    /* ── MoE discovery ─────────────────────────────────────────────────── */
    printf("moe fixture\n");
    poe_model *m;
    err[0] = '\0';
    CHECK(poe_model_open(&m, moe_path, err, sizeof err) == 0, "open moe fixture%s%s",
          err[0] ? ": " : "", err);
    if (m == NULL) { printf("cannot continue\n"); return 1; }

    CHECK(strcmp(m->arch, POE_FIX_ARCH) == 0, "arch %s", m->arch);
    CHECK(m->n_blocks == POE_FIX_BLOCKS, "blocks %u", m->n_blocks);
    CHECK(m->n_moe_blocks == POE_FIX_BLOCKS, "moe blocks %u", m->n_moe_blocks);
    CHECK(m->expert_count == POE_FIX_EXPERTS, "experts %u", m->expert_count);
    CHECK(m->experts_per_token == POE_FIX_TOPK, "top-k %u", m->experts_per_token);
    CHECK(m->expert_ff_length == POE_FIX_FF, "expert ff %u", m->expert_ff_length);
    CHECK(m->embedding_length == POE_FIX_EMBD, "embd %u", m->embedding_length);

    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (!(blk->is_moe && blk->expert_count == POE_FIX_EXPERTS &&
              blk->router_w && blk->gate_exps_w && blk->up_exps_w &&
              blk->down_exps_w && !blk->is_legacy_split && !blk->has_shared_expert)) {
            CHECK(0, "block %u mapping", b);
        }
    }
    CHECK(1, "all %u blocks mapped (router + gate/up/down exps)", m->n_blocks);

    /* Exact accounting, derived from the geometry. */
    const uint64_t per_blk_experts =
        (uint64_t)F32 * POE_FIX_EXPERTS * POE_FIX_EMBD * POE_FIX_FF * 3;  /* gate+up+down */
    const uint64_t per_blk_router = (uint64_t)F32 * POE_FIX_EMBD * POE_FIX_EXPERTS;
    const uint64_t per_blk_other  =
        (uint64_t)F32 * (POE_FIX_EMBD * 2                 /* attn_norm + ffn_norm */
                         + POE_FIX_EMBD * POE_FIX_EMBD * 4);  /* q k v o */
    const uint64_t global_other =
        (uint64_t)F32 * (POE_FIX_EMBD * POE_FIX_VOCAB + POE_FIX_EMBD);

    CHECK(m->expert_bytes == per_blk_experts * POE_FIX_BLOCKS,
          "expert bytes %llu", (unsigned long long)m->expert_bytes);
    CHECK(m->router_bytes == per_blk_router * POE_FIX_BLOCKS,
          "router bytes %llu", (unsigned long long)m->router_bytes);
    CHECK(m->shared_bytes == 0, "no shared expert bytes");
    CHECK(m->other_bytes == per_blk_other * POE_FIX_BLOCKS + global_other,
          "other bytes %llu", (unsigned long long)m->other_bytes);
    CHECK(m->total_bytes == m->expert_bytes + m->router_bytes + m->other_bytes,
          "total = experts + routers + other");

    CHECK(strncmp(m->fingerprint, "poe1:", 5) == 0 && strlen(m->fingerprint) == 21,
          "fingerprint format %s", m->fingerprint);

    char fp[24];
    snprintf(fp, sizeof fp, "%s", m->fingerprint);
    poe_model_close(m);

    /* ── fingerprint semantics ─────────────────────────────────────────── */
    printf("fingerprint\n");
    CHECK(poe_model_open(&m, moe_path2, err, sizeof err) == 0, "open copy");
    CHECK(strcmp(m->fingerprint, fp) == 0,
          "identical content, different path -> same fingerprint");
    poe_model_close(m);

    CHECK(poe_model_open(&m, moe_path3, err, sizeof err) == 0, "open seed 9");
    CHECK(strcmp(m->fingerprint, fp) != 0,
          "same structure, different weights -> different fingerprint");
    poe_model_close(m);

    /* ── dense model ───────────────────────────────────────────────────── */
    printf("dense fixture\n");
    CHECK(poe_model_open(&m, dense_path, err, sizeof err) == 0, "open dense");
    CHECK(m->n_moe_blocks == 0, "no moe blocks");
    CHECK(m->expert_bytes == 0, "no expert bytes");
    CHECK(m->total_bytes == m->other_bytes, "everything in 'other'");
    poe_model_close(m);

    /* ── failure path ──────────────────────────────────────────────────── */
    printf("errors\n");
    err[0] = '\0';
    CHECK(poe_model_open(&m, "build/does-not-exist.gguf", err, sizeof err) != 0,
          "missing file rejected");
    CHECK(m == NULL, "out handle NULL on failure");
    CHECK(err[0] != '\0', "error message present: %s", err);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
