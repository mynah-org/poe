/* test_residency.c — the placement arithmetic, and the flags that express it.
 *
 * The properties worth pinning are the ones a user would be misled by if they
 * broke silently: the bytes must add up exactly (nothing invented, nothing
 * lost between GPU and host), a budget that cannot hold the resident floor
 * must say so instead of quietly offloading everything, the eviction order
 * must be the one asked for, and --n-cpu-moe must only appear when it really
 * is equivalent to the -ot pattern printed next to it.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/residency.h"
#include "fixture.h"

static int failures;
static int checks;

#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

static poe_residency *plan(const poe_model *m, uint64_t vram,
                           poe_residency_rank rank, uint64_t reserve) {
    char err[256] = { 0 };
    poe_residency_opts o = {
        .vram_bytes = vram, .ctx = 512, .kv_type = "f16",
        .reserve_bytes = reserve, .rank = rank
    };
    poe_residency *r = NULL;
    if (poe_residency_build(&r, m, &o, err, sizeof err) != 0) {
        printf("  build failed: %s\n", err);
        return NULL;
    }
    return r;
}

int main(void) {
    const char *path = "build/fixture-residency.gguf";
    char err[256] = { 0 };

    CHECK(poe_fixture_moe(path, 3, err, sizeof err) == 0,
          "fixture written (%s)", err[0] ? err : "ok");

    poe_model *m = NULL;
    CHECK(poe_model_open(&m, path, err, sizeof err) == 0,
          "open the fixture (%s)", err[0] ? err : "ok");
    if (m == NULL) { printf("\n%d checks, %d failures\n", checks, ++failures);
                     return 1; }

    /* Budgets are derived from the plan's own resident floor, never guessed,
     * so a change to the KV model cannot silently turn these cases into
     * different ones. */
    const uint64_t BIG = 1024ULL * 1024 * 1024;
    uint64_t floor_bytes = 0;
    {
        poe_residency *probe = plan(m, BIG, POE_RESIDENCY_LAST, 4096);
        CHECK(probe != NULL, "probe plan for the resident floor");
        if (probe == NULL) { printf("\n%d checks, %d failures\n", checks,
                                    ++failures); return 1; }
        floor_bytes = probe->nonexpert_bytes + probe->kv_bytes +
                      probe->reserve_bytes;

        /* The KV term must come from the attention tensors' own widths, not
         * from block_count x head_count_kv: on a hybrid checkpoint the two
         * disagree by 4x, and the fixture is where that formula is pinned.
         * attn_k and attn_v are [embd, embd] in every block. */
        const uint64_t expect = 512ULL * POE_FIX_BLOCKS *
                                (POE_FIX_EMBD + POE_FIX_EMBD) * 2;
        CHECK(probe->kv_bytes == expect,
              "KV estimate reads the attention tensors: %llu == %llu",
              (unsigned long long)probe->kv_bytes, (unsigned long long)expect);
        poe_residency_free(probe);
    }

    /* ── a budget that holds everything ─────────────────────────────────── */
    {
        poe_residency *r = plan(m, BIG, POE_RESIDENCY_LAST, 4096);
        CHECK(r != NULL, "plan with a generous budget");
        if (r) {
            CHECK(r->fits_entirely == 1, "everything fits on the device");
            CHECK(r->expert_bytes_cpu == 0, "nothing displaced to the host");
            CHECK(r->expert_bytes_gpu == m->expert_bytes,
                  "expert bytes on GPU == the checkpoint's (%llu)",
                  (unsigned long long)r->expert_bytes_gpu);
            char f[256];
            poe_residency_flags(r, f, sizeof f);
            CHECK(strcmp(f, "-ngl 99") == 0, "flags are just -ngl 99 (%s)", f);
            CHECK(poe_residency_ncpumoe(r) == 0, "no --n-cpu-moe when nothing moves");
            poe_residency_free(r);
        }
    }

    /* ── a budget that holds the floor and some of the experts ──────────── */
    {
        /* room for the floor plus about half the expert bytes */
        poe_residency *r = plan(m, floor_bytes + m->expert_bytes / 2,
                                POE_RESIDENCY_LAST, 4096);
        CHECK(r != NULL, "plan with a partial budget");
        if (r) {
            CHECK(r->fits_at_all == 1, "the resident floor fits");
            CHECK(r->fits_entirely == 0, "not everything fits");
            CHECK(r->expert_bytes_gpu + r->expert_bytes_cpu == m->expert_bytes,
                  "GPU + host == every expert byte, exactly (%llu + %llu)",
                  (unsigned long long)r->expert_bytes_gpu,
                  (unsigned long long)r->expert_bytes_cpu);
            CHECK(r->expert_bytes_gpu <= r->vram_bytes - r->nonexpert_bytes -
                                         r->kv_bytes - r->reserve_bytes,
                  "what stays on the device fits in the headroom");

            /* 'last' must displace the deepest blocks */
            uint32_t lowest_cpu = m->n_blocks, highest_gpu = 0;
            for (uint32_t i = 0; i < r->n_units; i++) {
                if (r->units[i].on_cpu && r->units[i].block < lowest_cpu)
                    lowest_cpu = r->units[i].block;
                if (!r->units[i].on_cpu && r->units[i].block > highest_gpu)
                    highest_gpu = r->units[i].block;
            }
            CHECK(lowest_cpu > highest_gpu,
                  "rank=last displaces the deepest blocks (host from %u, "
                  "device to %u)", lowest_cpu, highest_gpu);

            char f[512];
            poe_residency_flags(r, f, sizeof f);
            CHECK(strstr(f, "-ot \"blk\\.(") != NULL, "an -ot override is emitted");
            CHECK(strstr(f, "_exps\\.weight=CPU") != NULL,
                  "the override names the expert tensors (%s)", f);
            CHECK(poe_residency_ncpumoe(r) == 0,
                  "no --n-cpu-moe for a suffix: it can only express a prefix");
            poe_residency_free(r);
        }
    }

    /* ── rank=first is the one --n-cpu-moe can express ──────────────────── */
    {
        poe_residency *r = plan(m, floor_bytes + m->expert_bytes / 2,
                                POE_RESIDENCY_FIRST, 4096);
        CHECK(r != NULL, "plan with rank=first");
        if (r) {
            uint32_t highest_cpu = 0, lowest_gpu = m->n_blocks;
            for (uint32_t i = 0; i < r->n_units; i++) {
                if (r->units[i].on_cpu && r->units[i].block > highest_cpu)
                    highest_cpu = r->units[i].block;
                if (!r->units[i].on_cpu && r->units[i].block < lowest_gpu)
                    lowest_gpu = r->units[i].block;
            }
            CHECK(highest_cpu < lowest_gpu,
                  "rank=first displaces the shallowest blocks");
            const uint32_t n = poe_residency_ncpumoe(r);
            CHECK(n == highest_cpu + 1,
                  "--n-cpu-moe %u matches the displaced prefix", n);
            poe_residency_free(r);
        }
    }

    /* ── a budget too small for the floor ───────────────────────────────── */
    {
        poe_residency *r = plan(m, 4096, POE_RESIDENCY_LAST, 4096);
        CHECK(r != NULL, "plan with an impossible budget");
        if (r) {
            CHECK(r->fits_at_all == 0, "reported as not fitting at all");
            CHECK(r->shortfall_bytes > 0, "the shortfall is quantified (%llu B)",
                  (unsigned long long)r->shortfall_bytes);
            CHECK(r->n_warnings > 0, "and it warns rather than pretending");
            poe_residency_free(r);
        }
    }

    /* ── determinism: same inputs, same plan ────────────────────────────── */
    {
        poe_residency *a = plan(m, floor_bytes + m->expert_bytes / 3,
                                POE_RESIDENCY_LAST, 4096);
        poe_residency *b = plan(m, floor_bytes + m->expert_bytes / 3,
                                POE_RESIDENCY_LAST, 4096);
        int same = a && b && a->n_units == b->n_units;
        for (uint32_t i = 0; same && i < a->n_units; i++)
            same = a->units[i].block == b->units[i].block &&
                   a->units[i].on_cpu == b->units[i].on_cpu;
        CHECK(same, "two builds of the same plan agree unit for unit");
        poe_residency_free(a);
        poe_residency_free(b);
    }

    /* ── a quantized KV cache is smaller than f16 ───────────────────────── */
    {
        char e2[256] = { 0 };
        poe_residency_opts o16 = { .vram_bytes = BIG, .ctx = 4096,
                                   .kv_type = "f16", .reserve_bytes = 4096 };
        poe_residency_opts o8  = { .vram_bytes = BIG, .ctx = 4096,
                                   .kv_type = "q8_0", .reserve_bytes = 4096 };
        poe_residency *a = NULL, *b = NULL;
        poe_residency_build(&a, m, &o16, e2, sizeof e2);
        poe_residency_build(&b, m, &o8, e2, sizeof e2);
        CHECK(a && b && b->kv_bytes < a->kv_bytes,
              "q8_0 cache estimated smaller than f16 (%llu < %llu)",
              (unsigned long long)(b ? b->kv_bytes : 0),
              (unsigned long long)(a ? a->kv_bytes : 0));
        poe_residency_free(a);
        poe_residency_free(b);

        poe_residency *bad = NULL;
        poe_residency_opts obad = { .vram_bytes = 1 << 20, .kv_type = "q3_k" };
        CHECK(poe_residency_build(&bad, m, &obad, e2, sizeof e2) != 0,
              "an unsupported cache type is refused, not guessed");
        poe_residency_free(bad);
    }

    /* ── a dense model has nothing to place ─────────────────────────────── */
    {
        const char *dense = "build/fixture-residency-dense.gguf";
        CHECK(poe_fixture_dense(dense, err, sizeof err) == 0, "dense fixture");
        poe_model *d = NULL;
        if (poe_model_open(&d, dense, err, sizeof err) == 0) {
            poe_residency *r = NULL;
            char e2[256] = { 0 };
            poe_residency_opts o = { .vram_bytes = 1 << 30 };
            CHECK(poe_residency_build(&r, d, &o, e2, sizeof e2) != 0,
                  "a dense model is refused with a reason");
            poe_residency_free(r);
            poe_model_close(d);
        }
    }

    poe_model_close(m);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
