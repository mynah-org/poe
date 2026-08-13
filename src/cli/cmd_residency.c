/* cmd_residency.c — `poe residency model.gguf --vram 24G [...]`
 *
 * The advisory form of axis C: what fits on the device you have, and the
 * llama.cpp flags that say so. Static command — no inference, no GPU, and
 * the checkpoint is only read.
 *
 * The report keeps two columns of arithmetic strictly apart. Expert slab
 * bytes are exact, straight off the tensor table. The KV cache and the
 * compute allowance are estimates, and are printed under a heading that says
 * so, because a number that fits in VRAM on paper and not in practice is
 * worse than no number.
 *
 * SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/residency.h"
#include "cli.h"

/* "24G", "24GiB", "16384M", "1.5T", or plain bytes. Binary units throughout,
 * matching how every VRAM figure is actually quoted. Returns 0 on failure. */
static uint64_t parse_size(const char *s) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0) return 0;
    while (*end == ' ') end++;
    double mul = 1.0;
    switch (*end) {
        case 'k': case 'K': mul = 1024.0; break;
        case 'm': case 'M': mul = 1024.0 * 1024; break;
        case 'g': case 'G': mul = 1024.0 * 1024 * 1024; break;
        case 't': case 'T': mul = 1024.0 * 1024 * 1024 * 1024; break;
        case '\0': break;
        default: return 0;
    }
    return (uint64_t)(v * mul);
}

static const char *usage_text(void) {
    return
    "usage: poe residency <model.gguf> --vram <size> [options]\n"
    "\n"
    "  --vram <size>        the budget: 24G, 16384M, bytes\n"
    "  --ctx <n>            context the KV estimate assumes (default: the\n"
    "                       checkpoint's own context_length)\n"
    "  --cache-type <t>     f16 (default), bf16, f32, q8_0, q4_0\n"
    "  --reserve <size>     compute-buffer allowance (default 512M)\n"
    "  --profile <p>        needed only by the experimental rankings\n"
    "  --rank <r>           which slabs leave the device first:\n"
    "                         last      deepest blocks first (default)\n"
    "                         first     shallowest first (--n-cpu-moe's order)\n"
    "                         workset   narrowest routing first  [experiment]\n"
    "                         workset-inverted  its control       [experiment]\n"
    "  --emit-flags         print only the llama.cpp flags, for $(...)\n"
    "  --force              accept a profile from another checkpoint\n"
    "  --json               machine-readable\n"
    "\n"
    "Offloading is quality-neutral: a slab computed on the host returns the\n"
    "same numbers. Only speed is at stake, so no ranking here protects\n"
    "quality, and none of them is a measured recommendation — see\n"
    "docs/residency.md.\n";
}

int poe_cmd_residency(int argc, char **argv) {
    const char *path = NULL, *profile_path = NULL, *cache_type = "f16";
    uint64_t vram = 0, reserve = 0;
    uint32_t ctx = 0;
    int json = 0, emit_flags = 0, force = 0;
    poe_residency_rank rank = POE_RESIDENCY_LAST;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json = 1;
        else if (strcmp(argv[i], "--emit-flags") == 0) emit_flags = 1;
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (strcmp(argv[i], "--vram") == 0 && i + 1 < argc) {
            vram = parse_size(argv[++i]);
            if (vram == 0) {
                fprintf(stderr, "poe residency: bad --vram '%s'\n", argv[i]);
                return 2;
            }
        } else if (strcmp(argv[i], "--reserve") == 0 && i + 1 < argc) {
            reserve = parse_size(argv[++i]);
            if (reserve == 0) {
                fprintf(stderr, "poe residency: bad --reserve '%s'\n", argv[i]);
                return 2;
            }
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v <= 0) {
                fprintf(stderr, "poe residency: bad --ctx '%s'\n", argv[i]);
                return 2;
            }
            ctx = (uint32_t)v;
        } else if (strcmp(argv[i], "--cache-type") == 0 && i + 1 < argc) {
            cache_type = argv[++i];
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile_path = argv[++i];
        } else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if      (strcmp(v, "last")  == 0) rank = POE_RESIDENCY_LAST;
            else if (strcmp(v, "first") == 0) rank = POE_RESIDENCY_FIRST;
            else if (strcmp(v, "workset") == 0) rank = POE_RESIDENCY_WORKSET;
            else if (strcmp(v, "workset-inverted") == 0)
                rank = POE_RESIDENCY_WORKSET_INV;
            else {
                fprintf(stderr, "poe residency: unknown --rank '%s'\n", v);
                return 2;
            }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "poe residency: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (path == NULL) path = argv[i];
        else {
            fprintf(stderr, "poe residency: more than one model given\n");
            return 2;
        }
    }
    if (path == NULL || vram == 0) {
        fputs(usage_text(), stderr);
        return 2;
    }

    char err[256];
    poe_model *m = NULL;
    if (poe_model_open(&m, path, err, sizeof err) != 0) {
        fprintf(stderr, "poe residency: %s\n", err);
        return 1;
    }

    poe_profile *pr = NULL;
    if (profile_path != NULL &&
        poe_profile_load(&pr, profile_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe residency: %s\n", err);
        poe_model_close(m);
        return 1;
    }

    poe_residency_opts opts = {
        .profile = pr, .vram_bytes = vram, .ctx = ctx, .kv_type = cache_type,
        .reserve_bytes = reserve, .rank = rank, .force = force
    };
    poe_residency *r = NULL;
    if (poe_residency_build(&r, m, &opts, err, sizeof err) != 0) {
        fprintf(stderr, "poe residency: %s\n", err);
        poe_profile_free(pr);
        poe_model_close(m);
        return 1;
    }

    char flags[4096];
    poe_residency_flags(r, flags, sizeof flags);
    const uint32_t ncmoe = poe_residency_ncpumoe(r);

    int rc = 0;
    if (emit_flags) {
        printf("%s\n", flags);
    } else if (json) {
        char esc[256];
        printf("{\n");
        printf("  \"fingerprint\": \"%s\",\n", r->model_fingerprint);
        printf("  \"arch\": \"%s\",\n", r->arch);
        printf("  \"rank\": \"%s\",\n", r->rank_method);
        printf("  \"vram_bytes\": %" PRIu64 ",\n", r->vram_bytes);
        printf("  \"exact\": {\n");
        printf("    \"model_bytes\": %" PRIu64 ",\n", r->model_bytes);
        printf("    \"nonexpert_bytes\": %" PRIu64 ",\n", r->nonexpert_bytes);
        printf("    \"expert_bytes\": %" PRIu64 ",\n", r->expert_bytes);
        printf("    \"expert_bytes_gpu\": %" PRIu64 ",\n", r->expert_bytes_gpu);
        printf("    \"expert_bytes_cpu\": %" PRIu64 "\n", r->expert_bytes_cpu);
        printf("  },\n");
        printf("  \"estimated\": {\n");
        printf("    \"kv_bytes\": %" PRIu64 ",\n", r->kv_bytes);
        printf("    \"kv_type\": \"%s\",\n", r->kv_type);
        printf("    \"ctx\": %u,\n", r->ctx);
        printf("    \"reserve_bytes\": %" PRIu64 "\n", r->reserve_bytes);
        printf("  },\n");
        printf("  \"fits_entirely\": %s,\n", r->fits_entirely ? "true" : "false");
        printf("  \"fits_at_all\": %s,\n", r->fits_at_all ? "true" : "false");
        printf("  \"shortfall_bytes\": %" PRIu64 ",\n", r->shortfall_bytes);
        printf("  \"is_split\": %s,\n", r->is_split ? "true" : "false");
        printf("  \"n_cpu_moe\": %u,\n", ncmoe);
        poe_json_escape(flags, esc, sizeof esc);
        printf("  \"flags\": \"%s\",\n", esc);
        printf("  \"units\": [\n");
        for (uint32_t i = 0; i < r->n_units; i++) {
            const poe_residency_unit *u = &r->units[i];
            printf("    {\"block\": %u, \"half\": \"%s\", \"bytes\": %" PRIu64
                   ", \"experts\": %u, \"place\": \"%s\"}%s\n",
                   u->block, u->is_cold ? "cold" : (r->is_split ? "hot" : "all"),
                   u->bytes, u->experts, u->on_cpu ? "cpu" : "gpu",
                   i + 1 < r->n_units ? "," : "");
        }
        printf("  ],\n");
        printf("  \"warnings\": [");
        for (uint32_t i = 0; i < r->n_warnings; i++) {
            poe_json_escape(r->warnings[i], esc, sizeof esc);
            printf("%s\"%s\"", i ? ", " : "", esc);
        }
        printf("]\n}\n");
    } else {
        char b1[32], b2[32], b3[32];

        printf("VRAM residency   %s   %u MoE blocks x %u experts\n",
               r->arch, m->n_moe_blocks, m->expert_count);
        printf("--------------\n");
        poe_format_bytes(r->model_bytes, b1, sizeof b1);
        poe_format_bytes(r->vram_bytes, b2, sizeof b2);
        printf("checkpoint              %s\n", b1);
        printf("budget                  %s\n", b2);
        printf("eviction order          %s%s\n", r->rank_method,
               (r->rank_method[0] == 'w') ? "   [experiment, unmeasured]" : "");
        printf("\n");

        printf("resident floor\n");
        poe_format_bytes(r->nonexpert_bytes, b1, sizeof b1);
        printf("  non-expert weights    %-10s  exact\n", b1);
        poe_format_bytes(r->kv_bytes, b1, sizeof b1);
        printf("  KV cache              %-10s  ESTIMATE  (%u ctx, %s)\n",
               b1, r->ctx, r->kv_type);
        poe_format_bytes(r->reserve_bytes, b1, sizeof b1);
        printf("  compute allowance     %-10s  ESTIMATE\n", b1);

        if (!r->fits_at_all) {
            poe_format_bytes(r->shortfall_bytes, b1, sizeof b1);
            printf("\nthe floor alone is over budget by %s — no placement of "
                   "experts can fix this.\n", b1);
            printf("lower --ctx, use a quantized KV cache, or start from a "
                   "smaller checkpoint.\n");
            rc = 1;
        } else {
            const uint64_t headroom = r->vram_bytes - r->nonexpert_bytes -
                                      r->kv_bytes - r->reserve_bytes;
            poe_format_bytes(headroom, b1, sizeof b1);
            printf("  headroom for experts  %-10s\n", b1);
            printf("\n");

            uint32_t n_cpu = 0;
            for (uint32_t i = 0; i < r->n_units; i++) n_cpu += r->units[i].on_cpu;
            poe_format_bytes(r->expert_bytes, b1, sizeof b1);
            poe_format_bytes(r->expert_bytes_gpu, b2, sizeof b2);
            poe_format_bytes(r->expert_bytes_cpu, b3, sizeof b3);
            printf("routed experts          %s total, %u placeable %s (exact)\n",
                   b1, r->n_units, r->is_split ? "halves" : "slabs");
            printf("  on GPU                %-10s  %u\n", b2, r->n_units - n_cpu);
            printf("  on host               %-10s  %u\n", b3, n_cpu);

            if (r->fits_entirely) {
                printf("\neverything fits on the device.\n");
            } else if (r->is_split) {
                uint32_t cold_cpu = 0;
                for (uint32_t i = 0; i < r->n_units; i++)
                    cold_cpu += r->units[i].on_cpu && r->units[i].is_cold;
                printf("\ncold halves displaced first: %u of them. On a split "
                       "checkpoint the host\nholds the experts this workload "
                       "routes to least, which is the one placement\nrule here "
                       "that is not a guess.\n", cold_cpu);
            }
        }

        printf("\nflags\n  %s\n", flags);
        if (ncmoe)
            printf("  equivalently          --n-cpu-moe %u\n", ncmoe);

        for (uint32_t i = 0; i < r->n_warnings; i++)
            printf("\nwarning: %s\n", r->warnings[i]);
    }

    poe_residency_free(r);
    poe_profile_free(pr);
    poe_model_close(m);
    return rc;
}
