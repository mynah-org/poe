/* main.c — the poe CLI: command dispatch.
 *
 * Static commands (inspect, experts) are implemented; the rest of the
 * pipeline exists as explicit stubs so `poe --help` already shows the shape
 * of the tool and scripts fail loudly rather than mysteriously.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <string.h>

#include "cli.h"

static const char USAGE[] =
"poe — Pruning & Optimization of Experts. Poe, the reaper of MoE.\n"
"\n"
"usage: poe <command> [args]\n"
"\n"
"static analysis (no inference, no GPU)\n"
"  inspect         <model.gguf> [--json]             MoE structure, storage, reductions\n"
"  experts         <model.gguf> [--layer N] [--json] expert tensor mapping and sizes\n"
"  routing-budget  <model.gguf> [--json]             active params as a function of K\n"
"  compare         <a.poeprofile> <b.poeprofile>...  expert-fingerprint overlap between\n"
"                  [--top P] [--json]                workload profiles\n"
"  plan            <model.gguf> --profile <p[:W]>... build a deterministic expert-removal\n"
"                  [--method reap|frequency|gate]    plan with exact byte accounting\n"
"                  [--prune P] [-o out.poeplan]\n"
"  estimate        <plan.poeplan> [model.gguf]       plan accounting (+ verification)\n"
"  diff            <a> <b>                           diff plans, profiles, or models\n"
"  residency       <model.gguf> --vram 24G           what fits on the device you have,\n"
"                  [--ctx N] [--rank R]             and the llama.cpp flags that\n"
"                  [--emit-flags]                   place the rest on the host\n"
"  quantplan       <model.gguf> --target-size S      mixed precision per expert slab:\n"
"                  [--profile <p>] [--types ...]     spend a byte budget on the slabs\n"
"                  [--tensor-types out.txt]          that matter, emit llama-quantize\n"
"                                                    --tensor-type-file\n"
"\n"
"checkpoint rewriting\n"
"  apply           <plan.poeplan> <model.gguf>       structural expert pruning: slice\n"
"                  -o <out.gguf> [--force]           experts, compact router, verify\n"
"  forge           <model.gguf> -o <out.gguf>        the whole pipeline in one command:\n"
"                  (--profile <p[:W]>... |           (profile) -> plan -> apply -> verify,\n"
"                   --dataset <text>)                intermediate artifacts preserved\n"
"                  [--method M] [--prune P]\n"
"  split           <model.gguf> -o <out.gguf>         hot and cold experts at two\n"
"                  --profile <p> --hot-type T        precisions, as two tensors per\n"
"                  --cold-type T [--hot-frac F]      slab (needs a patched runtime)\n"
"  requant         <model.gguf> -o <out.gguf>         per-expert precision, emulated\n"
"                  --carrier T --degrade-type T      inside one type: same file size,\n"
"                  [--degrade-frac F --profile <p>]  the damage aimed at cold experts\n"
"  validate        <model.gguf> [--plan <p>]         structural checks; with a plan,\n"
"                                                    source/pruned provenance and exact\n"
"                                                    byte accounting\n"
"\n"
"profiling (needs an inference backend)\n"
"  profile   use the standalone poe-profile tool for now:\n"
"            make profiler LLAMA_DIR=<llama.cpp>  ->  build/poe-profile\n"
"\n"
"\n"
"  --version print version and exit\n"
"  --help    this text\n";

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "help") == 0) {
        fputs(USAGE, argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "version") == 0) {
        printf("poe %s\n", POE_VERSION);
        return 0;
    }

    const char *cmd = argv[1];
    int    sub_argc = argc - 2;
    char **sub_argv = argv + 2;

    if (strcmp(cmd, "inspect")  == 0) return poe_cmd_inspect(sub_argc, sub_argv);
    if (strcmp(cmd, "experts")  == 0) return poe_cmd_experts(sub_argc, sub_argv);
    if (strcmp(cmd, "routing-budget") == 0)
        return poe_cmd_routing_budget(sub_argc, sub_argv);
    if (strcmp(cmd, "compare")  == 0) return poe_cmd_compare(sub_argc, sub_argv);
    if (strcmp(cmd, "plan")     == 0) return poe_cmd_plan(sub_argc, sub_argv);
    if (strcmp(cmd, "estimate") == 0) return poe_cmd_estimate(sub_argc, sub_argv);
    if (strcmp(cmd, "diff")     == 0) return poe_cmd_diff(sub_argc, sub_argv);
    if (strcmp(cmd, "quantplan") == 0) return poe_cmd_quantplan(sub_argc, sub_argv);
    if (strcmp(cmd, "apply")    == 0) return poe_cmd_apply(sub_argc, sub_argv);
    if (strcmp(cmd, "forge")    == 0) return poe_cmd_forge(sub_argc, sub_argv);
    if (strcmp(cmd, "requant")  == 0) return poe_cmd_requant(sub_argc, sub_argv);
    if (strcmp(cmd, "split")    == 0) return poe_cmd_split(sub_argc, sub_argv);
    if (strcmp(cmd, "validate") == 0) return poe_cmd_validate(sub_argc, sub_argv);
    if (strcmp(cmd, "residency") == 0) return poe_cmd_residency(sub_argc, sub_argv);

    if (strcmp(cmd, "profile")  == 0) {
        fprintf(stderr, "poe profile: use the standalone poe-profile tool "
                        "(make profiler LLAMA_DIR=<llama.cpp checkout>)\n");
        return 2;
    }

    fprintf(stderr, "poe: unknown command '%s' (try poe --help)\n", cmd);
    return 2;
}

void poe_json_escape(const char *src, char *dst, size_t dstsz) {
    size_t o = 0;
    for (const char *p = src; *p && o + 7 < dstsz; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { dst[o++] = '\\'; dst[o++] = (char)c; }
        else if (c < 0x20) o += (size_t)snprintf(dst + o, dstsz - o, "\\u%04x", c);
        else dst[o++] = (char)c;
    }
    dst[o] = '\0';
}
