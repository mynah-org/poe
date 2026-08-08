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
"  inspect  <model.gguf> [--json]            MoE structure, storage, reductions\n"
"  experts  <model.gguf> [--layer N] [--json] expert tensor mapping and sizes\n"
"\n"
"planned (see plan.md for the milestone each belongs to)\n"
"  profile   observe a workload, write a .poeprofile            (M2, M4, M5)\n"
"  compare   compare workload profiles                          (M3)\n"
"  plan      turn a profile + constraints into a .poeplan       (M6)\n"
"  estimate  disk/RAM/VRAM effect of a plan                     (M6)\n"
"  diff      diff models, profiles or plans                     (M6)\n"
"  apply     rewrite the checkpoint according to a plan         (M7)\n"
"  forge     profile + plan + apply in one command              (M8)\n"
"  validate  structural / smoke / behavioral checks             (M7+)\n"
"\n"
"  --version print version and exit\n"
"  --help    this text\n";

static int stub(const char *cmd, const char *milestone) {
    fprintf(stderr,
            "poe %s: not implemented yet (planned for %s — see plan.md)\n",
            cmd, milestone);
    return 2;
}

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

    if (strcmp(cmd, "profile")  == 0) return stub(cmd, "milestones M2/M4/M5");
    if (strcmp(cmd, "compare")  == 0) return stub(cmd, "milestone M3");
    if (strcmp(cmd, "plan")     == 0) return stub(cmd, "milestone M6");
    if (strcmp(cmd, "estimate") == 0) return stub(cmd, "milestone M6");
    if (strcmp(cmd, "diff")     == 0) return stub(cmd, "milestone M6");
    if (strcmp(cmd, "apply")    == 0) return stub(cmd, "milestone M7");
    if (strcmp(cmd, "forge")    == 0) return stub(cmd, "milestone M8");
    if (strcmp(cmd, "validate") == 0) return stub(cmd, "milestone M7+");

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
