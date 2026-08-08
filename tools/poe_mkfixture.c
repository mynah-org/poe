/* poe_mkfixture.c — write the synthetic fixtures to a path of your choosing,
 * for poking at the CLI without a real model:
 *
 *   poe-mkfixture out.gguf            # MoE fixture (seed 1)
 *   poe-mkfixture out.gguf --seed 7   # same structure, different weights
 *   poe-mkfixture out.gguf --dense    # dense (non-MoE) fixture
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tests/fixture.h"

int main(int argc, char **argv) {
    const char *path = NULL;
    uint32_t seed = 1;
    int dense = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dense") == 0) dense = 1;
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (argv[i][0] != '-' && path == NULL) path = argv[i];
        else { fprintf(stderr, "usage: poe-mkfixture <out.gguf> [--dense] [--seed N]\n"); return 2; }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: poe-mkfixture <out.gguf> [--dense] [--seed N]\n");
        return 2;
    }

    char err[256] = "";
    int rc = dense ? poe_fixture_dense(path, err, sizeof err)
                   : poe_fixture_moe(path, seed, err, sizeof err);
    if (rc != 0) { fprintf(stderr, "poe-mkfixture: %s\n", err); return 1; }
    printf("wrote %s\n", path);
    return 0;
}
