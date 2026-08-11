/* cmd_quantplan.c — `poe quantplan model.gguf [--profile p.poeprofile]
 *                    --target-size SIZE [--types q2_k,q4_k,...]
 *                    [-o out.poequant] [--tensor-types out.txt] [--force]`
 *
 * Milestone M9a: spend a byte budget across the routed expert slabs by
 * choosing a quantization type per slab, instead of deleting experts. The
 * plan is consumed by llama-quantize through a --tensor-type file.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/quantplan.h"
#include "cli.h"

/* "16G", "16GiB", "12.5G", "60%" (of the source slabs), or plain bytes. */
static int parse_size(const char *s, uint64_t base, uint64_t *out) {
    char *end = NULL;
    const double v = strtod(s, &end);
    if (end == s || v <= 0.0) return -1;
    while (*end == ' ') end++;
    double mult = 1.0;
    if (*end == '%') {
        if (end[1] != '\0') return -1;
        *out = (uint64_t)(base * v / 100.0);
        return 0;
    }
    switch (*end) {
    case 'k': case 'K': mult = 1024.0; break;
    case 'm': case 'M': mult = 1024.0 * 1024.0; break;
    case 'g': case 'G': mult = 1024.0 * 1024.0 * 1024.0; break;
    case 't': case 'T': mult = 1024.0 * 1024.0 * 1024.0 * 1024.0; break;
    case '\0': mult = 1.0; break;
    default: return -1;
    }
    if (*end != '\0') {
        end++;
        if (*end && strcmp(end, "B") != 0 && strcmp(end, "iB") != 0 &&
            strcmp(end, "b") != 0) return -1;
    }
    *out = (uint64_t)(v * mult);
    return 0;
}

static int parse_types(char *list, int *out, size_t cap, size_t *n) {
    *n = 0;
    for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        const int *ladder;
        const size_t nl = poe_quantplan_ladder(&ladder, NULL);
        int found = -1;
        for (size_t i = 0; i < nl; i++) {
            const char *name = ingot_type_name(ladder[i]);
            /* accept q4_k / Q4_K alike */
            size_t j = 0;
            for (; name[j] && tok[j]; j++) {
                const char a = name[j] >= 'A' && name[j] <= 'Z'
                             ? (char)(name[j] + 32) : name[j];
                const char b = tok[j] >= 'A' && tok[j] <= 'Z'
                             ? (char)(tok[j] + 32) : tok[j];
                if (a != b) break;
            }
            if (name[j] == '\0' && tok[j] == '\0') { found = ladder[i]; break; }
        }
        if (found < 0) {
            fprintf(stderr, "poe quantplan: unknown or unsupported type '%s'\n", tok);
            return -1;
        }
        if (*n == cap) return -1;
        out[(*n)++] = found;
    }
    return *n > 0 ? 0 : -1;
}

int poe_cmd_quantplan(int argc, char **argv) {
    const char *model_path = NULL, *profile_path = NULL;
    const char *out_path = NULL, *tt_path = NULL, *size_arg = NULL;
    char *types_arg = NULL;
    int force = 0, invert = 0;

    for (int i = 0; i < argc; i++) {
        if      (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_path = argv[++i];
        else if (strcmp(argv[i], "--target-size") == 0 && i + 1 < argc) size_arg = argv[++i];
        else if (strcmp(argv[i], "--types") == 0 && i + 1 < argc)   types_arg = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)        out_path = argv[++i];
        else if (strcmp(argv[i], "--tensor-types") == 0 && i + 1 < argc) tt_path = argv[++i];
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (strcmp(argv[i], "--invert") == 0) invert = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe quantplan: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (model_path == NULL) model_path = argv[i];
        else { fprintf(stderr, "poe quantplan: more than one model given\n"); return 2; }
    }
    if (model_path == NULL || size_arg == NULL) {
        fprintf(stderr,
            "usage: poe quantplan <model.gguf> --target-size SIZE\n"
            "                     [--profile p.poeprofile] [--types q2_k,q4_k,...]\n"
            "                     [-o out.poequant] [--tensor-types out.txt] [--force]\n"
            "\n"
            "  --target-size    budget for the routed expert slabs together:\n"
            "                   16G / 12.5GiB / 60%% (of the source slabs) / bytes\n"
            "  --profile        rank layers by saliency; without it, uniform\n"
            "  --tensor-types   also write the --tensor-type file llama-quantize reads\n"
            "  --invert         flip the saliency ranking: the control allocation an\n"
            "                   experiment needs, never something to ship\n");
        return 2;
    }

    char err[256];
    poe_model *m = NULL;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe quantplan: %s\n", err);
        return 1;
    }

    /* The budget is a fraction OF THE SLABS, so it needs their size first. */
    uint64_t slab_bytes = 0;
    for (uint32_t l = 0; l < m->n_blocks; l++) {
        const poe_block *b = &m->blocks[l];
        if (b->gate_exps_w) slab_bytes += b->gate_exps_w->nbytes;
        if (b->up_exps_w)   slab_bytes += b->up_exps_w->nbytes;
        if (b->down_exps_w) slab_bytes += b->down_exps_w->nbytes;
    }
    uint64_t target = 0;
    if (parse_size(size_arg, slab_bytes, &target) != 0) {
        fprintf(stderr, "poe quantplan: cannot parse size '%s'\n", size_arg);
        poe_model_close(m);
        return 2;
    }

    int types[8];
    size_t n_types = 0;
    if (types_arg != NULL && parse_types(types_arg, types, 8, &n_types) != 0) {
        poe_model_close(m);
        return 2;
    }

    poe_profile *prof = NULL;
    if (profile_path != NULL &&
        poe_profile_load(&prof, profile_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe quantplan: %s\n", err);
        poe_model_close(m);
        return 1;
    }

    poe_quantplan *p = NULL;
    if (poe_quantplan_build_ex(&p, m, prof, target, n_types ? types : NULL,
                               n_types, force, invert, err, sizeof err) != 0) {
        fprintf(stderr, "poe quantplan: %s\n", err);
        poe_profile_free(prof);
        poe_model_close(m);
        return 1;
    }

    /* Report: the type histogram is what a reader actually wants to see —
     * forty lines of per-layer types are unreadable and are in the file. */
    char b1[32], b2[32], b3[32];
    poe_format_bytes(p->bytes_before_total, b1, sizeof b1);
    poe_format_bytes(p->bytes_after_total, b2, sizeof b2);
    poe_format_bytes(p->target_bytes, b3, sizeof b3);
    printf("model     %s   %s   (%s, %u layers x %u experts)\n",
           model_path, p->model_fingerprint, p->arch, p->n_layers, p->n_experts);
    printf("method    %s%s\n", p->method,
           prof != NULL ? "" : "  (no profile: every layer weighted equally)");
    printf("\nRouted expert slabs\n");
    printf("  before        %10s\n", b1);
    printf("  target        %10s\n", b3);
    printf("  after         %10s   (%.1f%% of before)\n", b2,
           p->bytes_before_total ? 100.0 * (double)p->bytes_after_total /
                                   (double)p->bytes_before_total : 0.0);

    const int *ladder;
    const size_t nl = poe_quantplan_ladder(&ladder, NULL);
    printf("\nSlabs per type\n");
    for (size_t i = 0; i < nl; i++) {
        uint32_t count = 0;
        for (size_t j = 0; j < (size_t)p->n_layers * POE_QSLAB_NPROJ; j++)
            if (p->type[j] == ladder[i]) count++;
        if (count) printf("  %-6s %5u\n", ingot_type_name(ladder[i]), count);
    }
    for (uint32_t i = 0; i < p->n_warnings; i++)
        printf("\nwarning: %s\n", p->warnings[i]);

    char defout[512];
    if (out_path == NULL) {
        snprintf(defout, sizeof defout, "%s.poequant", model_path);
        out_path = defout;
    }
    int rc = 0;
    if (poe_quantplan_write(p, out_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe quantplan: %s\n", err);
        rc = 1;
    } else {
        printf("\nwrote %s\n", out_path);
    }
    if (rc == 0 && tt_path != NULL) {
        if (poe_quantplan_write_tensor_types(p, tt_path, err, sizeof err) != 0) {
            fprintf(stderr, "poe quantplan: %s\n", err);
            rc = 1;
        } else {
            printf("wrote %s  (llama-quantize --tensor-type-file)\n", tt_path);
        }
    }

    poe_quantplan_free(p);
    poe_profile_free(prof);
    poe_model_close(m);
    return rc;
}
