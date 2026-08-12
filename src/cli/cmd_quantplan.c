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

#include "poe/imatrix_stats.h"
#include "poe/quantplan.h"
#include "poe/stats.h"
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
    const char *model_path = NULL, *profile_path = NULL, *imatrix_path = NULL;
    const char *out_path = NULL, *tt_path = NULL, *size_arg = NULL;
    const char *imat_stat = "energy";
    char *types_arg = NULL;
    int force = 0, invert = 0;

    for (int i = 0; i < argc; i++) {
        if      (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_path = argv[++i];
        else if (strcmp(argv[i], "--imatrix") == 0 && i + 1 < argc) imatrix_path = argv[++i];
        else if (strcmp(argv[i], "--imatrix-stat") == 0 && i + 1 < argc) imat_stat = argv[++i];
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
            "                     [--profile p.poeprofile] [--imatrix i.gguf]\n"
            "                     [--types q2_k,q4_k,...] [-o out.poequant]\n"
            "                     [--tensor-types out.txt] [--force]\n"
            "\n"
            "  --target-size    budget for the routed expert slabs together:\n"
            "                   16G / 12.5GiB / 60%% (of the source slabs) / bytes\n"
            "  --profile        rank layers by REAP saliency; without it, uniform\n"
            "  --imatrix        rank layers by an imatrix statistic instead — the\n"
            "                   quantity a weighted encoder actually fits against\n"
            "  --imatrix-stat   energy (default) or concentration; concentration is\n"
            "                   scale-free and cannot be an activation-norm ramp\n"
            "  --tensor-types   also write the --tensor-type file llama-quantize reads\n"
            "  --invert         flip the ranking: the control allocation an\n"
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

    /* An imatrix ranking, when asked for, replaces the profile's saliency:
     * deletion asks which experts contribute least to the output, requant
     * asks which inputs a weighted fit must not lose, and those are not the
     * same question. With both files given the two rankings are reported
     * side by side — how far apart they fall is the cheap experiment. */
    poe_imatrix_stats *im = NULL;
    double *scores = NULL;
    char label[32] = {0};
    if (imatrix_path != NULL) {
        if (poe_imatrix_stats_load(&im, imatrix_path, err, sizeof err) != 0) {
            fprintf(stderr, "poe quantplan: %s\n", err);
            poe_profile_free(prof);
            poe_model_close(m);
            return 1;
        }
        const int by_conc = strncmp(imat_stat, "conc", 4) == 0;
        if (!by_conc && strcmp(imat_stat, "energy") != 0) {
            fprintf(stderr, "poe quantplan: --imatrix-stat must be "
                            "'energy' or 'concentration'\n");
            poe_imatrix_stats_free(im);
            poe_profile_free(prof);
            poe_model_close(m);
            return 2;
        }
        if (im->n_layers != m->n_blocks) {
            fprintf(stderr, "poe quantplan: the imatrix covers %u layers, "
                            "the model has %u\n", im->n_layers, m->n_blocks);
            poe_imatrix_stats_free(im);
            poe_profile_free(prof);
            poe_model_close(m);
            return 1;
        }
        if (im->n_experts != m->expert_count)
            fprintf(stderr, "poe quantplan: warning: the imatrix carries %u "
                            "experts per layer, the model has %u\n",
                    im->n_experts, m->expert_count);
        scores = by_conc ? im->concentration : im->energy;
        snprintf(label, sizeof label, "imatrix-%s", by_conc ? "conc" : "energy");
    }

    poe_quantplan *p = NULL;
    const poe_quantplan_opts opts = {
        .profile = prof, .layer_scores = scores,
        .score_label = scores ? label : NULL,
        .target_bytes = target, .types = n_types ? types : NULL,
        .n_types = n_types, .force = force, .invert = invert
    };
    if (poe_quantplan_build_opts(&p, m, &opts, err, sizeof err) != 0) {
        fprintf(stderr, "poe quantplan: %s\n", err);
        poe_imatrix_stats_free(im);
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
           (prof != NULL || im != NULL)
               ? "" : "  (no profile: every layer weighted equally)");
    if (im != NULL) {
        printf("imatrix   %s   %u entries, %u x %u, %llu routed slots",
               imatrix_path, im->n_entries, im->n_layers, im->n_experts,
               (unsigned long long)im->slots_total);
        if (im->experts_unseen)
            printf(", %u experts never routed", im->experts_unseen);
        printf("\n");
        if (im->dataset[0])
            printf("          calibrated on %s (%u chunks x %u tokens)\n",
                   im->dataset, im->chunk_count, im->chunk_size);
        /* Two rankings of the same layers: if they agree, the extra file
         * bought nothing; if they disagree, one of them is wrong and only a
         * matched-bytes measurement can say which. */
        if (prof != NULL && prof->reap_mean != NULL &&
            prof->n_layers == im->n_layers) {
            double *reap = calloc(im->n_layers, sizeof *reap);
            if (reap != NULL) {
                for (uint32_t l = 0; l < im->n_layers; l++)
                    for (uint32_t e = 0; e < prof->n_experts; e++)
                        reap[l] += prof->reap_mean[(size_t)l * prof->n_experts + e];
                printf("          Spearman vs REAP saliency %+.3f\n",
                       poe_spearman(p->layer_score, reap, im->n_layers));
                free(reap);
            }
        }
    }
    if (p->depth_rho != 0.0)
        printf("depth     Spearman %+.3f against the layer index%s\n",
               p->depth_rho,
               (p->depth_rho >= 0.9 || p->depth_rho <= -0.9)
                   ? "   (see the warning below)" : "");
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
    poe_imatrix_stats_free(im);
    poe_profile_free(prof);
    poe_model_close(m);
    return rc;
}
