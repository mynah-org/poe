/* cmd_requant.c — `poe requant <model.gguf> -o out.gguf --carrier T
 *                  --degrade-type T [--degrade-frac F --profile p] [--invert]`
 *
 * The M9b gate: per-expert precision, emulated inside one tensor type, so
 * the question can be measured before a runtime patch is paid for. See
 * include/poe/requant.h for what the emulation does and does not model.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poe/requant.h"
#include "cli.h"

/* q4_k / Q4_K alike, over the types ingot can encode. */
static int type_by_name(const char *s) {
    for (int t = 0; t < 64; t++) {
        const char *n = ingot_type_name(t);
        if (n == NULL || !ingot_can_quantize(t)) continue;
        size_t i = 0;
        for (; n[i] && s[i]; i++) {
            const char a = n[i] >= 'A' && n[i] <= 'Z' ? (char)(n[i] + 32) : n[i];
            const char b = s[i] >= 'A' && s[i] <= 'Z' ? (char)(s[i] + 32) : s[i];
            if (a != b) break;
        }
        if (n[i] == '\0' && s[i] == '\0') return t;
    }
    return -1;
}

int poe_cmd_requant(int argc, char **argv) {
    const char *model_path = NULL, *out_path = NULL, *profile_path = NULL;
    const char *carrier_arg = NULL, *degrade_arg = NULL;
    double frac = 1.0;
    int invert = 0, force = 0;

    for (int i = 0; i < argc; i++) {
        if      (strcmp(argv[i], "-o") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) profile_path = argv[++i];
        else if (strcmp(argv[i], "--carrier") == 0 && i + 1 < argc) carrier_arg = argv[++i];
        else if (strcmp(argv[i], "--degrade-type") == 0 && i + 1 < argc) degrade_arg = argv[++i];
        else if (strcmp(argv[i], "--degrade-frac") == 0 && i + 1 < argc) {
            frac = atof(argv[++i]);
            if (frac > 1.0) frac /= 100.0;              /* accept 57% or 0.57 */
        }
        else if (strcmp(argv[i], "--invert") == 0) invert = 1;
        else if (strcmp(argv[i], "--force") == 0) force = 1;
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe requant: unknown option '%s'\n", argv[i]);
            return 2;
        }
        else if (model_path == NULL) model_path = argv[i];
        else { fprintf(stderr, "poe requant: more than one model given\n"); return 2; }
    }
    if (model_path == NULL || out_path == NULL || carrier_arg == NULL ||
        degrade_arg == NULL) {
        fprintf(stderr,
            "usage: poe requant <model.gguf> -o <out.gguf>\n"
            "                   --carrier TYPE --degrade-type TYPE\n"
            "                   [--degrade-frac F --profile p.poeprofile]\n"
            "                   [--invert] [--force]\n"
            "\n"
            "  --carrier        the type every routed slab is stored as\n"
            "  --degrade-type   the type cold experts are pushed through and back\n"
            "  --degrade-frac   how many experts per layer are degraded (default 1,\n"
            "                   i.e. all of them — the matched-bytes control)\n"
            "  --profile        ranks experts within each layer; required below 1\n"
            "  --invert         degrade the HOTTEST experts: the control an\n"
            "                   experiment needs, never something to ship\n"
            "\n"
            "Every arm is the same file size (the carrier's), so two runs at the\n"
            "same average bits per weight differ only in where the damage went.\n");
        return 2;
    }

    poe_requant_opts o = { 0, 0, frac, NULL, invert, force };
    o.carrier_type = type_by_name(carrier_arg);
    o.degrade_type = type_by_name(degrade_arg);
    if (o.carrier_type < 0 || o.degrade_type < 0) {
        fprintf(stderr, "poe requant: unknown or unsupported type '%s'\n",
                o.carrier_type < 0 ? carrier_arg : degrade_arg);
        return 2;
    }

    char err[256];
    poe_model *m = NULL;
    if (poe_model_open(&m, model_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe requant: %s\n", err);
        return 1;
    }
    poe_profile *prof = NULL;
    if (profile_path != NULL &&
        poe_profile_load(&prof, profile_path, err, sizeof err) != 0) {
        fprintf(stderr, "poe requant: %s\n", err);
        poe_model_close(m);
        return 1;
    }
    o.profile = prof;

    poe_requant_stats st;
    if (poe_requant(m, &o, out_path, &st, err, sizeof err) != 0) {
        fprintf(stderr, "poe requant: %s\n", err);
        poe_profile_free(prof);
        poe_model_close(m);
        return 1;
    }

    char sz[32];
    poe_format_bytes(st.bytes_written, sz, sizeof sz);
    printf("model     %s   %s   (%s, %u layers x %u experts)\n",
           model_path, m->fingerprint, m->arch, m->n_blocks, m->expert_count);
    printf("carrier   %-6s  %.4f bits/weight\n",
           ingot_type_name(o.carrier_type), st.carrier_bits);
    printf("degraded  %-6s  %.4f bits/weight   %u of %u experts per layer%s\n",
           ingot_type_name(o.degrade_type), st.degrade_bits,
           st.degraded_per_layer, m->expert_count,
           o.invert ? "   (INVERTED: the hottest)" : "");
    if (st.degraded_per_layer < m->expert_count)
        printf("ranked by %s\n", st.ranked_by_reap ? "REAP saliency"
                                                   : "selection counts");
    printf("emulates  %.4f bits/weight over the routed slabs\n",
           st.emulated_bits);
    printf("\nwrote %s   %s   (%u slabs rewritten, %llu experts degraded)\n",
           out_path, sz, st.slabs_rewritten,
           (unsigned long long)st.experts_degraded);
    printf("note: the file is a plain %s checkpoint; the mixed allocation is "
           "emulated,\n      not stored, so this measures quality at a size "
           "the format can hold.\n", ingot_type_name(o.carrier_type));

    poe_profile_free(prof);
    poe_model_close(m);
    return 0;
}
