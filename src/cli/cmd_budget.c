/* cmd_budget.c — `poe routing-budget model.gguf [--profile p] [--json]`
 *
 * R0 of the adaptive-routing track: exact active-parameter accounting as a
 * function of the routed expert count K. This answers "can an A3B model
 * become ~A2B just by lowering K?" from the checkpoint itself:
 *
 *   ActiveParams(K) = fixed_active + Σ_blocks min(K, E_b) · (P_b / E_b)
 *
 * where P_b is the block's total routed-expert parameters and E_b its
 * expert count. The fixed component (attention, norms, routers, shared
 * experts) never scales with K, which is exactly why the naive
 * "3B × K/8" extrapolation is wrong. Embeddings are reported separately —
 * published "active parameter" figures usually include them.
 *
 * Static command: no inference, no GPU, exact counts only.
 *
 * SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "poe/profile.h"
#include "cli.h"

/* How many experts a token actually needs, and — the part that decides
 * whether token-adaptive routing is worth kernel work — how much that
 * varies from token to token.
 *
 * A static per-layer schedule can only exploit variation *across layers*,
 * and R7 measured that away. Adaptivity needs variation *across tokens*,
 * which a mean cannot show: a tight distribution means every token wants
 * the same budget and adaptivity has nothing to sell, a wide one means a
 * fixed K is overspending on most tokens to serve a minority. */
static void print_mass_k(const poe_profile *pr) {
    static const double thr[POE_PROFILE_NMASS] = { 0.80, 0.90, 0.95, 0.99 };
    int any = 0;
    for (int i = 0; i < POE_PROFILE_NMASS; i++) any |= pr->mass_k_hist[i] != NULL;
    if (!any) {
        printf("\nExperts a token actually needs\n");
        printf("  the profile predates the histogram; only means are available:\n");
        for (int i = 0; i < POE_PROFILE_NMASS; i++) {
            double sum = 0;
            for (uint32_t l = 0; l < pr->n_layers; l++) sum += pr->mass_k[i][l];
            printf("    %.0f%% mass   mean k %.2f\n", thr[i] * 100.0,
                   pr->n_layers ? sum / pr->n_layers : 0.0);
        }
        return;
    }

    const uint32_t E = pr->n_experts ? pr->n_experts : 1;
    const uint32_t w = (E + POE_PROFILE_KHIST - 1) / POE_PROFILE_KHIST;
    printf("\nExperts a token actually needs   (pooled over %u layers, "
           "%llu tokens, of %u experts)\n",
           pr->n_layers, (unsigned long long)pr->tokens, E);
    if (w > 1)
        printf("  bins are %u experts wide; values are the upper edge\n", w);
    printf("  mass    mean    p50   p90   p99   max    share of tokens at or below the mean\n");
    for (int i = 0; i < POE_PROFILE_NMASS; i++) {
        if (pr->mass_k_hist[i] == NULL) continue;
        uint64_t bins[POE_PROFILE_KHIST];
        uint64_t total = 0;
        for (int b = 0; b < POE_PROFILE_KHIST; b++) {
            bins[b] = 0;
            for (uint32_t l = 0; l < pr->n_layers; l++)
                bins[b] += pr->mass_k_hist[i][(size_t)l * POE_PROFILE_KHIST + b];
            total += bins[b];
        }
        if (total == 0) continue;

        /* bin b covers k in [b*w+1, (b+1)*w]; report its upper edge, capped
         * at the expert count, so a number is never larger than reality */
        double mean = 0;
        for (int b = 0; b < POE_PROFILE_KHIST; b++) {
            const uint32_t hi = (uint32_t)(b + 1) * w < E
                              ? (uint32_t)(b + 1) * w : E;
            mean += (double)hi * (double)bins[b];
        }
        mean /= (double)total;

        uint32_t q[3] = { 0, 0, 0 };
        const double want[3] = { 0.50, 0.90, 0.99 };
        uint64_t cum = 0;
        int qi = 0;
        uint32_t maxk = 1;
        for (int b = 0; b < POE_PROFILE_KHIST; b++) {
            const uint32_t hi = (uint32_t)(b + 1) * w < E
                              ? (uint32_t)(b + 1) * w : E;
            if (bins[b]) maxk = hi;
            cum += bins[b];
            while (qi < 3 && (double)cum / (double)total >= want[qi])
                q[qi++] = hi;
        }
        while (qi < 3) q[qi++] = maxk;

        uint64_t at_or_below = 0;
        for (int b = 0; b < POE_PROFILE_KHIST; b++) {
            const uint32_t hi = (uint32_t)(b + 1) * w < E
                              ? (uint32_t)(b + 1) * w : E;
            if ((double)hi <= mean) at_or_below += bins[b];
        }

        printf("  %3.0f%%   %6.1f  %4u  %4u  %4u  %4u     %5.1f%%\n",
               thr[i] * 100.0, mean, q[0], q[1], q[2], maxk,
               100.0 * (double)at_or_below / (double)total);
    }
    if (pr->topk_mass != NULL) {
        double sum = 0;
        uint32_t n = 0;
        double lo = 1.0, hi = 0.0;
        for (uint32_t l = 0; l < pr->n_layers; l++) {
            if (pr->topk_mass[l] <= 0.0) continue;
            sum += pr->topk_mass[l];
            if (pr->topk_mass[l] < lo) lo = pr->topk_mass[l];
            if (pr->topk_mass[l] > hi) hi = pr->topk_mass[l];
            n++;
        }
        if (n)
            printf("\n  the top-%u the model actually runs holds %.1f%% of the "
                   "mass   (per layer %.1f%%-%.1f%%)\n",
                   pr->top_k, 100.0 * sum / n, 100.0 * lo, 100.0 * hi);
    }
    printf("\n  A fixed K must be set for the tail, so the gap between p50 and p99\n"
           "  is what token-adaptive routing could recover — and nothing else can.\n");
}

/* The same question asked of output contribution instead of probability
 * mass. R9 established that mass governs nothing here — the top-8 holds a
 * sixth of it and the model is fine — so the quantity that decides whether K
 * can fall is how concentrated the *contribution* of the experts we already
 * run is. Bounded by top_k, which is what makes it a statement about K. */
static void print_contrib_k(const poe_profile *pr) {
    static const double thr[POE_PROFILE_NMASS] = { 0.80, 0.90, 0.95, 0.99 };
    int any = 0;
    for (int i = 0; i < POE_PROFILE_NMASS; i++) any |= pr->contrib_k_hist[i] != NULL;
    if (!any) {
        printf("\nContribution concentration inside the applied top-k\n");
        printf("  needs a profile captured with --metric reap\n");
        return;
    }
    const uint32_t K = pr->top_k ? pr->top_k : 1;
    printf("\nContribution concentration inside the applied top-%u   "
           "(gate x ||expert output||)\n", K);
    printf("  share    mean k   p50   p90   p99   share of tokens needing all %u\n", K);
    for (int i = 0; i < POE_PROFILE_NMASS; i++) {
        if (pr->contrib_k_hist[i] == NULL) continue;
        uint64_t bins[64] = { 0 };
        uint64_t total = 0;
        const uint32_t nb = K < 64 ? K : 64;
        for (uint32_t b = 0; b < nb; b++) {
            for (uint32_t l = 0; l < pr->n_layers; l++)
                bins[b] += pr->contrib_k_hist[i][(size_t)l * K + b];
            total += bins[b];
        }
        if (total == 0) continue;

        double mean = 0;
        for (uint32_t b = 0; b < nb; b++) mean += (double)(b + 1) * (double)bins[b];
        mean /= (double)total;

        uint32_t q[3] = { 1, 1, 1 };
        const double want[3] = { 0.50, 0.90, 0.99 };
        uint64_t cum = 0;
        int qi = 0;
        for (uint32_t b = 0; b < nb; b++) {
            cum += bins[b];
            while (qi < 3 && (double)cum / (double)total >= want[qi]) q[qi++] = b + 1;
        }
        while (qi < 3) q[qi++] = nb;

        printf("  %3.0f%%    %6.2f  %4u  %4u  %4u        %5.1f%%\n",
               thr[i] * 100.0, mean, q[0], q[1], q[2],
               100.0 * (double)bins[nb - 1] / (double)total);
    }
    printf("\n  A K below the p99 column costs the tail of tokens some of their\n"
           "  contribution; a K at or above it costs nothing measurable here.\n");
}

/* Σ_blocks min(K, E_b) · per-expert params. Per-block so that a model with
 * heterogeneous expert counts still adds up exactly. */
static uint64_t routed_params_at_k(const poe_model *m, uint32_t k) {
    uint64_t sum = 0;
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (!blk->is_moe || blk->expert_count == 0) continue;
        uint32_t use = k < blk->expert_count ? k : blk->expert_count;
        sum += blk->expert_params / blk->expert_count * use;
    }
    return sum;
}

int poe_cmd_routing_budget(int argc, char **argv) {
    const char *path = NULL, *profile_path = NULL;
    int json = 0;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json = 1;
        else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc)
            profile_path = argv[++i];
        else if (argv[i][0] == '-') {
            fprintf(stderr, "poe routing-budget: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (path == NULL) path = argv[i];
        else {
            fprintf(stderr, "poe routing-budget: more than one model given\n");
            return 2;
        }
    }
    if (path == NULL) {
        fprintf(stderr, "usage: poe routing-budget <model.gguf> "
                        "[--profile p.poeprofile] [--json]\n"
                        "\n"
                        "  --profile   also report how many experts a token\n"
                        "              actually needs, and how much that varies\n");
        return 2;
    }

    char err[256];
    poe_model *m;
    if (poe_model_open(&m, path, err, sizeof err) != 0) {
        fprintf(stderr, "poe routing-budget: %s\n", err);
        return 1;
    }
    if (m->n_moe_blocks == 0) {
        fprintf(stderr, "poe routing-budget: no MoE structure in this model — "
                        "every parameter is always active\n");
        poe_model_close(m);
        return 1;
    }

    /* Everything that runs no matter what K is. */
    const uint64_t fixed = m->router_params + m->shared_params + m->other_params;
    const uint32_t orig_k = m->experts_per_token;
    const uint32_t max_k  = orig_k ? orig_k : (m->expert_count < 8 ? m->expert_count : 8);

    if (json) {
        printf("{\n");
        printf("  \"fingerprint\": \"%s\",\n", m->fingerprint);
        printf("  \"experts_total\": %u,\n", m->expert_count);
        printf("  \"original_k\": %u,\n", orig_k);
        printf("  \"shared_expert_count\": %u,\n", m->shared_expert_count);
        printf("  \"params\": {\n");
        printf("    \"total\": %" PRIu64 ",\n", m->total_params);
        printf("    \"experts_stored\": %" PRIu64 ",\n", m->expert_params);
        printf("    \"routers\": %" PRIu64 ",\n", m->router_params);
        printf("    \"shared_experts\": %" PRIu64 ",\n", m->shared_params);
        printf("    \"embeddings\": %" PRIu64 ",\n", m->embedding_params);
        printf("    \"other\": %" PRIu64 ",\n", m->other_params);
        printf("    \"fixed_active\": %" PRIu64 "\n", fixed);
        printf("  },\n");
        printf("  \"active_by_k\": [\n");
        for (uint32_t k = max_k; k >= 1; k--) {
            uint64_t act = fixed + routed_params_at_k(m, k);
            printf("    {\"k\": %u, \"active\": %" PRIu64
                   ", \"active_with_embeddings\": %" PRIu64 "}%s\n",
                   k, act, act + m->embedding_params, k > 1 ? "," : "");
        }
        printf("  ]\n}\n");
        poe_model_close(m);
        return 0;
    }

    char p1[32], p2[32];

    printf("Model routing budget   (exact checkpoint accounting, embeddings listed apart)\n");
    printf("--------------------\n");
    printf("experts total           %u\n", m->expert_count);
    if (orig_k) printf("original routed K       %u\n", orig_k);
    else        printf("original routed K       unknown (no expert_used_count metadata)\n");
    if (m->shared_expert_count || m->shared_params) {
        poe_format_params(m->shared_params, p1, sizeof p1);
        printf("shared experts          %u   (%s params, always active)\n",
               m->shared_expert_count, p1);
    }
    printf("\n");

    poe_format_params(m->total_params, p1, sizeof p1);
    printf("total params            %s\n", p1);
    poe_format_params(m->expert_params, p1, sizeof p1);
    printf("routed expert params    %s stored\n", p1);
    poe_format_params(fixed, p1, sizeof p1);
    printf("fixed active params     %s   (attention, norms, routers, shared)\n", p1);
    poe_format_params(m->embedding_params, p1, sizeof p1);
    printf("embedding params        %s\n", p1);
    poe_format_params(routed_params_at_k(m, 1), p1, sizeof p1);
    printf("per-K routed params     %s per unit of K\n", p1);
    printf("\n");

    printf("  K    active params    with embeddings    vs K=%u\n", max_k);
    uint64_t base = fixed + routed_params_at_k(m, max_k);
    for (uint32_t k = max_k; k >= 1; k--) {
        uint64_t act = fixed + routed_params_at_k(m, k);
        poe_format_params(act, p1, sizeof p1);
        poe_format_params(act + m->embedding_params, p2, sizeof p2);
        printf("  %-4u %-16s %-18s %5.1f%%\n", k, p1, p2,
               base ? 100.0 * (double)act / (double)base : 0.0);
    }

    poe_format_params(fixed + routed_params_at_k(m, 1) + m->embedding_params,
                      p1, sizeof p1);
    printf("\nminimum routing-only floor (K=1, with embeddings)   %s\n", p1);
    printf("note: compute floor only — resident RAM/VRAM is unchanged unless\n");
    printf("      experts are also pruned (poe plan/apply) or offloaded.\n");

    if (profile_path != NULL) {
        poe_profile *pr = NULL;
        if (poe_profile_load(&pr, profile_path, err, sizeof err) != 0) {
            fprintf(stderr, "poe routing-budget: %s\n", err);
            poe_model_close(m);
            return 1;
        }
        if (strcmp(pr->fingerprint, m->fingerprint) != 0)
            printf("\nwarning: the profile was captured from a different "
                   "checkpoint\n");
        print_mass_k(pr);
        print_contrib_k(pr);
        poe_profile_free(pr);
    }

    poe_model_close(m);
    return 0;
}
