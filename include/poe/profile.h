/* poe/profile.h — .poeprofile loading and cross-profile comparison.
 *
 * A profile is the persistent, reusable output of `poe-profile`: routing
 * statistics of one model under one workload. Comparison is where POE
 * answers "are the same experts used for coding?" — all static, no
 * inference, exact arithmetic on the stored counters.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_PROFILE_H
#define POE_PROFILE_H

#include <stddef.h>
#include <stdint.h>

#define POE_PROFILE_NMASS 4    /* thresholds .80 .90 .95 .99, writer-fixed  */
#define POE_PROFILE_KHIST 64   /* min-k bins spanning 1..n_experts          */

typedef struct {
    uint32_t version;                 /* "poeprofile" format version       */
    char     poe_version[32];
    char     fingerprint[24];         /* source model                      */
    char     arch[64];
    char     dataset_hash[40];
    uint32_t n_layers, n_experts, top_k;
    uint64_t tokens;

    /* per layer [n_layers] */
    uint64_t *layer_tokens;
    double   *entropy_bits;
    double   *mass_k[POE_PROFILE_NMASS];

    /* The distribution behind that mean: [layer][bin], bin b counting
     * tokens that needed b+1 experts, the last bin counting "more than
     * POE_PROFILE_KHIST-1". NULL for profiles written before it existed.
     *
     * The mean says how many experts a token needs on average; only the
     * spread says whether tokens differ from each other, which is the
     * difference between a static schedule (dead, see R7) and token-adaptive
     * routing (undecided, and expensive to build). */
    uint64_t *mass_k_hist[POE_PROFILE_NMASS];

    /* How few of the experts a token actually runs carry most of what they
     * contribute: per layer the mean min-k over `gate x ||expert output||`
     * within the applied top-k, and its distribution [layer][k-1].
     *
     * This is the min-k question asked of the quantity that governs the
     * output rather than of probability mass, which R9 showed governs
     * nothing here. Bounded by top_k, so a number in this table is directly
     * a statement about K. NULL when the profile carries no REAP data. */
    double   *contrib_k[POE_PROFILE_NMASS];
    uint64_t *contrib_k_hist[POE_PROFILE_NMASS];

    /* Per layer, the mean probability mass the applied top-k actually holds.
     * The mirror of mass_k: not "how many experts would carry 80% of the
     * mass" but "how much mass do the experts we run carry". 0 for profiles
     * written before it existed. */
    double   *topk_mass;

    /* per layer × expert [n_layers * n_experts] */
    uint64_t *sel_count;
    double   *gate_mean;

    /* REAP saliency arrays, present only when the profile was captured
     * with --metric reap (NULL otherwise). */
    double   *reap_mean;      /* mean gate · ‖expert_output‖₂ over routed  */
    double   *actnorm_mean;   /* mean ‖expert_output‖₂ (unweighted)        */
} poe_profile;

int  poe_profile_load(poe_profile **out, const char *path,
                      char *err, size_t errsz);
void poe_profile_free(poe_profile *p);

/* ── pairwise comparison ────────────────────────────────────────────────── */

typedef struct {
    /* top-X% expert-set agreement, across layers */
    double   jaccard_mean, jaccard_min, jaccard_max;
    uint32_t jaccard_min_layer, jaccard_max_layer;
    /* weighted Jaccard of selection frequencies: Σ min / Σ max            */
    double   wjaccard_mean;
    /* Spearman rank correlation of selection counts (average-rank ties)   */
    double   spearman_mean;
    /* Jensen-Shannon divergence of selection distributions, bits          */
    double   jsd_bits_mean;
    /* experts in A's top set but not B's (and vice versa), summed         */
    uint64_t exclusive_a, exclusive_b;
    /* experts unselected in BOTH profiles, summed over layers             */
    uint64_t cold_both;

    /* REAP agreement, only when BOTH profiles carry reap data             */
    int      has_reap;
    double   reap_spearman_mean;      /* rank agreement of saliency        */
    double   reap_bottom_jaccard;     /* bottom-X% prune-set agreement     */
} poe_profile_cmp;

/* Compare two profiles of the same shape. `top_frac` selects the top-X%
 * expert set per layer (e.g. 0.25). `per_layer_jaccard`, if non-NULL,
 * receives n_layers values. Returns -1 on shape mismatch. */
int poe_profile_compare(const poe_profile *a, const poe_profile *b,
                        double top_frac, poe_profile_cmp *out,
                        double *per_layer_jaccard);

#endif
