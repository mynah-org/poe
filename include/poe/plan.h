/* poe/plan.h — .poeplan: a deterministic proposed transformation.
 *
 * A plan binds a source model (by fingerprint) and one or more profiles to
 * an explicit per-layer keep/prune decision plus exact byte accounting.
 * Building a plan never touches the checkpoint; `poe apply` consumes it. Reproducibility invariant: same model + same profiles + same
 * parameters + same POE version -> byte-identical plan file.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_PLAN_H
#define POE_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/profile.h"

#define POE_PLAN_MAX_WARN 8

typedef struct {
    uint32_t version;                  /* .poeplan format version           */
    char     poe_version[32];
    char     model_fingerprint[24];
    char     arch[64];
    char     method[16];               /* reap | frequency | gate           */
    double   prune_fraction;           /* requested fraction                */
    uint32_t n_layers, n_experts, top_k;
    uint32_t keep_per_layer;           /* uniform per-layer keep (MVP)      */

    /* membership map [n_layers * n_experts]: 1 = keep, 0 = prune.
     * The new index of a kept expert is its rank among kept ids in
     * ascending order — the remap `poe apply` will use. */
    uint8_t *keep;

    /* Exact on-disk accounting computed from the source model:
     * removed = pruned expert slices + pruned router rows. */
    uint64_t bytes_before, bytes_removed;
    uint64_t params_before, params_removed;

    /* Super-expert protection (M10). `rescued` is the number the ranking
     * would have deleted and protection kept — the only figure that says
     * whether the guard did anything, and the one to quote. `still_pruned`
     * is non-zero only when a layer held more flagged experts than the cut
     * could spare, which is loud rather than silent. */
    int      protect_super;
    uint32_t n_super_flagged;
    uint32_t n_super_rare;
    uint32_t n_super_rescued;
    uint32_t n_super_still_pruned;

    uint32_t n_warnings;
    char     warnings[POE_PLAN_MAX_WARN][160];
} poe_plan;

/* Build a plan from a model and n_profiles weighted profiles.
 * method: "reap" (needs reap data), "frequency", or "gate". Scores from
 * multiple profiles are min-max normalized per layer and combined by
 * `weights`. Selection is a uniform per-layer bottom cut, clamped so at
 * least top_k experts survive. Fingerprint mismatch between model and any
 * profile is an error unless `force`. */
int poe_plan_build(poe_plan **out, const poe_model *m,
                   const poe_profile *const *profiles, const double *weights,
                   size_t n_profiles, const char *method, double prune_frac,
                   int force, char *err, size_t errsz);

/* The same, plus super-expert protection (M10). Flagged experts are lifted
 * out of the cut and the *next* candidates take their place, so the per-layer
 * keep count — which `poe apply` requires to be uniform — is unchanged and
 * the byte accounting stays exact. Protection needs a profile with
 * activation norms; without one it is reported as unavailable, never
 * silently skipped.
 *
 * `protect_super` defaults on at the CLI, because the failure it guards
 * against is silent: a model that lost three super experts still writes
 * fluent text. */
typedef struct {
    const poe_profile *const *profiles;
    const double *weights;
    size_t       n_profiles;
    const char  *method;
    double       prune_frac;
    int          force;
    int          protect_super;
    double       super_z;              /* 0 -> POE_SUPER_DEFAULT_Z */
} poe_plan_opts;

int poe_plan_build_opts(poe_plan **out, const poe_model *m,
                        const poe_plan_opts *o, char *err, size_t errsz);

int  poe_plan_write(const poe_plan *p, const char *path,
                    char *err, size_t errsz);
int  poe_plan_load(poe_plan **out, const char *path, char *err, size_t errsz);
void poe_plan_free(poe_plan *p);

#endif
