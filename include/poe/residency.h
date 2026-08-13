/* poe/residency.h — what fits in VRAM, and what to tell llama.cpp about it.
 *
 * The third axis of §0: not what the checkpoint contains (pruning) nor what
 * it executes (routing) but *where it lives*. Given a VRAM budget, decide
 * which routed expert slabs stay on the device and emit the llama.cpp flags
 * that express the decision. No backend patch, no rewrite, no inference.
 *
 * One property separates this from the other two axes and shapes the whole
 * design: **offloading is quality-neutral**. A slab computed on the CPU
 * produces the same numbers as one computed on the GPU, so there is nothing
 * for a saliency metric to protect. Placement costs time, not accuracy, and
 * an ordering can only be justified by traffic — which is why the profile
 * does not pick layers here (see poe_residency_rank).
 *
 * Exact and estimated quantities are kept apart and labelled as such: slab
 * bytes come from the checkpoint, KV cache and compute buffers do not.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_RESIDENCY_H
#define POE_RESIDENCY_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/profile.h"

#define POE_RESIDENCY_MAX_WARN 8

/* Which unit leaves the device first when not everything fits.
 *
 * There is no measured evidence that any per-layer order beats any other,
 * and on this architecture per-layer differentiation has already lost twice
 * (R7 for routing budgets, M9a for precision). So the default is llama.cpp's
 * own convention, and the profile-driven orders ship as *experiments* with
 * their control attached, not as recommendations.
 *
 * The one ordering that is not a guess is hot/cold on a split checkpoint:
 * the cold tensor holds the experts the workload routes to least, so putting
 * it on the host is strictly less traffic than putting an arbitrary layer
 * there. That rule applies on top of whichever rank is chosen. */
typedef enum {
    POE_RESIDENCY_LAST = 0,   /* deepest blocks leave first (the default)     */
    POE_RESIDENCY_FIRST,      /* shallowest first: what --n-cpu-moe expresses */
    POE_RESIDENCY_WORKSET,    /* profile: narrowest routing leaves first      */
    POE_RESIDENCY_WORKSET_INV /* the control for the above                    */
} poe_residency_rank;

/* One placeable unit: a block's routed expert tensors. A split checkpoint
 * contributes two units per block, because its halves are separate tensors
 * and can therefore be placed separately — which is the only way per-expert
 * residency is expressible without patching the runtime. */
typedef struct {
    uint32_t block;
    int      is_cold;          /* the low-precision half of a split slab      */
    uint64_t bytes;            /* exact, from the checkpoint                  */
    uint32_t experts;          /* experts stored in this unit                 */
    double   score;            /* ranking key; the largest leaves first       */
    int      on_cpu;
} poe_residency_unit;

typedef struct {
    char     poe_version[32];
    char     model_fingerprint[24];
    char     arch[64];
    char     rank_method[24];

    uint64_t vram_bytes;           /* the budget, as given                    */

    /* Exact — read off the checkpoint. */
    uint64_t model_bytes;
    uint64_t nonexpert_bytes;      /* everything that is not a routed expert  */
    uint64_t expert_bytes;
    uint64_t expert_bytes_gpu;
    uint64_t expert_bytes_cpu;

    /* Estimates — stated separately because they are not measurements. */
    uint64_t kv_bytes;
    uint64_t reserve_bytes;
    uint32_t ctx;                  /* context the KV estimate assumes         */
    double   kv_bytes_per_elem;
    char     kv_type[8];

    /* Placement outcome. */
    int      fits_entirely;        /* nothing had to leave the device         */
    int      fits_at_all;          /* the resident floor alone fits           */
    uint64_t shortfall_bytes;      /* 0 unless fits_at_all is 0               */
    int      is_split;             /* the checkpoint carries hot/cold slabs   */

    uint32_t n_units;
    poe_residency_unit *units;

    uint32_t n_warnings;
    char     warnings[POE_RESIDENCY_MAX_WARN][160];
} poe_residency;

typedef struct {
    const poe_profile *profile;    /* only used by the WORKSET ranks          */
    uint64_t vram_bytes;           /* required                                */
    uint32_t ctx;                  /* 0 -> the model's own context_length     */
    const char *kv_type;           /* "f16" (default), "q8_0", "q4_0"         */
    uint64_t reserve_bytes;        /* 0 -> POE_RESIDENCY_DEFAULT_RESERVE      */
    poe_residency_rank rank;
    int      force;                /* accept a profile from another model     */
} poe_residency_opts;

/* A flat allowance for llama.cpp's compute buffers, which depend on batch
 * size, vocabulary and backend in ways this command deliberately does not
 * model. Crude on purpose, adjustable, and always reported as an estimate. */
#define POE_RESIDENCY_DEFAULT_RESERVE (512ULL * 1024 * 1024)

int  poe_residency_build(poe_residency **out, const poe_model *m,
                         const poe_residency_opts *o, char *err, size_t errsz);
void poe_residency_free(poe_residency *r);

/* The llama.cpp invocation that expresses the plan, written into `buf`:
 * `-ngl 99` plus an `-ot ...=CPU` override when anything was displaced.
 * Returns the length that would have been written (snprintf contract), or
 * -1 on error. */
int  poe_residency_flags(const poe_residency *r, char *buf, size_t bufsz);

/* --n-cpu-moe N when the displaced set is exactly the first N blocks and
 * whole (both halves of a split), else 0: the short flag is only equivalent
 * to the plan in that one case, and quietly printing it otherwise would be
 * an accounting lie. */
uint32_t poe_residency_ncpumoe(const poe_residency *r);

#endif
