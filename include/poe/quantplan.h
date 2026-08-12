/* poe/quantplan.h — .poequant: a per-slab mixed-precision decision.
 *
 * Where a .poeplan deletes experts, a .poequant re-types them: each routed
 * expert slab gets its own quantization type, chosen so a byte budget buys
 * the least damage. The unit is the slab and not the expert because a GGUF
 * tensor carries exactly one type and routed experts are packed one tensor
 * per projection — per-expert precision needs a runtime change, this does
 * not.
 *
 * The output is consumed two ways: as a `--tensor-type` file for
 * llama-quantize, and (later) by POE's own encoder. Building a plan never
 * touches the checkpoint.
 *
 * Reproducibility invariant, same as .poeplan: same model + same profile +
 * same parameters + same POE version -> byte-identical plan file.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_QUANTPLAN_H
#define POE_QUANTPLAN_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/profile.h"

#define POE_QUANT_MAX_WARN 8

/* The routed projections, in the order their tensors are named. */
typedef enum {
    POE_QSLAB_GATE = 0,
    POE_QSLAB_UP   = 1,
    POE_QSLAB_DOWN = 2,
    POE_QSLAB_NPROJ
} poe_qslab;

typedef struct {
    uint32_t version;
    char     poe_version[32];
    char     model_fingerprint[24];
    char     arch[64];
    char     method[32];               /* saliency | uniform | imatrix-*    */
    uint32_t n_layers, n_experts;

    /* Per layer × projection, indexed [layer * POE_QSLAB_NPROJ + proj]. */
    int      *type;                    /* chosen ingot type id              */
    int      *type_before;             /* what the source carried           */
    uint64_t *bytes_before, *bytes_after;
    uint64_t *nelem;

    /* Per layer, the normalized saliency the allocation was driven by
     * (1.0 = the most salient layer). Uniform method leaves these at 1. */
    double   *layer_score;

    /* Spearman of layer_score against depth. A score that is nearly a
     * monotone function of the layer index is the shape both losing
     * allocations had, so it is recorded in the artifact and warned about
     * rather than left for a reader to notice. 0 when there is no ranking. */
    double    depth_rho;

    /* Exact accounting over routed expert slabs only — the tensors this
     * plan re-types. Everything else in the checkpoint is untouched, so it
     * is deliberately not counted here. */
    uint64_t  bytes_before_total, bytes_after_total, target_bytes;
    uint64_t  model_bytes_before;      /* whole file, for context           */

    uint32_t  n_warnings;
    char      warnings[POE_QUANT_MAX_WARN][160];
} poe_quantplan;

/* Build a plan. `target_bytes` is the budget for the routed expert slabs
 * together; the allocator starts every slab at the cheapest candidate type
 * and spends the remaining budget on the upgrades that buy the most quality
 * per byte. `profile` may be NULL, which forces the uniform method.
 * `types`/`n_types` optionally restricts the candidate set (NULL = the
 * default ladder Q2_K..Q6_K). Fingerprint mismatch is an error unless
 * `force`. Returns 0 on success. */
int poe_quantplan_build(poe_quantplan **out, const poe_model *m,
                        const poe_profile *profile, uint64_t target_bytes,
                        const int *types, size_t n_types, int force,
                        char *err, size_t errsz);

/* Everything the allocator can be told, in one place.
 *
 * `layer_scores` is an externally computed per-layer ranking — an imatrix
 * statistic, say — and takes precedence over `profile`, which is then used
 * for nothing but its fingerprint check. `score_label` names it in the
 * artifact ("imatrix-energy"); NULL means "external". */
typedef struct {
    const poe_profile *profile;
    const double      *layer_scores;
    const char        *score_label;
    uint64_t           target_bytes;
    const int         *types;
    size_t             n_types;
    int                force;
    int                invert;
} poe_quantplan_opts;

int poe_quantplan_build_opts(poe_quantplan **out, const poe_model *m,
                             const poe_quantplan_opts *o,
                             char *err, size_t errsz);

/* The same, with `invert` flipping every layer score (score -> 1 - score).
 * A deliberately wrong allocation is the control an experiment needs: it
 * separates "the saliency signal is informative" from "any non-uniform map
 * moves the number", which is the only way to tell a real effect from an
 * artifact of, say, activation norms growing with depth. */
int poe_quantplan_build_ex(poe_quantplan **out, const poe_model *m,
                           const poe_profile *profile, uint64_t target_bytes,
                           const int *types, size_t n_types, int force,
                           int invert, char *err, size_t errsz);

/* The .poequant artifact (JSON, human-diffable). */
int poe_quantplan_write(const poe_quantplan *p, const char *path,
                        char *err, size_t errsz);

/* Read one back. Only what a comparison needs is reconstructed — the type
 * map, the per-slab bytes, the layer scores and the accounting — so a loaded
 * plan is for reporting, not for re-deriving an allocation. */
int poe_quantplan_load(poe_quantplan **out, const char *path,
                       char *err, size_t errsz);

/* The `--tensor-type` file llama-quantize reads: one `name=type` per line,
 * with the name written as a regex anchored tightly enough not to match a
 * sibling layer. */
int poe_quantplan_write_tensor_types(const poe_quantplan *p, const char *path,
                                     char *err, size_t errsz);

void poe_quantplan_free(poe_quantplan *p);

/* The candidate ladder and the measured relative-L2 error each type costs,
 * for reporting. Returns the entry count; `types` and `errors` may be NULL. */
size_t poe_quantplan_ladder(const int **types, const double **errors);

#endif
