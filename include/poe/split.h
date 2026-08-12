/* poe/split.h — store hot and cold experts at different precisions.
 *
 * This is the checkpoint half of M9b. A GGUF tensor carries exactly one
 * type, so per-expert precision cannot live inside one packed slab: the
 * slab has to become two tensors. `poe requant` measured that the idea is
 * worth building — aiming the damage at cold experts beat spreading it
 * evenly by 2.4x at identical bytes, in the calibration domain. This
 * writes the artifact that can actually hold the win.
 *
 * The layout, and why it needs no id remap at runtime:
 *
 *   experts are reordered so the H the workload needs most come first, and
 *   the router's rows are permuted the same way. Expert id therefore keeps
 *   meaning "row of the router", and "is this expert hot" becomes the test
 *   `id < H` — no lookup table, no indirection.
 *
 *   blk.N.ffn_{gate,up,down}_exps.weight        [.., .., H]      hot type
 *   blk.N.ffn_{gate,up,down}_exps_cold.weight   [.., .., E-H]    cold type
 *
 *   poe.split.hot_count tells a runtime where the line is.
 *
 * The output does NOT load in a stock runtime: consuming it needs two
 * `mul_mat_id` passes and a merge in `build_moe_ffn`. The command says so,
 * and `poe inspect` reads the file back so the accounting can be checked
 * before any of that exists.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_SPLIT_H
#define POE_SPLIT_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/profile.h"
#include "poe/rank.h"

typedef struct {
    const poe_profile *profile;  /* required: the order is the whole point  */
    double      hot_fraction;    /* share of experts kept at the hot type   */
    int         hot_type;        /* ingot type for the experts kept wide    */
    int         cold_type;       /* ingot type for the rest                 */
    poe_rank_by rank_by;
    int         invert;          /* keep the LEAST needed experts wide      */
    int         force;
    int         threads;         /* 0 = one per online core                 */
} poe_split_opts;

typedef struct {
    uint32_t hot_per_layer, cold_per_layer;
    uint32_t slabs_split;
    uint32_t routers_permuted;
    uint64_t bytes_written, bytes_before;
    double   hot_bits, cold_bits, mean_bits;
    int      ranked_by_reap;
} poe_split_stats;

int poe_split(const poe_model *m, const poe_split_opts *o, const char *out_path,
              poe_split_stats *stats, char *err, size_t errsz);

#endif
