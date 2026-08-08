/* poe/apply.h — structural GGUF pruning: materialize a .poeplan.
 *
 * Rewrites a checkpoint by slicing the packed expert tensors along the
 * expert dimension, compacting the router rows in the same keep order (the
 * plan's remap rule), patching <arch>.expert_count, and preserving every
 * other tensor and metadata key byte-for-byte. The source is read through
 * the existing mmap and streamed sequentially to the output — the model is
 * never materialized in RAM.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_APPLY_H
#define POE_APPLY_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/plan.h"

typedef struct {
    uint64_t bytes_written;            /* whole output file                 */
    uint64_t payload_bytes;            /* tensor payload section, exact     */
    uint32_t tensors_total;
    uint32_t tensors_sliced;           /* expert + router tensors rewritten */
    uint32_t kv_dropped;               /* stale poe.* keys from a prior run */
    int      expert_count_patched;     /* <arch>.expert_count updated       */
} poe_apply_stats;

/* Apply `p` to `m`, writing the pruned checkpoint to out_path. The plan's
 * fingerprint must match the model unless `force`; topology (n_layers,
 * n_experts) must match unconditionally. Single-file GGUF v2/v3 only.
 * Deterministic: same model + same plan -> byte-identical output. */
int poe_apply(const poe_model *m, const poe_plan *p, const char *out_path,
              int force, poe_apply_stats *stats, char *err, size_t errsz);

#endif
