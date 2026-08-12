/* profiler/imatrix.h — per-expert activation statistics, in llama.cpp's
 * imatrix format.
 *
 * An imatrix is the sum of squares of the *inputs* to a weight matmul,
 * accumulated over calibration tokens. Quantizers use it to weight their
 * least-squares fit by how much each input column actually matters, which is
 * worth real error at two and three bits and almost nothing at six.
 *
 * For MoE this is per expert: `poe-profile` already sees the expert matmul
 * nodes and their routing ids, so the statistic costs one more accumulator
 * on a pass we already make, and `counts[e]` is a routing frequency profile
 * for free. An expert the calibration never routed to keeps count 0 — legal
 * in the format, and for a single-domain calibration it is the normal case
 * and the signal a mixed-precision plan wants to read.
 *
 * Pure C, no inference-backend dependency: the runner copies tensors off the
 * backend and feeds them here, same contract as accum.h.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_IMATRIX_H
#define POE_IMATRIX_H

#include <stddef.h>
#include <stdint.h>

/* The three routed projections, in the order their tensors are named. */
typedef enum {
    POE_IMAT_GATE = 0,
    POE_IMAT_UP   = 1,
    POE_IMAT_DOWN = 2,
    POE_IMAT_NPROJ
} poe_imat_proj;

/* One non-MoE weight matrix's statistic, keyed by the tensor's own name.
 * The routed experts are the part POE needs for its own decisions, but a
 * whole-model quantize driven by an expert-only file falls back to
 * unweighted fits everywhere else — attention, embeddings, the dense path.
 * Closing that is one more accumulator on the same pass. */
typedef struct {
    char      name[128];
    uint32_t  cols;
    uint64_t  count;             /* rows folded in: llama.cpp's counts[0]   */
    double   *sum2;              /* [cols]                                  */
} poe_imat_plain;

typedef struct {
    uint32_t n_layers, n_experts;
    /* Per projection, allocated on first observation because the column
     * count is not known until a tensor arrives. sum2 is
     * [layer][expert][col], counts is [layer][expert]. */
    uint32_t  cols[POE_IMAT_NPROJ];
    double   *sum2[POE_IMAT_NPROJ];
    uint64_t *counts[POE_IMAT_NPROJ];
    /* Non-MoE weight matrices, grown as they are first seen. */
    poe_imat_plain *plain;
    size_t    n_plain, cap_plain;

    uint64_t  observations;      /* matmul nodes folded in                  */
    uint64_t  nonfinite;         /* squared inputs that were not finite     */
} poe_imatrix;

int  poe_imatrix_init(poe_imatrix *m, uint32_t n_layers, uint32_t n_experts);
void poe_imatrix_free(poe_imatrix *m);

/* Fold in one expert-matmul node for a batch of T tokens.
 *
 *   act      the matmul's input activations, [cols × act_rows × T] contiguous
 *            f32. act_rows is src1->ne[1]: it is `n_expert_used` when each
 *            selected expert gets its own row and 1 when the same input is
 *            broadcast to all of them, so the row for slot s is s % act_rows.
 *   ids      [n_used × T] contiguous, the expert chosen in each slot.
 *
 * Out-of-range ids are skipped and counted in *bad_ids. Returns 0, or -1 on
 * a column-count change (a shape the caller must not silently average over)
 * or allocation failure. */
int poe_imatrix_observe(poe_imatrix *m, uint32_t layer, poe_imat_proj proj,
                        uint32_t cols, uint32_t act_rows, uint32_t n_used,
                        uint32_t T, const float *act, const int32_t *ids,
                        uint64_t *bad_ids);

/* Fold in one plain matmul: `wname` is the weight tensor's own GGUF name,
 * `act` its [cols × rows] contiguous f32 input. Rows are tokens, and every
 * row counts — which is what llama.cpp's collector records in counts[0] for
 * a non-MoE tensor. Returns 0, or -1 on a column-count change or an
 * allocation failure. */
int poe_imatrix_observe_plain(poe_imatrix *m, const char *wname,
                              uint32_t cols, uint32_t rows, const float *act);

/* True once any projection has data — lets the runner decide whether an
 * imatrix is worth writing at all. */
int poe_imatrix_has_data(const poe_imatrix *m);

/* Write llama.cpp's GGUF imatrix (`general.type = "imatrix"`, one
 * `<weight>.in_sum2` [cols, n_experts] and `<weight>.counts` [1, n_experts]
 * per routed projection). Tensor names are the model's own weight names,
 * emitted in lexicographic order so the same profile yields the same bytes.
 * `chunk_size` is the token count a chunk was scored over; `dataset` is
 * recorded for provenance and may be NULL. Returns 0, or -1 with `err`. */
int poe_imatrix_write_gguf(const poe_imatrix *m, const char *path,
                           const char *dataset, uint32_t chunk_count,
                           uint32_t chunk_size, char *err, size_t errsz);

#endif
