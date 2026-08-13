/* poe/poe.h — POE public API: open a GGUF checkpoint and discover its
 * Mixture-of-Experts structure. Static inspection only: no inference, no
 * gradients, no GPU. Dynamic profiling lives behind a separate header once
 * milestone M2 lands.
 *
 * Errors are returned as a code plus a message in the caller's buffer; this
 * library never writes to stderr. (Same contract as ingot.)
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_H
#define POE_H

#include <stddef.h>
#include <stdint.h>

#include "ingot.h"

#ifdef __cplusplus
extern "C" {
#endif

#define POE_VERSION "0.1.0"

/* One transformer block's MoE view. Tensor pointers are borrowed from the
 * underlying ingot handle and stay valid until poe_model_close(); any of
 * them may be NULL when the block does not have that tensor. */
typedef struct {
    uint32_t block;                    /* blk.<N>                            */
    int      is_moe;                   /* has routed expert tensors          */
    int      is_legacy_split;          /* per-expert tensors (old exports)   */
    uint32_t expert_count;             /* routed experts in this block       */

    const ingot_tensor *router_w;      /* blk.N.ffn_gate_inp.weight          */
    const ingot_tensor *router_b;      /* ...ffn_gate_inp.bias / exp_probs_b */
    const ingot_tensor *gate_exps_w;   /* packed [embd, ff, n_expert]        */
    const ingot_tensor *up_exps_w;
    const ingot_tensor *down_exps_w;   /* packed [ff, embd, n_expert]        */

    /* A split checkpoint (poe split) stores the experts a workload needs
     * least in a second tensor at a lower precision. They are routed
     * experts like any other and are counted as such; `is_split` says the
     * layer carries them, and expert_count is hot + cold. */
    const ingot_tensor *gate_exps_cold_w;
    const ingot_tensor *up_exps_cold_w;
    const ingot_tensor *down_exps_cold_w;
    int      is_split;
    uint32_t hot_expert_count;         /* experts in the wide tensor         */
    uint32_t cold_expert_count;        /* experts in the low-precision one   */
    const ingot_tensor *gate_exps_b;   /* per-expert biases (gpt-oss style)  */
    const ingot_tensor *up_exps_b;
    const ingot_tensor *down_exps_b;

    int      has_shared_expert;        /* ffn_*_shexp tensors present        */

    /* Exact on-disk bytes, per block. */
    uint64_t expert_bytes;             /* routed expert weights + biases     */
    uint64_t router_bytes;
    uint64_t shared_bytes;             /* shared-expert (shexp) tensors      */

    /* Exact parameter counts (tensor elements), per block. */
    uint64_t expert_params;            /* all routed experts together        */
} poe_block;

typedef struct {
    ingot_gguf *g;                     /* owned; closed by poe_model_close   */

    char     arch[64];                 /* general.architecture               */
    char     name[128];                /* general.name, "" when absent       */

    uint32_t n_blocks;                 /* <arch>.block_count (or derived)    */
    uint32_t n_moe_blocks;
    uint32_t expert_count;             /* <arch>.expert_count (or derived)   */
    uint32_t experts_per_token;        /* <arch>.expert_used_count           */
    uint32_t shared_expert_count;      /* <arch>.expert_shared_count         */
    uint32_t embedding_length;         /* <arch>.embedding_length            */
    uint32_t expert_ff_length;         /* <arch>.expert_feed_forward_length  */

    poe_block *blocks;                 /* n_blocks entries, indexed by block */

    /* Whole-model storage accounting. Exact on-disk payload bytes — never
     * estimates. total = expert + router + shared + other. */
    uint64_t total_bytes;
    uint64_t expert_bytes;
    uint64_t router_bytes;
    uint64_t shared_bytes;
    uint64_t other_bytes;

    /* Whole-model parameter accounting (tensor elements), exact. The
     * embeddings (token_embd / output head) are split out so active-param
     * figures can be reported with and without them.
     * total = expert + router + shared + embedding + other. */
    uint64_t total_params;
    uint64_t expert_params;            /* routed experts, all blocks         */
    uint64_t router_params;
    uint64_t shared_params;
    uint64_t embedding_params;
    uint64_t other_params;

    /* Structural fingerprint: FNV-1a over architecture, the full tensor
     * table (names, types, shapes, sizes), metadata keys/scalars, and the
     * first bytes of every tensor payload. Binds profiles and plans to a
     * specific checkpoint. Format "poe1:<16 hex>". */
    char fingerprint[24];
} poe_model;

/* Open a GGUF (single file or llama.cpp-style split; the split naming is
 * handled by ingot) and discover its MoE structure. On success *out owns a
 * handle the caller must close; on failure *out is NULL and err holds a
 * printable message. Returns 0 on success. */
int  poe_model_open(poe_model **out, const char *path, char *err, size_t errsz);
void poe_model_close(poe_model *m);

/* "25.7 GiB" style human formatting (binary units, one decimal). */
void poe_format_bytes(uint64_t bytes, char *dst, size_t dstsz);

/* "3.2B" / "245.1M" style parameter-count formatting (decimal units). */
void poe_format_params(uint64_t params, char *dst, size_t dstsz);

#ifdef __cplusplus
}
#endif
#endif
