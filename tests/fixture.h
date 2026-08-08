/* fixture.h — synthetic GGUF fixtures, shared by the tests and the
 * poe-mkfixture tool. Tiny (a few hundred KB), deterministic, valid GGUF v3;
 * CI never downloads a model.
 *
 * The MoE fixture mimics a llama.cpp qwen3moe export: packed
 * ffn_{gate,up,down}_exps tensors, a ffn_gate_inp router per block, attention
 * stubs so the "other" bucket is non-empty.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_FIXTURE_H
#define POE_FIXTURE_H

#include <stddef.h>
#include <stdint.h>

/* MoE fixture geometry — tests derive their expected byte counts from these,
 * so a change here fails loudly rather than silently. */
#define POE_FIX_ARCH    "qwen3moe"
#define POE_FIX_BLOCKS  4u
#define POE_FIX_EMBD    32u
#define POE_FIX_VOCAB   96u
#define POE_FIX_FF      16u
#define POE_FIX_EXPERTS 8u
#define POE_FIX_TOPK    2u

/* Write the fixtures. `seed` perturbs the tensor payloads (same structure,
 * different weights) — two files from the same seed are byte-identical.
 * Return 0 on success, -1 with a message in err. */
int poe_fixture_moe(const char *path, uint32_t seed, char *err, size_t errsz);
int poe_fixture_dense(const char *path, char *err, size_t errsz);

#endif
