/* poe/requant.h — per-expert precision, emulated inside one tensor type.
 *
 * M9's actual thesis is per expert: quantize hard the experts a workload
 * never routes to, keep the hot ones wide. A GGUF tensor carries exactly
 * one type and routed experts are packed one tensor per projection, so that
 * allocation cannot be *stored* — reaching it needs the slab split in two
 * and a runtime patch on the hot path. M9a tested the only granularity the
 * format allows (one type per slab) and lost to uniform, which says nothing
 * about the per-expert question because no arm there could express it.
 *
 * This is how the question gets answered before the runtime work is paid
 * for. Every expert's weights are pushed through a quantizer and back, but
 * only the *cold* ones go through the hard type; the whole slab is then
 * stored at the carrier type. The file is exactly the size of a uniform
 * carrier quant, so it runs today, unpatched — and its quality is that of
 * the mixed allocation, because the information the hard type destroyed is
 * gone before the carrier ever sees it.
 *
 * The experiment that follows is byte-exact by construction: run this with
 * `degrade_fraction` f at the hard type, and again with f = 1 at whichever
 * type has the same average bits per weight. Both outputs are the same size
 * (the carrier's), both carry one extra carrier round trip, and the only
 * difference is whether the damage was spread evenly or aimed at the cold
 * experts. If aiming does not win here, it will not win in a patched
 * runtime either, and M9b is answered for the cost of two quantize runs.
 *
 * Two honest limits to state with any number this produces: ingot's
 * encoders are unweighted least squares where llama.cpp's are imatrix
 * weighted, and the emulation pays one carrier round trip that a real
 * per-expert artifact would not. Both apply equally to every arm.
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_REQUANT_H
#define POE_REQUANT_H

#include <stddef.h>
#include <stdint.h>

#include "poe/poe.h"
#include "poe/profile.h"

typedef struct {
    int    carrier_type;        /* ingot type every routed slab is stored as */
    int    degrade_type;        /* the type cold experts are pushed through  */
    double degrade_fraction;    /* per layer, in (0,1]; 1.0 is the control   */
    const poe_profile *profile; /* ranking source; NULL only when f == 1.0   */
    int    invert;              /* degrade the HOTTEST experts instead       */
    int    force;               /* accept a profile from another checkpoint  */
    int    threads;             /* 0 = one per online core                   */
} poe_requant_opts;

typedef struct {
    uint32_t slabs_rewritten;
    uint32_t tensors_total;
    uint32_t degraded_per_layer;
    uint64_t experts_degraded;
    uint64_t bytes_written;
    double   emulated_bits;     /* average bits/weight the map emulates      */
    double   carrier_bits, degrade_bits;
    uint32_t threads;
    int      ranked_by_reap;    /* 0 = selection counts, 1 = REAP saliency   */
} poe_requant_stats;

/* Rewrite `m` into `out_path`. Non-routed tensors are copied verbatim, so
 * the output is a normal checkpoint any runtime loads. Returns 0, or -1
 * with a message in `err`. */
int poe_requant(const poe_model *m, const poe_requant_opts *o,
                const char *out_path, poe_requant_stats *stats,
                char *err, size_t errsz);

/* Bits per weight of an ingot type, from its block geometry. 0 if unknown. */
double poe_type_bits(int type);

#endif
