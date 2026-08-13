/* residency.c — fit a checkpoint into a VRAM budget and say what to run.
 *
 * The arithmetic is deliberately small: the resident floor (everything that
 * is not a routed expert, plus the KV cache and a compute allowance) is
 * subtracted from the budget, and routed expert slabs are kept on the device
 * until the remainder runs out. What is left over goes to the host.
 *
 * The interesting decision is only *which* slabs leave, and the honest
 * answer for a uniform MoE is "it does not matter": every layer runs the
 * same K experts of the same size, so the traffic a displaced layer creates
 * is the same wherever it sits. The exception is a split checkpoint, where
 * hot and cold are separate tensors and the cold one is, by construction,
 * the half the workload routes to least.
 *
 * SPDX-License-Identifier: MIT */
#include "poe/residency.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

static void warn(poe_residency *r, const char *msg) {
    if (r->n_warnings < POE_RESIDENCY_MAX_WARN)
        snprintf(r->warnings[r->n_warnings++], sizeof r->warnings[0], "%s", msg);
}

/* ── model metadata the MoE view does not carry ─────────────────────────── */

/* "<arch>.<suffix>" as an unsigned scalar; arrays read their first element,
 * which is what a per-layer head count degenerates to on every architecture
 * POE handles today. Returns 0 on success. */
static int arch_kv_u64(const ingot_gguf *g, const char *arch,
                       const char *suffix, uint64_t *out) {
    char key[192];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    if (kv == NULL) return -1;
    if (ingot_kv_type(kv) == INGOT_KV_ARRAY) {
        uint64_t len = 0;
        int64_t  v   = 0;
        if (ingot_kv_arr_len(kv, &len) != 0 || len == 0) return -1;
        if (ingot_kv_arr_i64(kv, 0, &v) != 0 || v < 0) return -1;
        *out = (uint64_t)v;
        return 0;
    }
    return ingot_kv_u64(kv, out);
}

/* Bytes per stored KV element. K-quant cache types are not offered because
 * llama.cpp does not accept them for the cache; the three below are the ones
 * -ctk/-ctv actually take in practice. */
static double kv_elem_bytes(const char *type) {
    if (type == NULL)             return 2.0;
    if (strcmp(type, "f16")  == 0) return 2.0;
    if (strcmp(type, "bf16") == 0) return 2.0;
    if (strcmp(type, "f32")  == 0) return 4.0;
    if (strcmp(type, "q8_0") == 0) return 34.0 / 32.0;
    if (strcmp(type, "q4_0") == 0) return 18.0 / 32.0;
    return -1.0;
}

/* ctx · Σ_caching_layers (n_embd_k_gqa + n_embd_v_gqa) · bytes_per_element,
 * with both widths read off the attention tensors themselves.
 *
 * Counting layers from the tensor table rather than from block_count is not
 * pedantry: a hybrid checkpoint caches only its attention layers, and the
 * rest carry a recurrent state instead. Qwen3.6-35B-A3B is exactly that —
 * 10 attention blocks of 40, the other 30 holding `ssm_*` tensors — so the
 * metadata formula overstates its KV cache by 4x. The tensor table knows;
 * the block count does not.
 *
 * Still an estimate, and labelled as one: llama.cpp pads the context, may
 * keep a shorter window on sliding-attention layers, sizes the cache per
 * sequence, and holds recurrent state this does not model. */
static uint64_t kv_cache_bytes_from_tensors(const poe_model *m, uint32_t ctx,
                                            double elem_bytes,
                                            poe_residency *r, int *ok) {
    uint64_t gqa_sum = 0;
    uint32_t n_attn = 0, n_ssm = 0;

    for (uint32_t b = 0; b < m->n_blocks; b++) {
        char name[96];
        snprintf(name, sizeof name, "blk.%u.attn_k.weight", b);
        const ingot_tensor *k = ingot_gguf_find(m->g, name);
        snprintf(name, sizeof name, "blk.%u.attn_v.weight", b);
        const ingot_tensor *v = ingot_gguf_find(m->g, name);
        snprintf(name, sizeof name, "blk.%u.ssm_conv1d.weight", b);
        if (ingot_gguf_find(m->g, name) != NULL) n_ssm++;

        if (k == NULL || v == NULL || k->rank < 2 || v->rank < 2) continue;
        n_attn++;
        gqa_sum += k->ne[1] + v->ne[1];
    }

    if (n_attn == 0) { *ok = 0; return 0; }
    *ok = 1;

    if (n_ssm) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "hybrid checkpoint: %u attention blocks cache, %u recurrent "
                 "ones hold a state this estimate does not model", n_attn, n_ssm);
        warn(r, msg);
    } else if (n_attn < m->n_blocks) {
        char msg[160];
        snprintf(msg, sizeof msg, "%u of %u blocks carry attention tensors; "
                 "only those are counted", n_attn, m->n_blocks);
        warn(r, msg);
    }
    return (uint64_t)((double)ctx * (double)gqa_sum * elem_bytes);
}

/* The metadata route, used only when the tensor table cannot answer. */
static uint64_t kv_cache_bytes(const poe_model *m, uint32_t ctx,
                               double elem_bytes, poe_residency *r) {
    const char *arch = m->arch;
    uint64_t n_kv_head = 0, k_len = 0, v_len = 0, n_head = 0;

    int ok = 0;
    const uint64_t exact = kv_cache_bytes_from_tensors(m, ctx, elem_bytes, r, &ok);
    if (ok) return exact;

    if (arch_kv_u64(m->g, arch, "attention.head_count_kv", &n_kv_head) != 0 ||
        n_kv_head == 0) {
        warn(r, "no attention.head_count_kv: the KV estimate assumes 8 heads");
        n_kv_head = 8;
    }
    if (arch_kv_u64(m->g, arch, "attention.key_length", &k_len) != 0 || k_len == 0) {
        if (arch_kv_u64(m->g, arch, "attention.head_count", &n_head) == 0 &&
            n_head != 0 && m->embedding_length != 0) {
            k_len = m->embedding_length / n_head;
        } else {
            warn(r, "no attention.key_length: the KV estimate assumes 128");
            k_len = 128;
        }
    }
    if (arch_kv_u64(m->g, arch, "attention.value_length", &v_len) != 0 || v_len == 0)
        v_len = k_len;

    char probe[192];
    snprintf(probe, sizeof probe, "%s.attention.key_length_swa", arch);
    if (ingot_gguf_kv_find(m->g, probe) != NULL)
        warn(r, "sliding-window attention: the real KV cache is smaller than "
                "this estimate");

    const double per_layer = (double)ctx * (double)n_kv_head *
                             (double)(k_len + v_len) * elem_bytes;
    return (uint64_t)(per_layer * (double)m->n_blocks);
}

/* ── ranking ────────────────────────────────────────────────────────────── */

/* How many experts carry 90% of a layer's routed selections. The narrower
 * that set, the more a host-resident slab re-reads the same rows — the only
 * per-layer quantity that plausibly matters once quality is off the table.
 * Returns the layer's expert count when the profile cannot answer. */
static uint32_t workset_90(const poe_profile *pr, uint32_t layer,
                           uint32_t n_experts) {
    if (pr == NULL || pr->sel_count == NULL || layer >= pr->n_layers)
        return n_experts;
    const uint32_t E = pr->n_experts < n_experts ? pr->n_experts : n_experts;
    if (E == 0) return n_experts;

    const uint64_t *row = pr->sel_count + (size_t)layer * pr->n_experts;
    uint64_t total = 0;
    for (uint32_t e = 0; e < E; e++) total += row[e];
    if (total == 0) return n_experts;

    /* selection-sort the top of the distribution: E is a few hundred and
     * this runs once per layer, so the simplest correct thing wins */
    uint64_t seen = 0, cut = (total * 9) / 10;
    uint32_t taken = 0;
    unsigned char *used = calloc(E, 1);
    if (used == NULL) return n_experts;
    while (seen < cut && taken < E) {
        uint32_t best = E;
        for (uint32_t e = 0; e < E; e++)
            if (!used[e] && (best == E || row[e] > row[best])) best = e;
        if (best == E) break;
        used[best] = 1;
        seen += row[best];
        taken++;
    }
    free(used);
    return taken ? taken : n_experts;
}

static const char *rank_name(poe_residency_rank rank) {
    switch (rank) {
        case POE_RESIDENCY_FIRST:        return "first";
        case POE_RESIDENCY_WORKSET:      return "workset";
        case POE_RESIDENCY_WORKSET_INV:  return "workset-inverted";
        case POE_RESIDENCY_LAST:         break;
    }
    return "last";
}

/* Descending by score, ties by unit order, so the plan is deterministic. */
static int unit_cmp(const void *pa, const void *pb) {
    const poe_residency_unit *a = pa, *b = pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return  1;
    if (a->block != b->block) return a->block < b->block ? -1 : 1;
    return a->is_cold > b->is_cold ? -1 : 1;
}

/* ── build ──────────────────────────────────────────────────────────────── */

int poe_residency_build(poe_residency **out, const poe_model *m,
                        const poe_residency_opts *o, char *err, size_t errsz) {
    if (out == NULL || m == NULL || o == NULL)
        return fail(err, errsz, "residency: null argument");
    *out = NULL;
    if (o->vram_bytes == 0)
        return fail(err, errsz, "residency: a VRAM budget is required");
    if (m->n_moe_blocks == 0)
        return fail(err, errsz, "residency: no MoE structure in this model — "
                                "there is nothing to place separately");

    const double elem = kv_elem_bytes(o->kv_type);
    if (elem < 0)
        return fail(err, errsz, "residency: unknown cache type "
                                "(f16, bf16, f32, q8_0, q4_0)");

    const int wants_profile = o->rank == POE_RESIDENCY_WORKSET ||
                              o->rank == POE_RESIDENCY_WORKSET_INV;
    if (wants_profile && o->profile == NULL)
        return fail(err, errsz, "residency: this ranking needs --profile");
    /* A profile of the *source* is the normal case on a derived checkpoint:
     * the profile is what drove the derivation. `poe split` and friends
     * record where they came from, so accept that lineage rather than making
     * every such call pass --force, which would train the reflex of passing
     * it when the mismatch is real. */
    int via_source = 0;
    if (o->profile != NULL &&
        strcmp(o->profile->fingerprint, m->fingerprint) != 0) {
        const ingot_kv *kv = ingot_gguf_kv_find(m->g, "poe.source_fingerprint");
        const char *src = NULL;
        if (kv != NULL && ingot_kv_str(kv, &src) == 0 && src != NULL &&
            strcmp(o->profile->fingerprint, src) == 0)
            via_source = 1;
        else if (!o->force)
            return fail(err, errsz, "residency: the profile was captured from "
                                    "a different checkpoint (--force to "
                                    "override)");
    }

    poe_residency *r = calloc(1, sizeof *r);
    if (r == NULL) return fail(err, errsz, "residency: out of memory");
    if (via_source)
        warn(r, "the profile is of the checkpoint this one was derived from");

    snprintf(r->poe_version, sizeof r->poe_version, "%s", POE_VERSION);
    snprintf(r->model_fingerprint, sizeof r->model_fingerprint, "%s", m->fingerprint);
    snprintf(r->arch, sizeof r->arch, "%s", m->arch);
    snprintf(r->rank_method, sizeof r->rank_method, "%s", rank_name(o->rank));
    snprintf(r->kv_type, sizeof r->kv_type, "%s", o->kv_type ? o->kv_type : "f16");

    r->vram_bytes    = o->vram_bytes;
    r->model_bytes   = m->total_bytes;
    r->expert_bytes  = m->expert_bytes;
    r->nonexpert_bytes = m->total_bytes - m->expert_bytes;
    r->kv_bytes_per_elem = elem;
    r->reserve_bytes = o->reserve_bytes ? o->reserve_bytes
                                        : POE_RESIDENCY_DEFAULT_RESERVE;

    uint64_t ctx = o->ctx;
    if (ctx == 0) {
        uint64_t trained = 0;
        if (arch_kv_u64(m->g, m->arch, "context_length", &trained) == 0 && trained)
            ctx = trained;
        else {
            ctx = 4096;
            warn(r, "no context_length in the checkpoint: the KV estimate "
                    "assumes 4096");
        }
    }
    if (ctx > 0xFFFFFFFFULL) ctx = 0xFFFFFFFFULL;
    r->ctx      = (uint32_t)ctx;
    r->kv_bytes = kv_cache_bytes(m, r->ctx, elem, r);

    /* ── the placeable units ────────────────────────────────────────────── */

    uint32_t n = 0;
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (!blk->is_moe || blk->expert_bytes == 0) continue;
        n += blk->is_split ? 2u : 1u;
        if (blk->is_split) r->is_split = 1;
    }
    if (n == 0) {
        free(r);
        return fail(err, errsz, "residency: no routed expert tensors to place");
    }

    r->units = calloc(n, sizeof *r->units);
    if (r->units == NULL) { free(r); return fail(err, errsz, "residency: out of memory"); }
    r->n_units = n;

    uint32_t u = 0;
    for (uint32_t b = 0; b < m->n_blocks; b++) {
        const poe_block *blk = &m->blocks[b];
        if (!blk->is_moe || blk->expert_bytes == 0) continue;

        if (blk->is_split) {
            /* Split the block's exact byte total between the halves by the
             * bytes their own tensors carry, so the two units still add up
             * to the block and no byte is invented. */
            uint64_t cold = 0;
            if (blk->gate_exps_cold_w) cold += blk->gate_exps_cold_w->nbytes;
            if (blk->up_exps_cold_w)   cold += blk->up_exps_cold_w->nbytes;
            if (blk->down_exps_cold_w) cold += blk->down_exps_cold_w->nbytes;
            if (cold > blk->expert_bytes) cold = blk->expert_bytes;

            r->units[u].block   = b;
            r->units[u].is_cold = 1;
            r->units[u].bytes   = cold;
            r->units[u].experts = blk->cold_expert_count;
            u++;
            r->units[u].block   = b;
            r->units[u].is_cold = 0;
            r->units[u].bytes   = blk->expert_bytes - cold;
            r->units[u].experts = blk->hot_expert_count;
            u++;
        } else {
            r->units[u].block   = b;
            r->units[u].bytes   = blk->expert_bytes;
            r->units[u].experts = blk->expert_count;
            u++;
        }
    }

    /* Scores. The cold half of a split always outranks everything for
     * eviction: it holds the experts the workload needs least, so displacing
     * it is strictly less traffic than displacing an arbitrary layer. The
     * +1000 keeps that dominance whatever the per-layer scores end up being. */
    for (uint32_t i = 0; i < r->n_units; i++) {
        poe_residency_unit *un = &r->units[i];
        double s = 0;
        switch (o->rank) {
            case POE_RESIDENCY_FIRST:
                s = (double)(m->n_blocks - un->block);
                break;
            case POE_RESIDENCY_WORKSET:
                s = -(double)workset_90(o->profile, un->block, un->experts);
                break;
            case POE_RESIDENCY_WORKSET_INV:
                s = (double)workset_90(o->profile, un->block, un->experts);
                break;
            case POE_RESIDENCY_LAST:
            default:
                s = (double)un->block;
                break;
        }
        un->score = s + (un->is_cold ? 1000.0 : 0.0);
    }
    qsort(r->units, r->n_units, sizeof *r->units, unit_cmp);

    /* ── fit ────────────────────────────────────────────────────────────── */

    const uint64_t floor_bytes = r->nonexpert_bytes + r->kv_bytes + r->reserve_bytes;
    if (floor_bytes >= r->vram_bytes) {
        r->fits_at_all   = 0;
        r->shortfall_bytes = floor_bytes - r->vram_bytes;
        r->expert_bytes_cpu = r->expert_bytes;
        for (uint32_t i = 0; i < r->n_units; i++) r->units[i].on_cpu = 1;
        warn(r, "the resident floor alone exceeds the budget: no placement of "
                "experts can make this fit");
        *out = r;
        return 0;
    }
    r->fits_at_all = 1;

    const uint64_t headroom = r->vram_bytes - floor_bytes;
    uint64_t on_gpu = r->expert_bytes;
    for (uint32_t i = 0; i < r->n_units && on_gpu > headroom; i++) {
        r->units[i].on_cpu = 1;
        on_gpu -= r->units[i].bytes;
    }
    r->expert_bytes_gpu = on_gpu;
    r->expert_bytes_cpu = r->expert_bytes - on_gpu;
    r->fits_entirely    = r->expert_bytes_cpu == 0;

    if (!r->fits_entirely && o->rank == POE_RESIDENCY_LAST && o->profile == NULL)
        warn(r, "no profile: the eviction order is llama.cpp's own convention, "
                "not a measured one");

    *out = r;
    return 0;
}

void poe_residency_free(poe_residency *r) {
    if (r == NULL) return;
    free(r->units);
    free(r);
}

/* ── output ─────────────────────────────────────────────────────────────── */

uint32_t poe_residency_ncpumoe(const poe_residency *r) {
    if (r == NULL || r->expert_bytes_cpu == 0) return 0;

    /* Equivalent only when the displaced set is exactly blocks 0..N-1 and
     * every displaced block left whole. A split checkpoint whose cold halves
     * went to the host is never expressible this way. */
    uint32_t hi = 0;
    int seen_cpu = 0;
    for (uint32_t i = 0; i < r->n_units; i++) {
        if (!r->units[i].on_cpu) continue;
        seen_cpu = 1;
        if (r->units[i].block + 1 > hi) hi = r->units[i].block + 1;
    }
    if (!seen_cpu) return 0;
    for (uint32_t i = 0; i < r->n_units; i++) {
        const int should = r->units[i].block < hi;
        if (should != (r->units[i].on_cpu != 0)) return 0;
    }
    return hi;
}

/* snprintf that keeps accumulating the would-be length after the buffer is
 * full, so the caller gets the same "how much room did I need" contract. */
static void app(char *buf, size_t bufsz, size_t *off, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (*off < bufsz) {
        int w = vsnprintf(buf + *off, bufsz - *off, fmt, ap);
        if (w > 0) *off += (size_t)w;
    } else {
        char sink[1];
        int w = vsnprintf(sink, 0, fmt, ap);
        if (w > 0) *off += (size_t)w;
        (void)sink;
    }
    va_end(ap);
}

int poe_residency_flags(const poe_residency *r, char *buf, size_t bufsz) {
    if (r == NULL || buf == NULL || bufsz == 0) return -1;
    buf[0] = '\0';

    size_t off = 0;
    app(buf, bufsz, &off, "-ngl 99");
    if (r->expert_bytes_cpu == 0) return (int)off;

    /* One -ot per tensor class, so a split checkpoint can send its cold
     * tensors to the host without dragging the hot ones along. llama.cpp
     * matches these with regex_search against the full tensor name, so the
     * block number is anchored on both sides and `_exps\.weight` cannot
     * match `_exps_cold.weight`. */
    uint32_t maxb = 0;
    for (uint32_t i = 0; i < r->n_units; i++)
        if (r->units[i].block > maxb) maxb = r->units[i].block;

    /* Ascending block order in the pattern: the units are sorted by eviction
     * priority, which is not an order anyone wants to read in a regex. */
    for (int cold = 0; cold <= 1; cold++) {
        uint32_t have = 0, moved = 0;
        for (uint32_t i = 0; i < r->n_units; i++) {
            if (r->units[i].is_cold != cold) continue;
            have++;
            moved += r->units[i].on_cpu != 0;
        }
        if (moved == 0) continue;

        /* Whole class displaced: say so, instead of enumerating 40 blocks. */
        if (moved == have) {
            app(buf, bufsz, &off, " -ot \"blk\\.[0-9]+\\.ffn_(gate|up|down)"
                                  "_exps%s\\.weight=CPU\"", cold ? "_cold" : "");
            continue;
        }

        int first = 1;
        for (uint32_t b = 0; b <= maxb; b++) {
            int hit = 0;
            for (uint32_t i = 0; i < r->n_units && !hit; i++)
                hit = r->units[i].on_cpu && r->units[i].is_cold == cold &&
                      r->units[i].block == b;
            if (!hit) continue;
            if (first) app(buf, bufsz, &off, " -ot \"blk\\.(");
            app(buf, bufsz, &off, "%s%u", first ? "" : "|", b);
            first = 0;
        }
        if (!first)
            app(buf, bufsz, &off, ")\\.ffn_(gate|up|down)_exps%s\\.weight=CPU\"",
                cold ? "_cold" : "");
    }
    return (int)off;
}
