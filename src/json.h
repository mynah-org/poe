/* json.h — minimal JSON reader for POE's own artifacts (.poeprofile,
 * later .poeplan). Strict enough for interchange, small enough to audit:
 * a recursive-descent parser building a plain DOM. Not a general-purpose
 * library — no streaming, no comments, numbers as double (plus a separate
 * exact-u64 accessor for counters).
 *
 * SPDX-License-Identifier: MIT */
#ifndef POE_JSON_H
#define POE_JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    POE_JSON_NULL, POE_JSON_BOOL, POE_JSON_NUM,
    POE_JSON_STR,  POE_JSON_ARR,  POE_JSON_OBJ
} poe_json_type;

typedef struct poe_json poe_json;

/* Parse text[0..len). On failure returns NULL with a message (including
 * the byte offset) in err. */
poe_json *poe_json_parse(const char *text, size_t len, char *err, size_t errsz);
void      poe_json_free(poe_json *j);

poe_json_type poe_json_kind(const poe_json *j);

/* Object/array access. Return NULL when missing / wrong kind / OOB. */
const poe_json *poe_json_get(const poe_json *obj, const char *key);
const poe_json *poe_json_at(const poe_json *arr, size_t index);
size_t          poe_json_len(const poe_json *j);   /* arr/obj element count */

/* Scalar accessors with defaults for missing/mistyped nodes. */
double      poe_json_num(const poe_json *j, double dflt);
uint64_t    poe_json_u64(const poe_json *j, uint64_t dflt);  /* exact for counters */
const char *poe_json_str(const poe_json *j, const char *dflt);
/* JSON true/false. Numbers do not answer here and booleans do not answer to
 * poe_json_num: a bool read through the number accessor silently returns the
 * default, which is how a round-tripped flag turns itself off. */
int         poe_json_bool(const poe_json *j, int dflt);

#endif
