/* fmt.c — human formatting shared by the CLI.
 * SPDX-License-Identifier: MIT */
#include "poe/poe.h"

#include <stdio.h>

void poe_format_bytes(uint64_t bytes, char *dst, size_t dstsz) {
    static const char *const unit[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)bytes;
    size_t u = 0;
    while (v >= 1024.0 && u + 1 < sizeof unit / sizeof unit[0]) { v /= 1024.0; u++; }
    if (u == 0) snprintf(dst, dstsz, "%llu B", (unsigned long long)bytes);
    else        snprintf(dst, dstsz, "%.1f %s", v, unit[u]);
}

void poe_format_params(uint64_t params, char *dst, size_t dstsz) {
    if      (params >= 1000000000ull)
        snprintf(dst, dstsz, "%.2fB", (double)params / 1e9);
    else if (params >= 1000000ull)
        snprintf(dst, dstsz, "%.1fM", (double)params / 1e6);
    else if (params >= 1000ull)
        snprintf(dst, dstsz, "%.1fK", (double)params / 1e3);
    else
        snprintf(dst, dstsz, "%llu", (unsigned long long)params);
}
