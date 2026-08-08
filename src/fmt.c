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
