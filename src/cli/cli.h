/* cli.h — command entry points and small shared helpers.
 * SPDX-License-Identifier: MIT */
#ifndef POE_CLI_H
#define POE_CLI_H

#include "poe/poe.h"

int poe_cmd_inspect(int argc, char **argv);
int poe_cmd_experts(int argc, char **argv);
int poe_cmd_routing_budget(int argc, char **argv);
int poe_cmd_compare(int argc, char **argv);
int poe_cmd_plan(int argc, char **argv);
int poe_cmd_estimate(int argc, char **argv);
int poe_cmd_diff(int argc, char **argv);
int poe_cmd_apply(int argc, char **argv);

/* Minimal JSON string escaping ("\ and control chars) into a fixed buffer. */
void poe_json_escape(const char *src, char *dst, size_t dstsz);

#endif
