# poe — Pruning & Optimization of Experts. C11, no dependencies; the ingot
# amalgam is vendored in third_party/ingot/.
#
# SPDX-License-Identifier: MIT

CC      ?= cc
CFLAGS  ?= -O2
WARN     = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wvla
INCLUDE  = -Iinclude -Ithird_party/ingot
LDLIBS   = -lpthread -lm

LIB_SRC  = src/model.c src/fmt.c src/profiler/accum.c src/profiler/stability.c \
           src/json.c src/profile.c src/plan.c
CLI_SRC  = src/cli/main.c src/cli/cmd_inspect.c src/cli/cmd_experts.c \
           src/cli/cmd_budget.c src/cli/cmd_compare.c src/cli/cmd_plan.c \
           src/cli/cmd_estimate.c src/cli/cmd_diff.c
INGOT    = third_party/ingot/ingot.c

SRC      = $(LIB_SRC) $(CLI_SRC) $(INGOT)
OBJ      = $(SRC:.c=.o)

.PHONY: all test tools clean help
all: poe

## help: this list
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  make /'

## all: build the poe CLI
poe: $(OBJ)
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

%.o: %.c
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) -MMD -MP -c $< -o $@
-include $(SRC:.c=.d) tests/fixture.d tests/test_model.d tools/poe_mkfixture.d

build:
	mkdir -p build

## test: synthetic-fixture tests + a CLI smoke run (no model downloads)
test: build/test_model build/test_accum build/test_compare build/test_plan poe | build
	./build/test_model
	./build/test_accum
	./build/test_compare
	./build/test_plan
	@echo "── poe inspect (smoke) ──"
	./poe inspect build/fixture-moe.gguf
	./poe inspect build/fixture-moe.gguf --json > /dev/null
	./poe experts build/fixture-moe.gguf --layer 1
	./poe experts build/fixture-moe.gguf --json > /dev/null
	./poe routing-budget build/fixture-moe.gguf
	./poe routing-budget build/fixture-moe.gguf --json > /dev/null
	./poe compare build/pa.poeprofile build/pb.poeprofile
	./poe compare build/pa.poeprofile build/pb.poeprofile --json > /dev/null
	./poe plan build/plan-fix.gguf --profile build/plan.poeprofile --method reap --prune 25% -o build/smoke.poeplan
	./poe estimate build/smoke.poeplan build/plan-fix.gguf
	./poe plan build/plan-fix.gguf --profile build/plan.poeprofile --method frequency --prune 50% -o build/smoke50.poeplan > /dev/null
	./poe diff build/smoke.poeplan build/smoke50.poeplan
	./poe diff build/pa.poeprofile build/pb.poeprofile
	./poe diff build/fixture-moe.gguf build/fixture-moe-seed9.gguf

build/test_model: tests/test_model.o tests/fixture.o src/model.o src/fmt.o \
                  third_party/ingot/ingot.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

build/test_accum: tests/test_accum.o src/profiler/accum.o src/profiler/stability.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

build/test_compare: tests/test_compare.o src/json.o src/profile.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

build/test_plan: tests/test_plan.o tests/fixture.o src/plan.o src/profile.o \
                 src/json.o src/model.o src/fmt.o third_party/ingot/ingot.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

## profiler: build/poe-profile — MoE router profiler over llama.cpp (M2).
## Needs a llama.cpp checkout built with shared libs:
##   make profiler LLAMA_DIR=~/llama.cpp
LLAMA_DIR ?= ../llama.cpp
LLAMA_INC  = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_LIB  = -L$(LLAMA_DIR)/build/bin -lllama -lggml -lggml-base \
             -Wl,-rpath,$(LLAMA_DIR)/build/bin

profiler: build/poe-profile
build/poe-profile: tools/poe_profile.c src/profiler/accum.o \
                   src/profiler/stability.o src/model.o src/fmt.o \
                   third_party/ingot/ingot.o | build
	$(CC) $(WARN) $(CFLAGS) $(INCLUDE) $(LLAMA_INC) $< \
	      src/profiler/accum.o src/profiler/stability.o src/model.o src/fmt.o \
	      third_party/ingot/ingot.o $(LLAMA_LIB) $(LDLIBS) -o $@

## tools: build/poe-mkfixture (synthetic GGUF generator)
tools: build/poe-mkfixture
build/poe-mkfixture: tools/poe_mkfixture.o tests/fixture.o \
                     third_party/ingot/ingot.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

## clean: remove objects and binaries
clean:
	rm -f $(OBJ) $(SRC:.c=.d) poe
	rm -f tests/*.o tests/*.d tools/*.o tools/*.d
	rm -rf build
