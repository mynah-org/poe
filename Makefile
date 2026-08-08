# poe — Pruning & Optimization of Experts. C11, no dependencies; the ingot
# amalgam is vendored in third_party/ingot/.
#
# SPDX-License-Identifier: MIT

CC      ?= cc
CFLAGS  ?= -O2
WARN     = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wvla
INCLUDE  = -Iinclude -Ithird_party/ingot
LDLIBS   = -lpthread -lm

LIB_SRC  = src/model.c src/fmt.c
CLI_SRC  = src/cli/main.c src/cli/cmd_inspect.c src/cli/cmd_experts.c \
           src/cli/cmd_budget.c
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
test: build/test_model poe | build
	./build/test_model
	@echo "── poe inspect (smoke) ──"
	./poe inspect build/fixture-moe.gguf
	./poe inspect build/fixture-moe.gguf --json > /dev/null
	./poe experts build/fixture-moe.gguf --layer 1
	./poe experts build/fixture-moe.gguf --json > /dev/null
	./poe routing-budget build/fixture-moe.gguf
	./poe routing-budget build/fixture-moe.gguf --json > /dev/null

build/test_model: tests/test_model.o tests/fixture.o src/model.o src/fmt.o \
                  third_party/ingot/ingot.o | build
	$(CC) $(WARN) $(CFLAGS) $^ $(LDLIBS) -o $@

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
