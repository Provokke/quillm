# quillm build. `make` builds, `make test` runs the full core suite.
CC      ?= cc
PYTHON  ?= python3
BUILD   ?= build
OPT     ?= -O2
CFLAGS_BASE = -std=c11 -Wall -Wextra -Werror -Wshadow -Wvla -D_DEFAULT_SOURCE -g $(OPT) $(EXTRA_CFLAGS)
LDLIBS  = -lm -lpthread $(EXTRA_LDFLAGS)
ARCH := $(shell uname -m)

ifeq ($(ARCH),x86_64)
AVX2_FLAGS   = -mavx2 -mfma -mf16c
AVX512_FLAGS = -mavx512f -mavx512bw -mavx512vl -mavx2 -mfma -mf16c
endif

LIB_SRC = src/fp16.c src/quant.c src/arena.c src/cpu.c src/threadpool.c src/ops.c src/qlm_file.c \
          src/tokenizer.c src/sampler.c src/model.c \
          src/kernels_scalar.c src/kernels_avx2.c src/kernels_avx512.c src/kernels_neon.c
LIB_OBJ = $(patsubst src/%.c,$(BUILD)/obj/%.o,$(LIB_SRC))
LIB     = $(BUILD)/libquillm.a

TESTS   = test_fp16 test_quant test_kernels test_threads test_tokenizer test_sampler test_format test_model test_noalloc
TEST_BINS = $(addprefix $(BUILD)/,$(TESTS))
FIXTURES = $(BUILD)/fixtures/.stamp

.PHONY: all test test-c test-py asan tsan valgrind bench neon-check clean fixtures

all: $(BUILD)/quillm $(BUILD)/quillm-bench $(TEST_BINS)

$(BUILD)/obj/%.o: src/%.c src/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BASE) -c $< -o $@

$(BUILD)/obj/kernels_avx2.o: src/kernels_avx2.c src/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BASE) $(AVX2_FLAGS) -c $< -o $@

$(BUILD)/obj/kernels_avx512.o: src/kernels_avx512.c src/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BASE) $(AVX512_FLAGS) -c $< -o $@

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^

$(BUILD)/quillm: src/main.c $(LIB)
	$(CC) $(CFLAGS_BASE) -Isrc $< $(LIB) -o $@ $(LDLIBS)

$(BUILD)/quillm-bench: bench/bench.c $(LIB)
	$(CC) $(CFLAGS_BASE) -Isrc $< $(LIB) -o $@ $(LDLIBS)

$(BUILD)/test_noalloc: tests/test_noalloc.c tests/test.h $(LIB)
	$(CC) $(CFLAGS_BASE) -Isrc $< $(LIB) -o $@ $(LDLIBS) -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=aligned_alloc

$(BUILD)/test_%: tests/test_%.c tests/test.h $(LIB)
	$(CC) $(CFLAGS_BASE) -Isrc $< $(LIB) -o $@ $(LDLIBS)

# numpy-generated models, reference logits and tokenizer vectors
$(FIXTURES): tools/qlm.py tests/gen_fixtures.py
	$(PYTHON) tests/gen_fixtures.py $(BUILD)/fixtures
	@touch $@
fixtures: $(FIXTURES)

test-c: all $(FIXTURES)
	@set -e; for t in $(TESTS); do echo "== $$t"; QM_FIXTURES=$(BUILD)/fixtures $(BUILD)/$$t; done
	@QM_BIN=$(BUILD)/quillm QM_FIXTURES=$(BUILD)/fixtures sh tests/test_cli.sh

test-py: $(FIXTURES) $(BUILD)/quillm
	QM_BIN=$(BUILD)/quillm QM_FIXTURES=$(BUILD)/fixtures $(PYTHON) -m pytest -q tests/test_python.py

test: test-c test-py
	@echo "ALL TESTS PASSED"

# AddressSanitizer + UndefinedBehaviorSanitizer build of the whole suite
asan:
	$(MAKE) test-c BUILD=build-asan OPT=-O1 \
	  EXTRA_CFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" \
	  EXTRA_LDFLAGS="-fsanitize=address,undefined"

# valgrind memcheck (valgrind hides AVX-512 via cpuid, so AVX2/scalar paths are exercised)
valgrind: all $(FIXTURES)
	@set -e; for t in $(TESTS); do echo "== valgrind $$t"; \
	  QM_FIXTURES=$(BUILD)/fixtures QM_QUICK=1 valgrind -q --error-exitcode=99 --leak-check=full \
	    --errors-for-leak-kinds=definite $(BUILD)/$$t > /dev/null; done
	@echo "VALGRIND CLEAN"

bench: all $(FIXTURES)
	$(BUILD)/quillm-bench $(BUILD)/fixtures/bench_f32.qlm $(BUILD)/fixtures/bench_q8_0.qlm $(BUILD)/fixtures/bench_q4_0.qlm

# type-check the NEON path for aarch64 without a cross toolchain
neon-check:
	clang --target=aarch64-linux-gnu -ffreestanding -std=c11 -Wall -Wextra -Werror -fsyntax-only src/kernels_neon.c
	@echo "NEON path type-checks for aarch64"

clean:
	rm -rf build build-asan build-tsan

# ThreadSanitizer on the concurrency-heavy tests
tsan:
	$(MAKE) $(addprefix build-tsan/,test_threads test_model) BUILD=build-tsan OPT=-O1 \
	  EXTRA_CFLAGS="-fsanitize=thread" EXTRA_LDFLAGS="-fsanitize=thread"
	QM_QUICK=1 QM_FIXTURES=$(BUILD)/fixtures build-tsan/test_threads
	QM_QUICK=1 QM_FIXTURES=$(BUILD)/fixtures build-tsan/test_model
