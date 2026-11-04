UNAME_S := $(shell uname -s)

CC       ?= cc
CFLAGS   ?= -O3 -march=native
INCLUDES := -Iinclude

NVCC      ?= $(shell if [ -x /usr/local/cuda-12.8/bin/nvcc ]; then echo /usr/local/cuda-12.8/bin/nvcc; else echo nvcc; fi)
CUDA_ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fopenmp $(INCLUDES)

BUILD_DIR := build

ifeq ($(UNAME_S),Darwin)
    SHARED_EXT := dylib
    LDFLAGS    := -lm -lpthread
else
    SHARED_EXT := so
    LDFLAGS    := -lm -lpthread
endif

FASTME_LIB := $(BUILD_DIR)/libphylo2vec_fastme.$(SHARED_EXT)

.PHONY: all lib verify verify_asan bench_cpu cuda bench_metal paper clean

all: lib $(BUILD_DIR)/verify_all $(BUILD_DIR)/bench_optimizers

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ------------------------------------------------------------------------------
# 1. Core C99 Library (Phylo2Vec Discrete Gradient & FastME BME)
# ------------------------------------------------------------------------------
lib: $(FASTME_LIB)

$(FASTME_LIB): src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -ffast-math -shared -fPIC src/phylo2vec_fastme.c $(LDFLAGS) -o $@

# ------------------------------------------------------------------------------
# 2. CPU Verification & Sanitizer Suites
# ------------------------------------------------------------------------------
$(BUILD_DIR)/verify_all: tests/verify_all.c src/phylo2vec_fastme.c src/phylo2vec_eigen_dist.c | $(BUILD_DIR)
	$(CC) -O2 -march=native $(INCLUDES) tests/verify_all.c src/phylo2vec_fastme.c src/phylo2vec_eigen_dist.c $(LDFLAGS) -o $@

verify: $(BUILD_DIR)/verify_all $(FASTME_LIB)
	./$(BUILD_DIR)/verify_all
	python3 tests/verify_spr_coverage.py

$(BUILD_DIR)/verify_all_asan: tests/verify_all.c src/phylo2vec_fastme.c src/phylo2vec_eigen_dist.c | $(BUILD_DIR)
	$(CC) -O2 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(INCLUDES) tests/verify_all.c src/phylo2vec_fastme.c src/phylo2vec_eigen_dist.c $(LDFLAGS) -o $@

verify_asan: $(BUILD_DIR)/verify_all_asan
	./$(BUILD_DIR)/verify_all_asan

# ------------------------------------------------------------------------------
# 3. CPU Optimizer Benchmark
# ------------------------------------------------------------------------------
$(BUILD_DIR)/bench_optimizers: benchmarks/bench_optimizers.c src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -ffast-math benchmarks/bench_optimizers.c src/phylo2vec_fastme.c $(LDFLAGS) -o $@

bench_cpu: $(BUILD_DIR)/bench_optimizers
	./$(BUILD_DIR)/bench_optimizers

# ------------------------------------------------------------------------------
# 4. CUDA GPU Solvers & Benchmarks (Linux / NVIDIA GPUs)
# ------------------------------------------------------------------------------
cuda: $(BUILD_DIR)/verify_eigen_dist $(BUILD_DIR)/bench_minibatch_recovery $(BUILD_DIR)/bench_rnasim_gpu $(BUILD_DIR)/phylo2vec_cuda_8xh100

$(BUILD_DIR)/verify_eigen_dist: tests/verify_eigen_dist.cu src/phylo2vec_eigen_dist.c src/phylo2vec_eigen_dist_cuda.cu src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) tests/verify_eigen_dist.cu src/phylo2vec_eigen_dist.c src/phylo2vec_eigen_dist_cuda.cu src/phylo2vec_fastme.c -o $@

$(BUILD_DIR)/bench_minibatch_recovery: benchmarks/bench_minibatch_recovery.cu src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) benchmarks/bench_minibatch_recovery.cu src/phylo2vec_fastme.c -o $@

$(BUILD_DIR)/bench_rnasim_gpu: benchmarks/bench_rnasim_gpu.cu src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) benchmarks/bench_rnasim_gpu.cu src/phylo2vec_fastme.c -o $@

$(BUILD_DIR)/phylo2vec_cuda_8xh100: benchmarks/phylo2vec_cuda_8xh100.cu src/phylo2vec_fastme.c | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) benchmarks/phylo2vec_cuda_8xh100.cu src/phylo2vec_fastme.c -o $@

# ------------------------------------------------------------------------------
# 5. Apple Silicon Metal Benchmark (macOS only)
# ------------------------------------------------------------------------------
bench_metal: $(BUILD_DIR)/bench_parallel_gpu
	./$(BUILD_DIR)/bench_parallel_gpu

$(BUILD_DIR)/bench_parallel_gpu: benchmarks/bench_parallel_gpu.m src/phylo2vec_fastme.c | $(BUILD_DIR)
	clang $(CFLAGS) -ffast-math -framework Metal -framework Foundation benchmarks/bench_parallel_gpu.m src/phylo2vec_fastme.c -o $@

# ------------------------------------------------------------------------------
# 6. LaTeX Paper Manuscript
# ------------------------------------------------------------------------------
paper:
	cd paper && pdflatex -interaction=nonstopmode phylo2vec_bme_paper.tex && pdflatex -interaction=nonstopmode phylo2vec_bme_paper.tex

clean:
	rm -rf $(BUILD_DIR) paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.fls paper/*.fdb_latexmk

