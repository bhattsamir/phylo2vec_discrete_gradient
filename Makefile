CC = /usr/bin/clang
CFLAGS = -O3 -march=native -ffast-math

.PHONY: all lib verify bench bench_gpu paper clean

all: lib verify bench bench_gpu

lib: libphylo2vec_fastme.dylib

libphylo2vec_fastme.dylib: phylo2vec_fastme.c
	$(CC) $(CFLAGS) -shared -fPIC phylo2vec_fastme.c -o libphylo2vec_fastme.dylib

verify: verify_all
	./verify_all

verify_all: verify_all.c phylo2vec_fastme.c
	$(CC) -O3 -march=native verify_all.c phylo2vec_fastme.c -o verify_all

bench: bench_optimizers
	./bench_optimizers

bench_optimizers: bench_optimizers.c phylo2vec_fastme.c
	$(CC) $(CFLAGS) bench_optimizers.c -o bench_optimizers

bench_gpu: bench_parallel_gpu
	./bench_parallel_gpu

bench_parallel_gpu: bench_parallel_gpu.m phylo2vec_fastme.c
	$(CC) $(CFLAGS) -framework Metal -framework Foundation bench_parallel_gpu.m phylo2vec_fastme.c -o bench_parallel_gpu

paper:
	cd paper && pdflatex -interaction=nonstopmode phylo2vec_bme_paper.tex && pdflatex -interaction=nonstopmode phylo2vec_bme_paper.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out
