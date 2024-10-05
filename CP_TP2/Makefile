CC = gcc
BIN = bin/
SRC = src/
EXEC = k_means
THREADS = 32

CFLAGS = -std=c99 -Ofast -fomit-frame-pointer -funroll-loops -ftree-vectorize -msse4 -fopenmp

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)$(EXEC)

clean:
	rm -r bin/*

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)