CC = gcc
BIN = bin/
SRC = src/
EXEC = k_means

CFLAGS = -std=c99 -O2 -fomit-frame-pointer -funroll-loops -ftree-vectorize -msse4

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c
	$(CC) $(CFLAGS) $(SRC)k_means.c -o $(BIN)$(EXEC)

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC)
