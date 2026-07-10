CC   ?= gcc
FLAGS = -O3 -mavx512f -mfma -lm

SRC = src/gru.c
BIN = gru

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(FLAGS) -o $@ $<

.PHONY: clean run

clean:
	rm -f $(BIN) *.o

run: $(BIN)
	./$(BIN)
