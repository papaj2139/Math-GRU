# math-GRU

GRU training in C to do addition and subtraction.

A single-file (~550 line) implementation of a GRU recurrent neural network that learns to add and subtract 1–3 digit integers.
With a GRU cell with fused gate computation, AdamW with cosine LR schedule and Grokfast slow-gradient amplification,
Using AVX512 for vectorizing the backward and forward pass

Problems are encoded as reversed-character sequences (LSD  first), e.x `12+34=46` → `[2,1,+,4,3,=,6,4,=,=]`. The GRU processes the input prefix, then generates the answer autoregressively one char at a time. 
Loss is computed only over the answer portion (tokens after the first `=`).

## Requirements

- AMD64 CPU with **AVX-512F** and **FMA** support
- GCC (or compatible C compiler)
- GNU Make

## Build & Run

```bash
make #build the binary
make run #build and train
make clean #remove build artifacts
```

the training runs for 200,000 epochs, prints loss every 1,000, and evaluates test problems on completion.

## Architecture

Vocab size - 13 (`0-9`, `+`, `-`, `=`) 
Embedding dim - 16 
Hidden dim - 64
Parameters - ~17K 
Batch size - 16 
Learning rate - 0.005 

All set by #define's at the top of the src/gru.c file

## Implementation

The entire network lives in `src/gru.c`
Weights are stored column-major for contiguous vector loads, and all core loops operate on 16-wide AVX-512 vectors. Gradients are derived analytically and back-propagated through time

## License 
GPLv3
