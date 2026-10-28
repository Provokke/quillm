<div align="center">

# quillm

A small, dependency-free LLM inference engine in C11. It has its own tensor/kernel library, block quantization, model file format, Llama-style transformer, BPE tokenizer and sampler. The only build requirements are a C compiler and pthreads. The P...

![License](https://img.shields.io/badge/LICENSE-MIT-blue?style=for-the-badge)  ![Language](https://img.shields.io/badge/C11-555555?style=for-the-badge)

Refer to [LICENSE](LICENSE) for the legal bits.

</div>

## 🏗️ Architecture

```
            tools/make_model.py (numpy)                       tests/gen_fixtures.py
     random weights | llama2.c .bin | (HF fetch)       numpy reference forward -> logits
                     \       |       /                                 |
                      v      v      v                                  v
                  .qlm file: [hdr][KV meta][tensor table][pad][64B-aligned data]
                                     |  mmap (qlm_file.c, bounds-checked)
                                     v
  quillm run -p "..." --> tokenizer.c (BPE) --> model.c forward(token, pos) --> sampler.c --> text
                                                    |
           +----------------------------------------+---------------------------+
           | rmsnorm / rope / softmax / swiglu      | attention over heads      |
           | (ops.c, scalar)                        | (thread pool)             |
           +------------------+---------------------+---------------------------+
                              | qm_matvec / qm_matmul_nt  (ops.c, threadpool.c)
                              v
                    qm_kernels_active()  <-- cpu.c: cpuid + xgetbv (+ QUILLM_ISA)
             +-----------+-------------+---------------+----------------+
             | scalar    | AVX2+FMA    | AVX-512 BW/VL | NEON (aarch64) |
             | reference | +F16C       | masked tails  | compiled out   |
             +-----------+-------------+---------------+----------------+
                dot_f32 · dot_q8 · dot_q4 · gemm_nt (gemm_driver.h: 64x256 blocks, 4x4 tiles)
```

There are more details in .

## 🚀 Quick start

```sh
make -j2                                   # build/quillm, build/quillm-bench, tests
python3 tools/make_model.py random m.qlm --dtype q8_0     # stories15M-shaped random model
python3 tools/make_model.py llama2c stories15M.bin m.qlm --dtype q4_0   # convert llama2.c ckpt
build/quillm run m.qlm -p "Once upon a time" -n 64 --temp 0.8 --top-p 0.95 --seed 1 -t 4
build/quillm info m.qlm
```

Here is real output. The weights are **random**, so the text is noise: the model has
never been trained. The prompt is tokenized with the 200 BPE merges learned from a small
built-in corpus, which is why some pieces are corpus words.

```
$ build/quillm run build/fixtures/small_f32.qlm -p "the fox" -n 30 --temp 0
the foxy on�tch the � the b�thendy and. The bird��P� inot s waP jX����P e �
[quillm] isa=avx512 threads=4 prompt=4 tok (568.9 tok/s) generated=30 tok (954.4 tok/s)

$ build/quillm info build/fixtures/bench_q8_0.qlm
file: build/fixtures/bench_q8_0.qlm (25948800 bytes)
metadata (11):
  general.name             str "quillm-random"
  llama.dim                u32 288
  llama.hidden_dim         u32 768
  llama.n_layers           u32 6
  ...
  tokenizer.merges         u32[400]
tensors (57):
  tok_embd               q8_0  [32000, 288]
```

## 🧪 Testing

`make test` runs the core suite. It generates fixtures with numpy, then runs 9 C test
binaries, a CLI script and pytest. The last run gave: **0 failures across ~162k C checks,
CLI 9/9, pytest 21/21.**

| Command | What it runs | Last result |
|---|---|---|
| `make test` | all C tests + CLI + pytest | ALL TESTS PASSED |
| `make asan` | whole C suite with `-fsanitize=address,undefined -fno-sanitize-recover=all` | clean (it found and we fixed a real use-after-free in a parser error path) |
| `make tsan` | thread tests + model tests under ThreadSanitizer (quick mode) | clean |
| `make valgrind` | whole C suite under memcheck (quick mode). Valgrind hides AVX-512, so this covers scalar + AVX2 | `VALGRIND CLEAN` |
| `make neon-check` | type-checks the NEON kernels with `clang --target=aarch64` | passes |
| `make CC=clang` | `-Wall -Wextra -Werror -Wshadow -Wvla` with clang 18 (gcc 13 is the default) | clean |

What the tests cover:

- **`test_fp16`:** checks all 65 536 halves round-trip and match F16C decoding. It also
  checks float→half against F16C on a strided sweep of all 2³² float bit patterns plus
  2 M random values: 3 047 809 compared, 0 mismatches.
- **`test_quant`:** Q8_0 and Q4_0 round trips on 1200 random lengths (1–700, mostly not
  multiples of 32) and scales from 2⁻¹² to 2¹², with outliers.
  - Per-element error must stay within `d/2` (plus a clamp term).
  - Error energy must stay within 1.6× of the `d²/12` rounding model.
  - Padding must quantize to zero.
  - Output must be **byte-identical to the numpy quantizer**.
- **`test_kernels`:** checks every available ISA (scalar, AVX2, AVX-512 on this machine)
  against a float64 oracle **and** against the scalar kernel.
  - dot_f32 uses n = 0…69 plus random n up to 3000.
  - The Q8/Q4 dots use random n up to 2000.
  - GEMM uses random M ≤ 40, N ≤ 150, K ≤ 600, which crosses the 64/256 block edges.
  - The tolerance is the γₙ·Σ|aᵢbᵢ| accumulation bound.
  - Elementwise ops (RMSNorm, softmax, RoPE, SwiGLU) are checked against closed forms.
- **`test_threads`:**
  - Pool coverage (every index exactly once, grain-aligned).
  - A stress test of 20k tiny jobs.
  - **Bitwise** thread-count invariance (1–7 threads) of f32/Q8/Q4 matvec and matmul
    (both split strategies), for every ISA.
- **`test_model`:** checks the full forward pass against the **numpy reference**
  (`tools/qlm.py:forward_ref`, float64) on 7 models: f32/Q8/Q4 × two shapes (GQA,
  dims of 64 and 72, hidden sizes of 172 and 200, so blocks are partial), plus a model
  converted from a llama2.c checkpoint. Each model runs 20 positions on each ISA.
  - Results must be bitwise equal at 2, 3 and 4 threads, and identical on a re-run.
  - Seeded sampled generation must be deterministic.
- **`test_noalloc`:** wraps `malloc`/`calloc`/`realloc`/`aligned_alloc` and counts calls.
  64 decode steps plus sampling make 0 allocations.
- **`test_format`:**
  - Every truncation of the header must be rejected.
  - 30 000 random header mutations must never crash or return an out-of-bounds tensor.
    About 40% still parse, and all of those stay in bounds.
  - Bad magic and bad version must be rejected.
- **`test_tokenizer` / `test_sampler`:**
  - C encodings must exactly equal Python's on 47 vectors (UTF-8, binary, repetitive
    input).
  - 3000 random round-trips.
  - Sampled frequencies must match softmax(logits/T) within 4σ over 400k draws.
  - top-k and top-p must restrict and renormalize correctly.
  - The top-p prefilter must match the unfiltered path draw for draw.
  - Same seed → same sequence.
- **`test_cli.sh` / `test_python.py`:**
  - Greedy text from the CLI must equal greedy decoding with the numpy reference
    (f32, 12 tokens).
  - Output must be identical across thread counts and across scalar/AVX2/AVX-512.
  - Exit codes, the converter round trip and causality of the reference are checked.

Numerical agreement with the numpy reference (max |err| / max |logit| over 20 positions,
from `build/test_model`):

| model | vs numpy, same (dequantized) weights | vs numpy, original fp32 weights | argmax agreement vs fp32 |
|---|---|---|---|
| f32 (both shapes, all ISAs) | 1.2e-7 – 5.5e-7 | same | 20/20 |
| Q8_0 | 1.3e-2 – 1.4e-2 | 2.2e-2 | 18–20/20 |
| Q4_0 | 1.5e-2 – 1.6e-2 | 3.2e-1 – 3.5e-1 | 8–9/20 |

For Q8/Q4, the error against the same weights comes from quantizing the *activations* to
Q8_0 inside the matvec. Asserted bounds: f32 < 1e-5; Q8/Q4 against the same weights
< 2e-2; against fp32, Q8 < 3e-2 and Q4 < 0.5. The large Q4-vs-fp32 error is real.
4-bit round-to-nearest with one scale per 32 elements, applied to every matrix of a
tiny random model whose logits are nearly tied, flips most argmaxes. Real trained models
tolerate this much better, but we did not measure that here.

## ⏱️ Benchmarks

The machine is a 4-vCPU x86-64 with AVX-512F/BW/VL/VNNI, running gcc 13.3 at `-O2`.
**It was shared with about 15 other build jobs (load average about 8 during the run),
so multithreaded (MT, 4 threads) numbers are noisy and often worse than 1 thread.**
Single-thread numbers are the more meaningful ones. Reproduce with `make bench`.

Kernel throughput (GFLOP/s; for matvec, 2·rows·cols per call):

| kernel | scalar | AVX2 | AVX-512 |
|---|---|---|---|
| dot f32 (n=4096, L1-resident) | 1.93 | 35.50 | 32.18 |
| matvec f32 4096×4096 (1T) | 0.91 | 8.58 | 8.31 |
| matvec f32 4096×4096 (MT) | 1.30 | 6.85 | 7.99 |
| matvec q8_0 4096×4096 (1T) | 4.35 | 18.69 | 15.78 |
| matvec q8_0 4096×4096 (MT) | 4.68 | 38.22 | 15.10 |
| matvec q4_0 4096×4096 (1T) | 5.61 | 15.84 | 15.74 |
| matvec q4_0 4096×4096 (MT) | 8.28 | 22.31 | 30.22 |
| gemm f32 512³ (1T) | 9.80 | 38.59 | 33.90 |
| gemm f32 512³ (MT) | 8.76 | 42.34 | 45.54 |

Decode tokens/sec on a stories15M-shaped random model (dim 288, 6 layers, 6 heads,
hidden 768, vocab 32000, 15.2 M params). This measures greedy decode of 128 tokens after
8 warm-up tokens (32 tokens for scalar):

| model file | threads | scalar | AVX2 | AVX-512 |
|---|---|---|---|---|
| bench_f32.qlm (97.6 MB) | 1 | 48.1 | 243.1 | 209.9 |
| bench_f32.qlm | 4 | 80.5 | 279.4 | 263.4 |
| bench_q8_0.qlm (25.9 MB) | 1 | 204.0 | 504.4 | 549.0 |
| bench_q8_0.qlm | 4 | 196.1 | 503.3 | 293.1 |
| bench_q4_0.qlm (13.7 MB) | 1 | 231.1 | 525.3 | 421.3 |
| bench_q4_0.qlm | 4 | 254.6 | 446.9 | 348.1 |

What the numbers say:
- The scalar reference is deliberately plain C at `-O2` without `-ffast-math`, so the
  compiler cannot vectorize the float reductions.
- AVX-512 is **not** faster than AVX2 here. f32 matvec is memory-bound. The AVX-512
  quantized dots widen int8→int16 and use `vpmaddwd`; they do not use VNNI
  (`vpdpbusd`), which this CPU has. Adding VNNI is the obvious next step.
- At this model size each matvec is tiny (288×288). Thread-pool wake-up cost through
  condition variables dominates, so extra threads do not help on a loaded machine.

## ⚠️ Honest limits

- **We have not run any real trained model.** huggingface.co was blocked by the
  sandbox's network proxy (HTTP 403 on CONNECT). `make_model.py fetch-tinystories`
  detects this and prints `SKIP`. The llama2.c converter is tested only on synthetic
  llama2.c-format checkpoints written by the test suite.
- The llama2.c **tokenizer** (sentencepiece `tokenizer.bin`) is not converted. Only the
  weights are. A converted model generates ids that our BPE vocabulary cannot decode, so
  they print as nothing. The same happens with the 32k-vocab benchmark models, which
  carry only 200 merges.
- The prompt is processed one token at a time (no batched prefill). The tiled GEMM is
  implemented, tested and benchmarked, but the model does not use it yet.
- Elementwise ops (RMSNorm, RoPE, softmax, SwiGLU, attention value accumulation) are
  scalar C. Only the dot, matvec and GEMM kernels are SIMD.
- The NEON path is only type-checked (`make neon-check`). We had no ARM hardware, so it
  has never run.
- There is no fp16/bf16 weight dtype. fp16 is used only for block scales. There is no
  K-quant / importance-aware quantization.
- Thread-pool dispatch uses a mutex and condvar (no spinning). This is good for a shared
  machine, but it adds per-op latency on small models.
