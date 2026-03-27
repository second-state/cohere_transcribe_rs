# CLAUDE.md — cohere_transcribe_rs

## Project Introduction

`cohere_transcribe_rs` is a pure-Rust CLI that runs the
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
automatic speech recognition model without any Python or PyTorch runtime dependency.

Two backends are supported, selected at compile time via Cargo features:

| Feature | Platform | Library | Accelerator |
|---------|----------|---------|-------------|
| `tch-backend` (default) | Linux x86\_64 / aarch64 | libtorch C++ | CPU |
| `mlx` | macOS Apple Silicon | mlx-c (C bindings for Apple MLX) | Metal GPU |

The `tch-backend` follows the same approach as
[second-state/qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs):
use the `tch` Rust crate as a thin binding to libtorch, load SafeTensors weights directly,
and implement the full model in Rust.

The `mlx` backend uses direct `extern "C"` FFI against the mlx-c library — no published
Rust crate; same pattern as calling any C API. All model ops are re-implemented using
MLX primitives (channels-last layout, lazy evaluation, fused Metal kernels).

---

## Architecture and Design

```
Audio file
  └─ symphonia → decode WAV/MP3/FLAC/AAC/OGG to f32 samples
  └─ rubato     → resample to 16 kHz mono

Pre-processing (pure Rust, src/audio.rs)
  ├─ Pre-emphasis filter (α = 0.97)
  ├─ Hann-windowed STFT via rustfft (n_fft=512, win=400, hop=160)
  ├─ Mel filterbank (128 bins, 0–8 kHz, Slaney norm)
  ├─ Log energy
  └─ Per-feature normalisation (mean/std over time)

ConformerEncoder (src/encoder.rs)   ← 48 layers, d_model=1280, 8 heads
  ├─ ConvSubsampling — 5 × Conv2d + Linear (stride=8 total)
  ├─ ConformerLayer × 48
  │    ├─ FeedForward  (SiLU, expansion factor 4)
  │    ├─ RelPosAttn   (relative-position multi-head self-attention)
  │    ├─ ConformerConv (pointwise→GLU→depthwise→BN→SiLU→pointwise)
  │    └─ FeedForward  (second macaron half)
  └─ Linear projection 1280→1024  (encoder_decoder_proj)

TransformerDecoder (src/decoder.rs)  ← 8 layers, hidden=1024, 8 heads
  ├─ Token embedding + fixed sinusoidal positional encoding + LayerNorm
  ├─ DecoderLayer × 8
  │    ├─ Pre-LN self-attention  (with KV cache, grows per step)
  │    ├─ Pre-LN cross-attention (K/V pre-computed once from encoder)
  │    └─ Pre-LN FFN  (ReLU, hidden 4096)
  └─ Final LayerNorm → Linear(1024, 16384) = logits

Inference (src/inference.rs)
  ├─ Encode mel → encoder hidden states
  ├─ Pre-compute cross-attention K/V (one-time per utterance)
  ├─ Feed 9-token prompt to prime the KV cache
  └─ Greedy decode: argmax → next token, append to KV cache, repeat until EOS

Output: SentencePiece detokenization (▁ word-boundary markers, <0xXX> byte tokens)
```

**Prompt format** (9 tokens):
```
<|startofcontext|> <|startoftranscript|> <|emo:undefined|>
<|{lang}|> <|{lang}|> <|pnc|or|nopnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
```

**Long audio**: files exceeding `max_audio_clip_s` (~35 s) are split into overlapping
chunks (5 s overlap) and the transcripts concatenated.

---

## File Descriptions

### Shared (both backends, both binaries)

| File | Purpose |
|------|---------|
| `src/lib.rs` | Library crate root. Re-exports all shared modules so both binaries (`transcribe` and `transcribe-server`) access the same code without duplication. |
| `src/main.rs` | CLI entry point. Feature-gated dispatch: `run_backend` selects tch or MLX path at compile time. Handles audio chunking for long files. |
| `src/bin/server.rs` | Axum API server. OpenAI-compatible `POST /v1/audio/transcriptions` + `GET /health`. Loads model at startup, serialises inference through `Arc<Mutex<ModelState>>`. |
| `src/config.rs` | Deserialises `config.json` / `preprocessor_config.json` into typed structs. |
| `src/audio.rs` | Pure-Rust audio pipeline: symphonia decode, rubato resample, STFT + mel filterbank, per-feature normalisation. |
| `src/tokenizer.rs` | Loads `vocab.json`. Decodes token IDs to text (▁ boundaries, `<0xXX>` bytes, special token suppression). |

### tch-backend (Linux)

| File | Purpose |
|------|---------|
| `src/weights.rs` | Reads `model.safetensors`, converts BF16/F16/F32 → `tch::Tensor` (F32). |
| `src/encoder.rs` | `ConformerEncoder`: ConvSubsampling (NCHW), RelPosAttn, ConformerConv (GLU + BN), FeedForward (SiLU). |
| `src/decoder.rs` | `TransformerDecoder`: DecoderAttn, DecoderFFN (ReLU), KV cache, `precompute_cross_kv`. |
| `src/inference.rs` | Greedy decode loop using `tch::Tensor`. |

### MLX backend (macOS Apple Silicon)

| File | Purpose |
|------|---------|
| `src/mlx/ffi.rs` | Raw `extern "C"` FFI declarations for libmlxc: array lifecycle, shapes, arithmetic, conv, norm, attention. |
| `src/mlx/array.rs` | RAII `Array` wrapper (Drop → `mlx_array_free`). `from_data_f32`, `to_vec_f32`, `argmax_flat`, `shallow_clone`. |
| `src/mlx/stream.rs` | Global `OnceLock<MlxContext>` — `init_mlx(use_gpu)` and `default_stream()`. |
| `src/mlx/ops.rs` | Safe wrappers: `matmul`, `linear`, `conv2d`, `conv1d`, `layer_norm`, `scaled_dot_product_attention`, etc. |
| `src/mlx/weights.rs` | Reads `model.safetensors`, converts BF16/F16/F32 → `Array` (F32). |
| `src/mlx/encoder.rs` | MLX `ConformerEncoder`: weights transposed OIHW→OHWI for channels-last conv; BN eval via running stats. |
| `src/mlx/decoder.rs` | MLX `TransformerDecoder`: KV cache with `ops::cat`, cross-attn K/V pre-computed once. |
| `src/mlx/inference.rs` | Greedy decode loop using `Array`. |
| `src/mlx/mod.rs` | Module re-exports + mutual-exclusivity `compile_error!` for tch-backend + mlx. |

### Build / tooling

| File | Purpose |
|------|---------|
| `tools/extract_vocab.py` | One-time: dumps 16,384-token vocab to `vocab.json`. Python not needed after. |
| `Cargo.toml` | Features: `tch-backend` (default), `mlx`. `tch = { version = "0.20", optional = true }`. |
| `.cargo/config.toml` | `LIBTORCH_BYPASS_VERSION_CHECK=1`. |
| `build.rs` | Links Metal frameworks when `mlx` feature active; links libtorch rpath for `tch-backend`. |
| `.github/workflows/ci.yml` | CI: Linux x86\_64 (tch), Linux aarch64 (tch), macOS M-series (mlx). |
| `.gitignore` | Excludes `models/`, `*.safetensors`, `libtorch/`, `/target`. |

---

## Build Instructions

### Linux — tch-backend

**1. Download libtorch:**

```bash
# x86_64:
curl -Lo libtorch.zip \
  'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip'
sudo unzip libtorch.zip -d /opt    # → /opt/libtorch/

# ARM64 (requires SVE on the CPU — see Key Learning #3):
curl -Lo libtorch.tar.gz \
  'https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz'
sudo tar xzf libtorch.tar.gz -C /opt    # → /opt/libtorch/
```

> **Docker on macOS:** do NOT extract libtorch to the macOS volume mount.
> Use `/opt/libtorch` or `/tmp/libtorch` (native Linux filesystem).

**2. Generate vocab.json (one time):**
```bash
pip install sentencepiece
python tools/extract_vocab.py --model_dir models/cohere-transcribe-03-2026
```

**3. Compile:**
```bash
# Native Linux:
LIBTORCH=/opt/libtorch cargo build --release

# Docker volume-mount source (avoids SIGBUS):
LIBTORCH=/opt/libtorch CARGO_TARGET_DIR=/tmp/cohere_target cargo build --release -j 1
```

`LIBTORCH_BYPASS_VERSION_CHECK=1` is set automatically via `.cargo/config.toml`.

`build.rs` emits `-Wl,-rpath,<libtorch>/lib` so the binary runs without `LD_LIBRARY_PATH`.
This mirrors the approach in second-state/qwen3_asr_rs.

**4. Run:**
```bash
./target/release/transcribe --model-dir models/cohere-transcribe-03-2026 audio.wav
# No LD_LIBRARY_PATH needed — RPATH is baked in at build time.
```

---

### macOS — MLX backend

**1. Install mlx-c:**
```bash
brew install mlx
git clone --depth 1 https://github.com/ml-explore/mlx-c.git /tmp/mlx-c
cmake -S /tmp/mlx-c -B /tmp/mlx-c/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix mlx)" \
  -DCMAKE_INSTALL_PREFIX=/opt/mlx
cmake --build /tmp/mlx-c/build --parallel "$(sysctl -n hw.logicalcpu)"
sudo cmake --install /tmp/mlx-c/build
```

**2. Generate vocab.json** — same as Linux (Step 2 above).

**3. Compile:**
```bash
MLX_DIR=/opt/mlx cargo build --release --no-default-features --features mlx
```

**4. Run:**
```bash
./target/release/transcribe --model-dir models/cohere-transcribe-03-2026 audio.wav
# No DYLD_LIBRARY_PATH needed — RPATH is baked in at build time.
```

---

## Testing

There is no automated test suite yet. Manual verification:

```bash
# Smoke test — binary starts and prints help:
LD_LIBRARY_PATH=/opt/libtorch/lib ./target/release/transcribe --help

# Functional test with a synthetic WAV (pure tone — model will output silence/short text):
python3 -c "
import wave, struct, math
sr, dur = 16000, 3
s = [int(32767*math.sin(2*math.pi*440*t/sr)) for t in range(sr*dur)]
with wave.open('/tmp/test.wav','wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
    f.writeframes(struct.pack('<'+'h'*len(s),*s))
"
./target/release/transcribe --model-dir models/cohere-transcribe-03-2026 /tmp/test.wav

# Full transcription test requires a real speech file and ≥ 8 GB RAM.
```

**Memory note:** the model weights are ~2.8 GB in BF16, expanded to F32 at load time (~5.6 GB).
A machine with at least 8 GB RAM is required. On systems with less RAM the process is killed
(exit code 137 / OOM). The Docker development environment on Apple Silicon (3.8 GiB, no swap)
is insufficient for a full inference run; build and `--help` work, but inference is killed by the OOM killer.

---

## Key Learnings

### 1. No Python at runtime — tch links directly against libtorch

The `tch` crate supports two modes:
- `LIBTORCH_USE_PYTORCH=1` — finds libtorch inside a `pip install torch` installation. **Avoid** for production; requires Python + PyTorch at runtime.
- `LIBTORCH=/path/to/libtorch` — links against a standalone C++ library downloaded from pytorch.org or second-state. **Preferred.** No Python needed at build or runtime.

### 2. tch version vs libtorch version compatibility

`tch 0.20` (torch 2.4 C++ API) is the sweet spot for use with libtorch 2.7.x:
- `LIBTORCH_BYPASS_VERSION_CHECK=1` suppresses the runtime version mismatch warning.
- tch 0.20's generated C++ wrapper (`torch_api_generated.cpp`) only references ops that exist in all CPU-only libtorch builds (no CUDA, no ROCm, no quantization-only ops).
- Higher versions (tch 0.22, 0.24) embed references to CUDA-specific ops (`_cudnn_attention_backward`, `_fused_rms_norm`, `_grouped_mm`, etc.) that are absent from CPU-only libtorch headers and cause compilation failures.
- This is identical to the approach in [qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs).

### 3. second-state libtorch for aarch64

PyTorch does not publish an official standalone libtorch C++ download for Linux aarch64.
[second-state/libtorch-releases](https://github.com/second-state/libtorch-releases) provides
pre-built aarch64 binaries. Key caveats:

- The aarch64 2.7.1 build is compiled with **SVE256** (ARM Scalable Vector Extension).
  It will crash with `SIGILL` (illegal instruction, exit 132) on systems without SVE — including
  Apple Silicon (`implementer 0x61`) running Linux in Docker, which only exposes NEON/ASIMD.
  SVE is available on AWS Graviton3, Ampere Altra, and similar server-class ARM CPUs.
- The `nm -D` tool reports no symbols (empty output) for the second-state `libtorch_cpu.so`
  because nm uses BFD which cannot parse the large ELF section string table in the file.
  This is a **tooling limitation**, not a corrupt file — `readelf -s` returns 193,000+ functions normally.

### 4. Docker volume mount and the linker

When the project lives on a macOS volume mount (`virtiofs`) inside Docker:

| Problem | Cause | Fix |
|---------|-------|-----|
| `SIGBUS` during `rustc` compilation | Compiler tries to `mmap` build artifacts through virtiofs | `CARGO_TARGET_DIR=/tmp/cohere_target` |
| `ELF section name out of range` at link time | `ld.bfd` / `ld.gold` cannot read large `.so` through virtiofs | Place libtorch on native Linux fs (e.g. `/opt`, `/tmp`) |

These issues are **only** present in the Docker-on-Mac development environment.
On a native Linux machine (CI, production server) neither workaround is needed.

### 5. BF16 weights loaded as F32

The safetensors file stores weights in BF16 to halve disk size (~2.8 GB).
`src/weights.rs` converts them to F32 on load with a simple bit-shift (`(x as u32) << 16`).
This doubles RAM usage at runtime but avoids requiring BF16 hardware support in libtorch.

### 6. Cross-attention K/V pre-computation

A key inference optimisation: the encoder runs once per utterance, producing hidden states
of shape `(1, T', 1024)`. Before the first decoder step, cross-attention K and V projections
are computed for all 8 decoder layers in one pass (`TransformerDecoder::precompute_cross_kv`)
and reused at every autoregressive step. This eliminates `8 × max_tokens` redundant matmuls.

### 7. vocab.json instead of sentencepiece C++ library

Linking the C++ sentencepiece library is complex (cmake, shared-lib path) and adds a large
dependency. The `tools/extract_vocab.py` script runs once to dump the full 16,384-token
vocabulary to `vocab.json`. At runtime the Rust code reads this JSON — no native sentencepiece
linkage required, and the Python script is never needed again after this one-time setup.

### 8. MLX backend — channels-last (NHWC) convolution layout

Apple MLX uses **channels-last** memory layout for convolutions, opposite to PyTorch's channels-first (NCHW):

| | Input | Weight |
|-|-------|--------|
| PyTorch Conv2d | (N, C, H, W) | (O, I/g, kH, kW) |
| MLX Conv2d | (N, H, W, C) | (O, kH, kW, I/g) |
| PyTorch Conv1d | (N, C, L) | (O, I/g, kW) |
| MLX Conv1d | (N, L, C) | (O, kW, I/g) |

Weights loaded from safetensors are in PyTorch layout. In `src/mlx/encoder.rs`, helper functions
`pt_conv2d_to_mlx` and `pt_conv1d_to_mlx` transpose them at load time using `ops::transpose`.

### 9. MLX lazy evaluation model

MLX uses **lazy evaluation**: array operations are not executed until `eval()` is called or data is
read. This allows the runtime to fuse and optimise compute graphs before dispatch to Metal.

In practice:
- Call `array.eval()` before pre-computing cross-attention K/V (after encoder forward).
- Call `array.to_vec_f32()` which calls `eval()` internally when reading logits.
- `mlx::stream::synchronize()` blocks until all pending GPU operations complete — call this
  when benchmarking or before clean process exit.

### 10. CI strategy

`.github/workflows/ci.yml` has three jobs, each building **both** binaries (`transcribe` CLI
and `transcribe-server` API server):

| Job | Runner | Backend | Notes |
|-----|--------|---------|-------|
| `linux-x86_64` | `ubuntu-latest` | tch-backend | libtorch from pytorch.org cached; all smoke tests run |
| `linux-aarch64` | `ubuntu-24.04-arm` | tch-backend | second-state libtorch; tests run only if SVE detected |
| `macos-mlx` | `macos-15` | mlx | mlx-c built from source against Homebrew MLX; cached |

Each job:
1. `cargo check` — fast compile gate (no linking)
2. `cargo build --release` — builds both binaries
3. CLI smoke tests: `--help` and missing-model-dir clean error
4. Server smoke tests: `--help` and missing-model-dir clean error

Full end-to-end tests (actual audio → transcript) require the ~2.8 GB model weights and
are run via the manual `integration-test.yml` workflow (see below).

### 11. API server architecture

`src/bin/server.rs` implements an OpenAI-compatible transcription endpoint using Axum.

**Design decisions:**

- **Single binary, two purposes**: the server uses the same library modules as the CLI. No code
  duplication — `src/lib.rs` exposes everything, both binaries import from the crate library.

- **Blocking inference via `Arc<Mutex<ModelState>>`**: tch `Tensor` is `Send` but not `Sync`.
  Wrapping the model in `Mutex` serialises inference (one request at a time). This is appropriate
  for a local transcription server where requests are infrequent and inference dominates latency.
  For high-throughput use, replace with a worker-thread channel pattern.

- **Audio via temp file**: the OpenAI API accepts raw audio bytes in multipart form. We write
  them to a `tempfile` so symphonia can read from a seekable file path (symphonia requires
  `Seek`). The temp file is deleted automatically on drop.

- **RPATH embedded**: the server binary also has libtorch/mlx-c paths baked in via `build.rs`,
  so it runs without `LD_LIBRARY_PATH` just like the CLI.

- **Response formats**: supports `json` (default), `text`, `verbose_json`, `srt`, `vtt`.
  SRT and VTT use duration-based timing (start=0, end=audio_duration) since the model
  does not output word-level timestamps in its current greedy decode mode.

### 12. Model weights in CI

Full end-to-end inference tests need the ~2.8 GB safetensors file. Two options:

**Option A — GitHub Actions cache with pre-uploaded weights (recommended for private repos):**
Upload `model.safetensors` + `config.json` + `vocab.json` as a GitHub Actions cache entry
manually once, then restore it in CI:
```yaml
- uses: actions/cache/restore@v4
  with:
    path: models/cohere-transcribe-03-2026
    key: cohere-model-weights-03-2026
    fail-on-cache-miss: true
```
The initial population is done via a one-time manual workflow dispatch that downloads from
HuggingFace and saves to the cache.

**Option B — HuggingFace Hub download in CI (for public repos):**
```yaml
- name: Download model weights
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    pip install huggingface_hub
    huggingface-cli download CohereLabs/cohere-transcribe-03-2026 \
      --local-dir models/cohere-transcribe-03-2026 \
      --token "$HF_TOKEN"
    python tools/extract_vocab.py --model_dir models/cohere-transcribe-03-2026
```
Store `HF_TOKEN` as a GitHub Actions secret. This adds ~5–10 min download time per run.

### 13. MLX ConvSubsampling flatten order — NHWC vs NCHW

The most critical MLX bug was in `ConvSubsampling`. After the five Conv2d layers, the output
has shape `(1, T', F, C)` in MLX's NHWC layout. The PyTorch code flattens the last two dims
in NCHW order `(1, T', C, F)` before the linear projection. Naively flattening NHWC output
produces a different feature ordering, which garbles the encoder output and causes the decoder
to emit only a single token (EOS immediately).

**Fix:** transpose dims 2 and 3 before flatten:
```rust
let x = ops::transpose(&x, &[0, 1, 3, 2]);  // (1, T', F, C) → (1, T', C, F)
let x = ops::reshape(&x, &[1, t_prime, feat]);
```

**Lesson:** when porting PyTorch → MLX, any operation that depends on memory layout (flatten,
reshape, view) must account for the NHWC↔NCHW difference. Convolutions and matmuls handle
this internally, but flatten/reshape do not.

### 14. MLX `mlx_array_set` for O(1) weight cloning

The `shallow_clone()` method originally did a full CPU round-trip: `to_vec_f32()` →
`from_data_f32()`. For the ~2100 weight tensors loaded at startup, this added ~75 seconds
to encoder construction.

**Fix:** use `mlx_array_set(dst, src)` which shares underlying storage via reference counting:
```rust
pub fn shallow_clone(&self) -> Self {
    let mut new = Self::empty();
    unsafe { ffi::mlx_array_set(&mut new.ptr, self.ptr) };
    new
}
```
This is O(1) and preserves lazy evaluation — no data is copied or evaluated.

### 15. MLX batch norm — avoid CPU round-trips in the compute graph

The original `batch_norm_nlc` used helper functions that called `to_vec_f32()` to compute
`1/sqrt(var + eps)` on the CPU, then re-uploaded the result. This forced 96 GPU→CPU→GPU
round-trips per encoder forward pass (48 layers × 2 ops), breaking Metal kernel fusion.

**Fix:** use `mlx_rsqrt` (1/sqrt(x)) which is a single GPU-native op:
```rust
let inv_std = ops::rsqrt(&ops::add(&var, &eps_arr));
let x_norm = ops::mul(&ops::sub(x, &mean), &inv_std);
```

**General rule:** never call `to_vec_f32()` or `eval()` in the middle of a compute graph
unless you specifically need to read data on the CPU. Any CPU round-trip breaks lazy
evaluation and prevents Metal kernel fusion.

### 16. Tracing output must go to stderr

Both `src/main.rs` and `src/bin/server.rs` must configure `tracing_subscriber` with
`.with_writer(std::io::stderr)`. Without this, log output goes to stdout and contaminates
transcript output when captured via `result=$(./transcribe ...)` in shell scripts or CI.

### 17. Release CI — RPATH and bundled libraries

The release workflow (`release.yml`) bundles platform-specific runtime libraries in each
release zip so users need zero configuration:

- **Linux (tch):** the full `libtorch/` directory is included. `RUSTFLAGS` sets
  `-Wl,-rpath,$ORIGIN/libtorch/lib` so the binary finds libraries relative to itself.
- **macOS (MLX):** `mlx.metallib` (Metal shader library) is copied next to the binary.
  The MLX backend is statically linked, so no dylibs are needed.
- **vocab.json** is generated once in a separate job and included in every platform zip,
  so users only need to copy it into their model directory.

### 18. MLX eval() placement and GPU-side argmax

MLX lazy evaluation builds a computation graph that should be evaluated at outer loop
boundaries, not per-layer. Our encoder correctly runs all 48 conformer layers as one lazy
graph with a single `eval()` after. The decoder runs 8 layers per step — also fine.

The decode loop originally called `to_vec_f32()` on the logits (shape: 16,384) at every
step to perform argmax on the CPU. This transferred 64 KB per token and broke the lazy graph.

**Fix:** use `Array::argmax_flat()` which calls `mlx_argmax_axis` on GPU and transfers a
single i32 to CPU. The full graph (8 decoder layers + layer norm + linear head + argmax) is
now evaluated as one fused Metal dispatch per step.

**Rule of thumb:**
- `eval()` after encoder forward (1 call)
- `argmax_flat()` after each decoder step (1 call per token, transfers 4 bytes not 64 KB)
- Never `eval()` or `to_vec_f32()` per-layer or mid-graph

### 19. MLX weight count differs from tch (2104 vs 2152)

The MLX weight loader skips `num_batches_tracked` tensors (I64 dtype, used only during
PyTorch training). This results in 2104 loaded tensors vs 2152 for the tch backend
(difference = 48 conformer layers × 1 `num_batches_tracked` tensor each). This is expected
and correct — batch norm in eval mode uses `running_mean` and `running_var` only.
