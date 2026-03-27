# Cohere Transcribe in Rust

Rust implementation for the
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
model. Includes a self-contained CLI and an OpenAI-compatible API server for AI agents.

Supports English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Greek,
Arabic, Japanese, Chinese, Vietnamese, and Korean.

---

## Quick Start (pre-built binaries)

No build tools needed — just download and run.

### 1. Download model weights

The model is gated — you must log in to HuggingFace and accept the license terms first:
<https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>

Then download:

```bash
pip install huggingface_hub
huggingface-cli download CohereLabs/cohere-transcribe-03-2026 \
  --local-dir models/cohere-transcribe-03-2026
```

### 2. Download the release for your platform

Go to [Releases](https://github.com/second-state/cohere_transcribe_rs/releases) and
download the zip for your platform:

| Platform | Asset |
|----------|-------|
| Linux x86\_64 (CPU) | `transcribe-linux-x86_64.zip` |
| Linux x86\_64 (CUDA 12.6) | `transcribe-linux-x86_64-cuda.zip` |
| Linux aarch64 (CPU, SVE) | `transcribe-linux-aarch64.zip` |
| Linux aarch64 (CUDA 12.6) | `transcribe-linux-aarch64-cuda.zip` |
| macOS Apple Silicon | `transcribe-macos-aarch64.zip` |

```bash
# Example for Linux x86_64:
unzip transcribe-linux-x86_64.zip
cd transcribe-linux-x86_64
```

### 3. Copy vocab.json into the model directory

The release includes a pre-generated `vocab.json`. Copy it to the model folder:

```bash
cp vocab.json ../models/cohere-transcribe-03-2026/
```

### 4. Test the CLI

```bash
./transcribe --model-dir ../models/cohere-transcribe-03-2026 --language en recording.wav
```

No `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` needed — the binary has RPATH baked in
and finds `libtorch/` (Linux) or `mlx.metallib` (macOS) in the same directory.

### 5. Test the API server

```bash
# Start the server
./transcribe-server --model-dir ../models/cohere-transcribe-03-2026 --port 8080 &

# Wait for "Listening on ..." message, then:
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=cohere-transcribe" \
  -F "language=en"
```

The server is OpenAI Whisper API compatible — works with any OpenAI client library.

---

## CLI Reference

```
USAGE:
    transcribe --model-dir <PATH> [OPTIONS] <AUDIO>...

ARGUMENTS:
    <AUDIO>...          One or more audio files to transcribe

OPTIONS:
    -m, --model-dir     Path to the model directory
    -l, --language      Language code [default: en]
        --no-punctuation  Strip punctuation from output
        --max-tokens    Max tokens to generate per segment [default: 448]
    -v, --verbose       Enable info logging (-vv for debug)
    -h, --help          Print help
```

### Language codes

| Code | Language   | Code | Language   | Code | Language  |
|------|------------|------|------------|------|-----------|
| `en` | English    | `de` | German     | `nl` | Dutch     |
| `fr` | French     | `es` | Spanish    | `pl` | Polish    |
| `ar` | Arabic     | `it` | Italian    | `el` | Greek     |
| `ja` | Japanese   | `pt` | Portuguese | `vi` | Vietnamese|
| `zh` | Chinese    | `ko` | Korean     |      |           |

### Audio formats

WAV, FLAC, MP3, AAC, OGG — anything supported by
[symphonia](https://github.com/pdeljanov/Symphonia).
Audio is automatically converted to 16 kHz mono.

Files longer than ~35 seconds are split into overlapping chunks (5 s overlap)
and the results joined automatically.

### Examples

```bash
# Transcribe a single file
./transcribe -m models/cohere-transcribe-03-2026 interview.mp3

# French, no punctuation
./transcribe -m models/cohere-transcribe-03-2026 --language fr --no-punctuation speech.wav

# Multiple files — prints filename before each transcript
./transcribe -m models/cohere-transcribe-03-2026 call1.wav call2.wav call3.flac

# Show model loading progress
./transcribe -m models/cohere-transcribe-03-2026 -v audio.wav
```

---

## API Server

`transcribe-server` is an OpenAI-compatible HTTP server built on [Axum](https://github.com/tokio-rs/axum).
It serves the same model as the CLI and is a drop-in replacement for the OpenAI Whisper API.

### Start the server

```bash
./transcribe-server \
  --model-dir models/cohere-transcribe-03-2026 \
  --host 0.0.0.0 \
  --port 8080
```

The server loads the model at startup (~30–90 s depending on hardware), then prints:
```
Listening on http://0.0.0.0:8080
Endpoint: POST http://0.0.0.0:8080/v1/audio/transcriptions
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio — OpenAI compatible |
| `GET`  | `/health` | Liveness probe → `{"status":"ok"}` |

### Transcription request

```bash
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=cohere-transcribe" \
  -F "language=en" \
  -F "response_format=json"
```

**Form fields** (all OpenAI-compatible):

| Field | Required | Default | Notes |
|-------|----------|---------|-------|
| `file` | yes | — | Audio bytes (WAV, MP3, FLAC, AAC, OGG) |
| `model` | yes | — | Any string; the server ignores it |
| `language` | no | `en` | ISO-639-1 code |
| `response_format` | no | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | no | — | Ignored; always greedy |
| `prompt` | no | — | Ignored |

**Response formats:**

```bash
# json (default)
{"text": "Hello, world."}

# text
Hello, world.

# verbose_json
{"task":"transcribe","language":"en","duration":3.0,"text":"Hello, world.","segments":[...]}

# srt
1
00:00:00,000 --> 00:00:03,000
Hello, world.

# vtt
WEBVTT

00:00:00.000 --> 00:00:03.000
Hello, world.
```

### Using with OpenAI client libraries

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",  # required by the client, ignored by the server
)

with open("recording.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="cohere-transcribe",
        file=f,
        language="en",
    )
print(transcript.text)
```

```javascript
import OpenAI from "openai";
import fs from "fs";

const openai = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "not-used",
});

const transcript = await openai.audio.transcriptions.create({
  model: "cohere-transcribe",
  file: fs.createReadStream("recording.wav"),
  language: "en",
});
console.log(transcript.text);
```

### Server options

```
OPTIONS:
  -m, --model-dir     Model directory (required)
      --host          Bind address [default: 0.0.0.0]
  -p, --port          Port [default: 8080]
  -l, --language      Default language when not specified by client [default: en]
      --max-tokens    Max tokens per segment [default: 448]
  -v, --verbose       Logging (-v info, -vv debug)
  -h, --help          Print help
```

---

## Build from Source

### Backends

Two compute backends are available — select one at compile time:

| Backend | Platform | Feature flag | Accelerator |
|---------|----------|-------------|-------------|
| **libtorch** (default) | Linux x86\_64, Linux aarch64 | `--features tch-backend` | CPU (BLAS-optimized) |
| **MLX** | macOS Apple Silicon | `--features mlx` | Apple GPU (Metal) |

Both backends produce identical output from the same weights.

### Requirements

**All platforms:**
- **Rust** stable (1.70+) — install from [rustup.rs](https://rustup.rs)
- **8 GB RAM** — the model weights expand to ~5.6 GB at runtime
- **Python + sentencepiece** — one-time only, to extract `vocab.json`

**Linux (libtorch backend):**
- **libtorch** C++ library — downloaded once, ~500 MB

**macOS Apple Silicon (MLX backend):**
- macOS 14+ with an M-series chip
- mlx-c is built automatically from the git submodule by `build.rs`

### Step 1 — Install libtorch (Linux only)

Pick the build for your platform and extract it to `/opt/libtorch`.
This is the C++ library from PyTorch.org; no Python runtime is involved.

**Linux x86\_64:**
```bash
curl -Lo libtorch.zip \
  'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip'
sudo unzip libtorch.zip -d /opt
```

**Linux ARM64** (AWS Graviton3, Ampere Altra — requires SVE support):
```bash
curl -Lo libtorch.tar.gz \
  'https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz'
sudo tar xzf libtorch.tar.gz -C /opt
```

Both commands produce `/opt/libtorch/`. Set `LIBTORCH=/your/path` to use a different location.

> **Docker on macOS:** Extract libtorch to a native Linux path such as `/opt/libtorch`,
> not onto the macOS volume mount (e.g. `/Users/…`). The Linux linker cannot read
> large shared libraries through the virtiofs layer.

### Step 2 — Download model weights

```bash
pip install huggingface_hub
huggingface-cli download CohereLabs/cohere-transcribe-03-2026 \
  --local-dir models/cohere-transcribe-03-2026
```

### Step 3 — Extract the vocabulary (one time only)

The model uses a SentencePiece tokenizer. Run this script once to produce `vocab.json`,
which the Rust binary reads at runtime. Python is not needed after this step.

```bash
pip install sentencepiece
python tools/extract_vocab.py --model_dir models/cohere-transcribe-03-2026
```

### Step 4 — Build

**Linux (libtorch backend, default):**
```bash
LIBTORCH=/opt/libtorch cargo build --release
```

The `LIBTORCH` path is baked into the binary's RPATH by `build.rs`, so the binary
runs without `LD_LIBRARY_PATH`.

**macOS Apple Silicon (MLX backend):**
```bash
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

> **Docker on macOS (Linux builds):** If the project source is on a macOS volume mount,
> set `CARGO_TARGET_DIR` to a native Linux path to prevent SIGBUS during compilation:
> ```bash
> LIBTORCH=/opt/libtorch CARGO_TARGET_DIR=/tmp/cohere_target cargo build --release -j 1
> ```

### Step 5 — Run

```bash
./target/release/transcribe --model-dir models/cohere-transcribe-03-2026 recording.wav
```

No environment variables needed — RPATH is baked in at build time.

---

## Troubleshooting

**`libtorch not found`** (Linux)
Set `LIBTORCH=/path/to/libtorch` before building, or install to `/opt/libtorch`.

**`Missing required file 'vocab.json'`**
Run `python tools/extract_vocab.py --model_dir <model_dir>`, or copy `vocab.json`
from the release zip into the model directory.

**Process killed immediately (exit 137)**
Out of memory. The model needs ~5.6 GB of RAM. Check with `free -h` (Linux) or
Activity Monitor (macOS).

**`Illegal instruction` (exit 132) on ARM Linux**
The ARM64 libtorch build requires SVE (Scalable Vector Extension), available on
Graviton3, Ampere Altra, and similar server-class CPUs. It does not run on
Apple Silicon Linux VMs, which only expose NEON. Use a native Linux ARM64 server with SVE,
or use the macOS MLX backend on Apple hardware.

**`ELF section name out of range` at link time** (Linux)
libtorch is on a Docker volume-mounted macOS path. Move it to a native Linux
path such as `/opt/libtorch`.

---

## Performance

Benchmarked on an **M4 Mac Mini** (MLX/Metal GPU backend). Model loading takes a few
seconds on first run, but once loaded, inference is **10x–24x faster than real-time**.
NVIDIA GPU builds (CUDA libtorch) will be significantly faster still.

### CLI

| Audio | Duration | Wall time | Notes |
|-------|----------|-----------|-------|
| Demo WAV (real speech) | 5.44 s | 12.9 s | Includes 3.3 s model load + Metal compile |
| Synthetic 60 s tone | 60 s | 18.1 s | Tests chunking (chunks split at ~35 s) |

### API Server

The server loads the model once at startup. Subsequent requests skip model loading
and Metal shader compilation, making warm requests extremely fast.

| Request | Audio | Response time | Notes |
|---------|-------|---------------|-------|
| 1st request (cold Metal) | 5.44 s | 8.3 s | Includes Metal shader compilation |
| 2nd+ requests (warm) | 5.44 s | 0.4 s | 13x faster than audio duration |
| 60 s chunked (warm) | 60 s | 2.5 s | 24x faster than audio duration |
