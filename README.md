# RedDeadConvolver

A lightweight Python utility for convolving one signal (or audio file) with another, with stereo-aware behavior and configurable processing.

## Features

- Mono or stereo input support.
- Configurable convolution mode: `full`, `same`, `valid`.
- Channel mismatch handling strategies:
  - `match` (best-effort channel matching)
  - `sum_ir` (collapse IR to mono and apply to all output channels)
  - `left` / `right` (pick one IR channel for all outputs)
- Dry/wet mix controls.
- Output normalization modes: `none`, `peak`, `rms`.
- No third-party runtime dependencies for the core DSP engine.
- Dashboard MVP via FastAPI + browser UI.

> Current audio I/O support is **16-bit PCM WAV** for portability.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

To run the dashboard:

```bash
pip install -e .[web]
rdc-dashboard
```

Then open `http://localhost:8000`.

## CLI usage

```bash
rdc-convolve \
  --signal ./audio/dry.wav \
  --ir ./audio/room_ir.wav \
  --output ./audio/out.wav \
  --mode full \
  --channel-strategy match \
  --dry 0.1 \
  --wet 0.9 \
  --normalize peak
```

### Sample-rate mismatch policy

- `--sample-rate-policy error` (default): fail fast if rates differ.
- `--sample-rate-policy signal`: use the signal sample rate for output metadata.
- `--sample-rate-policy ir`: use the IR sample rate for output metadata.

## Dashboard MVP (GUI)

The first GUI slice is implemented and currently uses:

- **Backend:** FastAPI (`src/reddeadconvolver/webapp.py`)
- **Frontend:** HTML/JS dashboard served from FastAPI (`src/reddeadconvolver/web/templates/index.html`)

Available controls:

- Upload signal WAV + IR WAV
- Convolution mode, channel strategy
- Dry/wet values
- Normalization + RMS target
- Sample-rate mismatch policy

When submitted, the app returns `convolved_output.wav` for download.

## Python API usage

```python
from reddeadconvolver.convolve import ConvolutionConfig, convolve_signals

signal = [[0.2, 0.1], [0.0, 0.3], [0.1, 0.0]]  # stereo signal
ir = [1.0, 0.5, 0.25]  # mono IR

cfg = ConvolutionConfig(mode="full", channel_strategy="match", dry=0.0, wet=1.0, normalize="peak")
out = convolve_signals(signal, ir, cfg)
```

## Next UI step (React/TypeScript)

This MVP dashboard is intentionally simple. The planned next phase is still:

1. Keep FastAPI as processing backend.
2. Add a React + TypeScript frontend for richer waveform UX and presets.
3. Optionally introduce background jobs for long renders.

## Development

```bash
python -m unittest discover -s tests -v
```
