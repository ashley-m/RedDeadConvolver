from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math
import struct
import wave

ConvolutionMode = Literal["full", "same", "valid"]
ChannelStrategy = Literal["match", "sum_ir", "left", "right"]
Normalization = Literal["none", "peak", "rms"]

Audio = list[list[float]]  # [samples][channels]


@dataclass
class ConvolutionConfig:
    mode: ConvolutionMode = "full"
    channel_strategy: ChannelStrategy = "match"
    dry: float = 0.0
    wet: float = 1.0
    normalize: Normalization = "peak"
    rms_target: float = 0.1


def _to_2d(audio: list[float] | Audio) -> Audio:
    if not audio:
        return []
    first = audio[0]
    if isinstance(first, list):
        return audio  # type: ignore[return-value]
    return [[float(v)] for v in audio]  # type: ignore[arg-type]


def _extract_channel(audio: Audio, idx: int) -> list[float]:
    return [frame[idx] for frame in audio]


def _repeat_channel(channel: list[float], channels: int) -> Audio:
    return [[sample for _ in range(channels)] for sample in channel]


def _prepare_ir(ir: list[float] | Audio, channels: int, strategy: ChannelStrategy) -> Audio:
    ir_2d = _to_2d(ir)
    ir_channels = len(ir_2d[0]) if ir_2d else channels

    if ir_channels == channels:
        return ir_2d

    if strategy == "match":
        if ir_channels == 1 and channels == 2:
            return _repeat_channel(_extract_channel(ir_2d, 0), 2)
        if channels == 1 and ir_channels == 2:
            return [[(frame[0] + frame[1]) / 2.0] for frame in ir_2d]
        raise ValueError(f"Cannot match IR channels ({ir_channels}) to signal channels ({channels})")

    if strategy == "sum_ir":
        mono = [sum(frame) / len(frame) for frame in ir_2d]
        return _repeat_channel(mono, channels)

    if strategy == "left":
        return _repeat_channel(_extract_channel(ir_2d, 0), channels)

    if strategy == "right":
        idx = min(1, ir_channels - 1)
        return _repeat_channel(_extract_channel(ir_2d, idx), channels)

    raise ValueError(f"Unknown channel strategy: {strategy}")


def _convolve_1d(signal: list[float], kernel: list[float], mode: ConvolutionMode) -> list[float]:
    n, m = len(signal), len(kernel)
    if n == 0 or m == 0:
        return []

    full_len = n + m - 1
    full = [0.0] * full_len
    for i in range(n):
        si = signal[i]
        for j in range(m):
            full[i + j] += si * kernel[j]

    if mode == "full":
        return full
    if mode == "same":
        start = (full_len - n) // 2
        return full[start : start + n]
    if mode == "valid":
        length = max(n, m) - min(n, m) + 1
        start = min(n, m) - 1
        return full[start : start + length]
    raise ValueError(f"Unknown mode: {mode}")


def normalize_signal(audio: Audio, method: Normalization, rms_target: float = 0.1) -> Audio:
    if method == "none" or not audio:
        return audio

    flat = [abs(sample) for frame in audio for sample in frame]
    if method == "peak":
        peak = max(flat) if flat else 0.0
        if peak == 0:
            return audio
        scale = 1.0 / peak
    elif method == "rms":
        power = [sample * sample for frame in audio for sample in frame]
        rms = math.sqrt(sum(power) / len(power)) if power else 0.0
        if rms == 0:
            return audio
        scale = rms_target / rms
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return [[sample * scale for sample in frame] for frame in audio]


def convolve_signals(
    signal: list[float] | Audio,
    impulse_response: list[float] | Audio,
    config: ConvolutionConfig | None = None,
) -> list[float] | Audio:
    cfg = config or ConvolutionConfig()

    signal_2d = _to_2d(signal)
    if not signal_2d:
        return []

    channels = len(signal_2d[0])
    ir_2d = _prepare_ir(impulse_response, channels, cfg.channel_strategy)

    output_channels: list[list[float]] = []
    for ch in range(channels):
        sig_ch = _extract_channel(signal_2d, ch)
        ir_ch = _extract_channel(ir_2d, ch)
        wet = _convolve_1d(sig_ch, ir_ch, cfg.mode)

        if cfg.mode == "full":
            dry = sig_ch + ([0.0] * (len(wet) - len(sig_ch)))
        elif cfg.mode == "same":
            dry = sig_ch
        else:
            target = len(wet)
            start = max(0, (len(sig_ch) - target) // 2)
            dry = sig_ch[start : start + target]

        mixed = [(cfg.dry * dry[i]) + (cfg.wet * wet[i]) for i in range(len(wet))]
        output_channels.append(mixed)

    # transpose channels -> samples
    samples = len(output_channels[0]) if output_channels else 0
    output = [[output_channels[ch][i] for ch in range(channels)] for i in range(samples)]
    output = normalize_signal(output, cfg.normalize, cfg.rms_target)

    if channels == 1:
        return [frame[0] for frame in output]
    return output


def load_audio(path: str | Path) -> tuple[list[float] | Audio, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported right now.")

    ints = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    scale = 32768.0

    audio: Audio = []
    for i in range(0, len(ints), channels):
        frame = [max(-1.0, min(1.0, ints[i + ch] / scale)) for ch in range(channels)]
        audio.append(frame)

    if channels == 1:
        return [frame[0] for frame in audio], sample_rate
    return audio, sample_rate


def save_audio(path: str | Path, audio: list[float] | Audio, sample_rate: int) -> None:
    audio_2d = _to_2d(audio)
    channels = len(audio_2d[0]) if audio_2d else 1

    ints: list[int] = []
    for frame in audio_2d:
        for sample in frame:
            clamped = max(-1.0, min(1.0, sample))
            ints.append(int(clamped * 32767.0))

    packed = struct.pack("<" + "h" * len(ints), *ints)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(packed)
