from __future__ import annotations

import argparse
from pathlib import Path

from .convolve import ConvolutionConfig, convolve_signals, load_audio, save_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convolve a source audio file with an impulse response.")
    parser.add_argument("--signal", required=True, help="Input signal audio path")
    parser.add_argument("--ir", required=True, help="Impulse response audio path")
    parser.add_argument("--output", required=True, help="Output audio path")
    parser.add_argument("--mode", choices=["full", "same", "valid"], default="full")
    parser.add_argument(
        "--channel-strategy",
        choices=["match", "sum_ir", "left", "right"],
        default="match",
        help="How to handle channel mismatch between signal and IR",
    )
    parser.add_argument("--dry", type=float, default=0.0, help="Dry mix amount")
    parser.add_argument("--wet", type=float, default=1.0, help="Wet mix amount")
    parser.add_argument("--normalize", choices=["none", "peak", "rms"], default="peak")
    parser.add_argument(
        "--method",
        choices=["fft", "direct"],
        default="fft",
        help="Convolution implementation (fft is faster for long audio)",
    )
    parser.add_argument("--rms-target", type=float, default=0.1)
    parser.add_argument(
        "--sample-rate-policy",
        choices=["error", "signal", "ir"],
        default="error",
        help="How to resolve sample-rate mismatch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal, sr_signal = load_audio(args.signal)
    ir, sr_ir = load_audio(args.ir)

    if sr_signal != sr_ir:
        if args.sample_rate_policy == "error":
            raise ValueError(
                f"Sample rate mismatch: signal={sr_signal}, ir={sr_ir}. "
                "Resample beforehand or pass --sample-rate-policy signal|ir."
            )
        sample_rate = sr_signal if args.sample_rate_policy == "signal" else sr_ir
    else:
        sample_rate = sr_signal

    cfg = ConvolutionConfig(
        mode=args.mode,
        channel_strategy=args.channel_strategy,
        dry=args.dry,
        wet=args.wet,
        normalize=args.normalize,
        rms_target=args.rms_target,
        method=args.method,
    )
    output = convolve_signals(signal, ir, cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(out_path, output, sample_rate)
    print(f"Wrote convolved output to {out_path} at {sample_rate} Hz")


if __name__ == "__main__":
    main()
