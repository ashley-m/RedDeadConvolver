from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from .convolve import (
    ChannelStrategy,
    ConvolutionConfig,
    ConvolutionMode,
    Normalization,
    convolve_signals,
    load_audio,
    save_audio,
)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI dependencies are required for dashboard features. "
        "Install with: pip install 'reddeadconvolver[web]'"
    ) from exc


app = FastAPI(title="RedDeadConvolver Dashboard", version="0.2.0")
TEMPLATE_PATH = Path(__file__).parent / "web" / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/api/convolve")
def convolve_api(
    signal: UploadFile = File(...),
    ir: UploadFile = File(...),
    mode: str = Form("full"),
    channel_strategy: str = Form("match"),
    dry: float = Form(0.0),
    wet: float = Form(1.0),
    normalize: str = Form("peak"),
    rms_target: float = Form(0.1),
    sample_rate_policy: str = Form("error"),
) -> FileResponse:
    if signal.filename is None or ir.filename is None:
        raise HTTPException(status_code=400, detail="Both signal and IR files are required.")

    with NamedTemporaryFile(suffix="_signal.wav", delete=False) as signal_file:
        signal_file.write(signal.file.read())
        signal_path = Path(signal_file.name)

    with NamedTemporaryFile(suffix="_ir.wav", delete=False) as ir_file:
        ir_file.write(ir.file.read())
        ir_path = Path(ir_file.name)

    valid_modes: set[ConvolutionMode] = {"full", "same", "valid"}
    valid_strategies: set[ChannelStrategy] = {"match", "sum_ir", "left", "right"}
    valid_norms: set[Normalization] = {"none", "peak", "rms"}

    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    if channel_strategy not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid channel strategy: {channel_strategy}")
    if normalize not in valid_norms:
        raise HTTPException(status_code=400, detail=f"Invalid normalize option: {normalize}")

    try:
        signal_audio, signal_sr = load_audio(signal_path)
        ir_audio, ir_sr = load_audio(ir_path)

        if signal_sr != ir_sr:
            if sample_rate_policy == "error":
                raise HTTPException(
                    status_code=400,
                    detail=f"Sample rate mismatch: signal={signal_sr}, ir={ir_sr}",
                )
            out_sr = signal_sr if sample_rate_policy == "signal" else ir_sr
        else:
            out_sr = signal_sr

        cfg = ConvolutionConfig(
            mode=mode,
            channel_strategy=channel_strategy,
            dry=dry,
            wet=wet,
            normalize=normalize,
            rms_target=rms_target,
        )
        output = convolve_signals(signal_audio, ir_audio, cfg)

        with NamedTemporaryFile(suffix="_out.wav", delete=False) as output_file:
            output_path = Path(output_file.name)

        save_audio(output_path, output, out_sr)
        return FileResponse(path=output_path, media_type="audio/wav", filename="convolved_output.wav")
    finally:
        signal_path.unlink(missing_ok=True)
        ir_path.unlink(missing_ok=True)
