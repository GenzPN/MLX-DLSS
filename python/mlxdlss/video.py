"""Video conversion through FFmpeg: decode frames, enhance them, encode the result.

FFmpeg decodes the source to raw RGB on a pipe, frames go through the pipeline
one by one (or in batches on GPU devices), and a second FFmpeg process encodes
them with user-controlled codec arguments while copying the source audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .pipeline import NeuralRenderingPipeline

DEFAULT_ENCODE_ARGS = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
PIXEL_FORMATS = {"rgb24": (np.uint8, 255.0, 3), "rgb48le": ("<u2", 65535.0, 6)}


class VideoToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_rate: Fraction
    frame_count: int | None
    duration: float | None
    has_audio: bool
    pixel_format: str | None
    codec: str | None

    @property
    def fps(self) -> float:
        return float(self.frame_rate)


def find_tool(name: str, explicit: str | None = None) -> str:
    path = explicit or shutil.which(name)
    if not path or not Path(path).exists() and not shutil.which(path):
        raise VideoToolError(f"{name} not found; install FFmpeg or pass --{name}")
    return path


def probe(path: str | Path, *, ffprobe: str | None = None) -> VideoInfo:
    tool = find_tool("ffprobe", ffprobe)
    command = [tool, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", "-count_packets", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise VideoToolError(f"ffprobe failed: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoToolError(f"no video stream in {path}")
    rate = Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1")
    if rate == 0:
        rate = Fraction(video.get("r_frame_rate") or "25/1")
    count = None
    for key in ("nb_frames", "nb_read_packets"):
        if video.get(key) not in (None, "N/A"):
            count = int(video[key]); break
    duration = None
    for source in (video, data.get("format", {})):
        if source.get("duration") not in (None, "N/A"):
            duration = float(source["duration"]); break
    if count is None and duration is not None:
        count = int(round(duration * float(rate)))
    return VideoInfo(
        width=int(video["width"]), height=int(video["height"]), frame_rate=rate, frame_count=count, duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        pixel_format=video.get("pix_fmt"), codec=video.get("codec_name"),
    )


@dataclass
class ConvertOptions:
    start_frame: int = 0
    frame_limit: int | None = None
    batch: int = 1
    pixel_format: str = "rgb24"
    decode_args: list[str] = field(default_factory=list)
    encode_args: list[str] | None = None
    audio: str = "copy"
    overwrite: bool = False
    status_interval: float = 60.0
    enhance: dict[str, Any] = field(default_factory=dict)
    temporal: bool = True
    robust_motion: bool = True
    motion: str = "flow"           # temporal mode: 'flow' (optical flow) or 'zero'
    scene_cut_threshold: float = 0.3
    blend_scale: float | None = None  # None: the network's own blend cap
    backend: str = "torch"         # 'torch' (this pipeline) or 'mlxdlss' (Swift Metal runtime via `mlxdlss stream`, macOS)
    model_package: str | None = None
    mlxdlss: str | None = None
    execution: str = "metal-fused"
    precision: str = "float16"
    prefetch: bool = True          # prepare one following temporal frame while the GPU renders


@dataclass
class ConvertResult:
    frames: int
    seconds: float
    width: int
    height: int
    fps: float
    output: Path
    scene_cuts: int = 0
    temporal: bool = False
    motion: str = "none"
    processing_scale: float = 1.0


def convert(
    source: str | Path,
    destination: str | Path,
    pipeline: NeuralRenderingPipeline,
    options: ConvertOptions | None = None,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ConvertResult:
    """Decode ``source`` with FFmpeg, enhance every frame, encode to ``destination``.

    ``progress(frames_done, frames_expected)`` is called after every encoded frame;
    ``should_stop()`` is polled once per decoded frame and ends the conversion early
    (the output is finalised with the frames written so far)."""
    from .video_pipeline import NeuralRenderStage, run_video

    options = options or ConvertOptions()
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    stage = NeuralRenderStage(pipeline, options)
    log(f"MODE {'temporal' if options.temporal else 'independent'} motion={options.motion if options.temporal else 'none'} scale={options.enhance.get('processing_scale', 1.0):g}")
    info, _, frames, seconds, rate = run_video(
        source, destination, [stage], options, ffmpeg=ffmpeg, ffprobe=ffprobe,
        log=log, progress=progress, should_stop=should_stop,
    )
    return ConvertResult(frames, seconds, info.width, info.height, float(rate), Path(destination),
                         stage.scene_cuts, options.temporal, options.motion if options.temporal else "none",
                         options.enhance.get("processing_scale", 1.0))


def compare_command(original: str | Path, processed: str | Path, *, player: str | None = None) -> list[str]:
    """mpv command showing the original and the processed video side by side."""
    tool = find_tool("mpv", player)
    return [tool, str(original), f"--external-file={processed}", "--lavfi-complex=[vid1][vid2]hstack[vo]", "--keep-open=yes"]
