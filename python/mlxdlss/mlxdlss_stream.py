"""Metal backend for the converter: frames stream through ``mlxdlss stream`` over pipes.

Protocol (per frame): little-endian uint32 flags (bit 0 = reset history), then
float32 colour (H, W, 3); in temporal mode also motion (H, W, 2) as normalised
history-UV offsets and depth (H, W, 1). One float32 RGB frame comes back per
input frame. Protocol 2 adds flag bit 1 for an optional float32 H×W×1 history
confidence plane after depth; the temporal adapter explicitly requests it.
Protocol 3 adds bit 2 to omit constant-one depth and bits 3/4 for packed
uint8/uint16 RGB when the source is integral and no resampling is needed.
Protocol 4 composes source-size display RGB on Metal. When the processing extent
differs, source float32 RGB follows the guides. Optional native FG returns the
first frame immediately, then interleaved generated/original frames per window.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from .composition import compose_detail, resample
from .temporal import FlowMotionEstimator, zero_motion, resolve_motion, prepare_temporal_frame

RESET_FLAG = 1


def find_mlxdlss(explicit: str | None = None) -> str:
    import os

    candidates = [explicit, os.environ.get("MLXDLSS_BINARY"), shutil.which("mlxdlss")]
    root = Path(__file__).resolve().parents[2]
    candidates += [str(root / ".build" / "release" / "mlxdlss"), str(root / ".build" / "debug" / "mlxdlss")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("mlxdlss binary not found: build it with `swift build -c release --product mlxdlss`, pass --mlxdlss PATH or set MLXDLSS_BINARY")


class MLXDLSSStreamSession:
    """Sequential Metal renderer; source-size frames, processing-size history."""

    def __init__(self, model_package: str | Path, width: int, height: int, *, temporal: bool = True,
                 motion: Callable | str = "flow", scene_cut_threshold: float = 0.3, mlxdlss: str | None = None,
                 profile: str = "standard", intensity: float = 1.0, execution: str = "metal-fused",
                 precision: str = "float16", processing_scale: float = 1.0, detail_strength: float = 1.0,
                 colour_strength: float = 1.0, detail_radius: float = 4.0, blend_scale: float | None = None,
                 robust_motion: bool = True, protocol_version: int = 4,
                 framegen_weights: str | Path | None = None, framegen_factor: int = 2,
                 framegen_batch: int = 4, framegen_precision: str = "float16", output_format: str = "f32"):
        if protocol_version not in (2, 3, 4):
            raise ValueError("protocol_version must be 2, 3 or 4")
        if not temporal and protocol_version == 4:
            protocol_version = 3
        self.protocol_version = protocol_version
        if not all(math.isfinite(x) for x in (detail_strength, colour_strength, detail_radius)) or detail_radius <= 0:
            raise ValueError("strengths and radius must be finite; radius must be positive")
        if output_format not in ("f32", "u8", "u16"):
            raise ValueError("output format must be f32, u8 or u16")
        if (framegen_weights is not None or output_format != "f32") and (not temporal or protocol_version != 4):
            raise ValueError("GPU frame generation and packed output require temporal protocol 4")
        if framegen_weights is not None and (framegen_factor < 2 or framegen_batch < 1):
            raise ValueError("frame generation requires factor >= 2 and batch >= 1")
        self.framegen_factor = framegen_factor if framegen_weights is not None else 1
        self.framegen_batch = framegen_batch
        self.pending_pairs = 0
        self.output_dtype = np.dtype({"f32": "<f4", "u8": "u1", "u16": "<u2"}[output_format])
        if not 1 <= processing_scale <= 4:
            raise ValueError("processing_scale must be within [1, 4]")
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.width, self.height, self.temporal = width, height, temporal
        self.processing_width = round(width * processing_scale) if temporal else width
        self.processing_height = round(height * processing_scale) if temporal else height
        self.scene_cut_threshold = scene_cut_threshold
        self.processing_scale = processing_scale if temporal else 1.0
        self.robust_motion = robust_motion
        self.detail = (detail_strength, colour_strength, detail_radius)
        if not temporal or motion == "zero":
            self.motion = zero_motion
        elif motion == "flow":
            self.motion = FlowMotionEstimator()
        elif callable(motion):
            self.motion = motion
        else:
            raise ValueError("motion must be 'flow', 'zero' or a callable")
        command = [find_mlxdlss(mlxdlss), "stream", str(model_package),
                   "--width", str(self.processing_width), "--height", str(self.processing_height),
                   "--mode", "temporal" if temporal else "first-frame", "--execution", execution,
                   "--precision", precision, "--profile", profile, "--intensity", str(intensity)]
        if temporal:
            command += ["--protocol-version", str(protocol_version)]
            if blend_scale is not None:
                command += ["--blend-scale", str(blend_scale)]
            if protocol_version == 4:
                command += ["--output-width", str(width), "--output-height", str(height),
                            "--detail-strength", str(detail_strength), "--colour-strength", str(colour_strength),
                            "--detail-radius", str(detail_radius), "--output-format", output_format]
                if framegen_weights is not None:
                    command += ["--framegen-weights", str(framegen_weights), "--framegen-factor", str(framegen_factor),
                                "--framegen-batch", str(framegen_batch), "--framegen-precision", framegen_precision]
        else:
            command += ["--processing-scale", str(processing_scale), "--detail-strength", str(detail_strength),
                        "--colour-strength", str(colour_strength), "--detail-radius", str(detail_radius)]
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
        except BaseException:
            self._stderr.close()
            raise
        self.previous: np.ndarray | None = None
        self.frame_index = self.scene_cuts = self.frames = 0
        self._reset_pending = False
        self._closed = self._finished = False
        self._summary = {}
        self._depth = (np.ones((self.processing_height, self.processing_width, 1), dtype="<f4").tobytes()
                       if temporal and protocol_version == 2 else None)

    def reset(self):
        self.previous = None
        self.frame_index = 0
        self._reset_pending = True

    def _error_text(self):
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")[-1000:]

    def process_frame(self, frame: np.ndarray, *, motion: np.ndarray | None = None,
                      history_confidence: np.ndarray | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("mlxdlss stream is closed")
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.height, self.width, 3) or not np.isfinite(frame).all():
            raise ValueError(f"frame must be finite ({self.height}, {self.width}, 3)")
        if not self.temporal and history_confidence is not None:
            raise ValueError("history confidence requires temporal mode")
        estimate = resolve_motion(self.motion, frame, self.previous, motion, history_confidence,
                                  scene_cut_threshold=self.scene_cut_threshold, robust_motion=self.robust_motion)
        return self._process_prepared(prepare_temporal_frame(frame, estimate, self.processing_scale))

    def _process_prepared(self, prepared) -> np.ndarray:
        if self.framegen_factor != 1:
            raise RuntimeError("use push_prepared for a frame-generation stream")
        output = self.push_prepared(prepared)[0]
        if self.temporal and self.protocol_version < 4:
            detail, colour, radius = self.detail
            output = compose_detail(prepared.source, resample(output, self.width, self.height),
                                    detail_strength=detail, colour_strength=colour, radius=radius)
        return output

    def push_prepared(self, prepared) -> list[np.ndarray]:
        """Send prepared NR inputs; return final frames when the native FG window completes."""
        if self._closed or self._finished:
            raise RuntimeError("mlxdlss stream is closed")
        frame = prepared.source
        flags = RESET_FLAG if self._reset_pending or prepared.reset else 0
        if prepared.reset:
            self.scene_cuts += 1
        width, height = self.processing_width, self.processing_height
        processing = prepared.color
        confidence = prepared.confidence if self.temporal else None
        if confidence is not None:
            flags |= 2
        color_dtype = "<f4"
        if self.temporal and self.protocol_version >= 3:
            flags |= 4
            if prepared.packed_color is not None:
                processing = prepared.packed_color
                color_dtype = "u1" if processing.dtype == np.uint8 else "<u2"
                flags |= 8 if processing.dtype == np.uint8 else 16
        payload = [struct.pack("<I", flags), memoryview(np.ascontiguousarray(processing, dtype=color_dtype)).cast("B")]
        if self.temporal:
            payload.append(memoryview(np.ascontiguousarray(prepared.motion, dtype="<f4")).cast("B"))
            if self._depth is not None:
                payload.append(self._depth)
            if confidence is not None:
                payload.append(memoryview(np.ascontiguousarray(confidence, dtype="<f4")).cast("B"))
        if self.protocol_version == 4 and (width, height) != (self.width, self.height):
            payload.append(memoryview(np.ascontiguousarray(frame, dtype="<f4")).cast("B"))
        try:
            for field in payload:
                self.process.stdin.write(field)
            self.process.stdin.flush()
            count = 1
            if self.framegen_factor > 1 and self.frames > 0:
                self.pending_pairs += 1
                count = self.pending_pairs * self.framegen_factor if self.pending_pairs == self.framegen_batch else 0
            outputs = self._read_outputs(count)
            if count:
                self.pending_pairs = 0
        except (BrokenPipeError, OSError) as error:
            detail = self._error_text()
            self.abort()
            raise RuntimeError(f"mlxdlss stream failed (rebuild the binary if its protocol is outdated): {detail}") from error
        except BaseException:
            self.abort()
            raise
        self.previous = frame.copy()
        self.frame_index = 1 if flags & RESET_FLAG else self.frame_index + 1
        self.frames += 1
        self._reset_pending = False
        return outputs

    def _read_outputs(self, count):
        width, height = ((self.width, self.height) if self.protocol_version == 4 else
                         (self.processing_width, self.processing_height))
        expected = height * width * 3 * self.output_dtype.itemsize
        outputs = []
        for _ in range(count):
            data = bytearray(expected)
            view, received = memoryview(data), 0
            while received < expected:
                size = self.process.stdout.readinto(view[received:])
                if not size:
                    raise RuntimeError(f"mlxdlss stream ended early (rebuild the binary if its protocol is outdated): {self._error_text()}")
                received += size
            outputs.append(np.frombuffer(data, dtype=self.output_dtype).reshape(height, width, 3))
        return outputs

    def finish(self) -> list[np.ndarray]:
        if self._closed or self._finished:
            return []
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            outputs = self._read_outputs(self.pending_pairs * self.framegen_factor)
            self.pending_pairs = 0
            self._finished = True
            return outputs
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if self._closed:
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self._release()

    def _release(self):
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        self._stderr.close()
        self._closed = True

    def close(self) -> dict:
        if self._closed:
            return self._summary
        try:
            self.finish()
            self.process.wait(timeout=30)
            stderr = self._error_text()
            if self.process.returncode:
                raise RuntimeError(f"mlxdlss stream exited with {self.process.returncode}: {stderr}")
            for line in stderr.splitlines():
                if line.startswith("{"):
                    try:
                        self._summary = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            return self._summary
        finally:
            self.abort()


class MLXDLSSFrameGenStream:
    """Frame generation through ``mlxdlss framegen-stream`` (Metal): push frames in order and collect the
    ``factor - 1`` generated frames of every consecutive pair, in stream order, as uint8 or float32 arrays.

    The server computes ``batch`` pairs per pass, so a pair's frames come back once its window is
    complete (``push`` returns them) or when the input ends (``finish`` returns the rest)."""

    def __init__(self, weights: str | Path, width: int, height: int, *, factor: int = 2, precision: str = "float16",
                 mlxdlss: str | None = None, batch: int = 4, format: str = "u8"):
        if format not in ("u8", "f32"):
            raise ValueError("frame format must be 'u8' or 'f32'")
        self.width, self.height, self.factor, self.batch = width, height, factor, max(1, int(batch))
        self.dtype = np.dtype(np.uint8 if format == "u8" else "<f4")
        command = [find_mlxdlss(mlxdlss), "framegen-stream", "--weights", str(weights), "--width", str(width), "--height", str(height),
                   "--factor", str(factor), "--batch", str(self.batch), "--format", format, "--precision", precision]
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
        except BaseException:
            self._stderr.close()
            raise
        self.frames = 0
        self.pending_pairs = 0
        self._closed = self._finished = False
        self._summary = {}

    def _error_text(self):
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")[-2000:]

    def _read_frames(self, count: int) -> list[np.ndarray]:
        expected = self.height * self.width * 3 * self.dtype.itemsize
        outputs = []
        for _ in range(count):
            data = bytearray(expected)
            view, received = memoryview(data), 0
            while received < expected:
                count = self.process.stdout.readinto(view[received:])
                if not count:
                    raise RuntimeError(f"mlxdlss framegen-stream ended early: {self._error_text()}")
                received += count
            outputs.append(np.frombuffer(data, dtype=self.dtype).reshape(self.height, self.width, 3))
        return outputs

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Send one RGB (H, W, 3) frame; returns the generated frames of every pair whose window
        just completed (``batch * (factor - 1)`` frames, or none)."""
        if self._closed or self._finished:
            raise RuntimeError("mlxdlss framegen-stream is closed")
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be ({self.height}, {self.width}, 3)")
        if not np.isfinite(frame).all():
            raise ValueError("frame must be finite")
        if self.dtype == np.uint8:
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        elif frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / np.float32(255)
        payload = memoryview(np.ascontiguousarray(frame, dtype=self.dtype)).cast("B")
        try:
            self.process.stdin.write(payload); self.process.stdin.flush()
        except OSError as error:
            detail = self._error_text()
            self.abort()
            raise RuntimeError(f"mlxdlss framegen-stream failed: {detail}") from error
        self.frames += 1
        if self.frames == 1:
            return []
        self.pending_pairs += 1
        if self.pending_pairs < self.batch:
            return []
        try:
            outputs = self._read_frames(self.pending_pairs * (self.factor - 1))
        except BaseException:
            self.abort()
            raise
        self.pending_pairs = 0
        return outputs

    def finish(self) -> list[np.ndarray]:
        """End the input and return the generated frames of the pairs still pending."""
        if self._closed or self._finished:
            return []
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.stdin = None
        try:
            outputs = self._read_frames(self.pending_pairs * (self.factor - 1)) if self.pending_pairs else []
        except BaseException:
            self.abort()
            raise
        self.pending_pairs = 0
        self._finished = True
        return outputs

    def abort(self):
        if self._closed:
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        self._stderr.close()
        self._closed = True

    def close(self) -> dict:
        """Finish the stream (discarding any pending output) and return the server's JSON summary."""
        if self._closed:
            return self._summary
        try:
            self.finish()
            self.process.wait(timeout=30)
            stderr = self._error_text()
            if self.process.returncode:
                raise RuntimeError(f"mlxdlss framegen-stream exited with {self.process.returncode}: {stderr}")
            for line in stderr.splitlines():
                if line.startswith("{"):
                    try:
                        self._summary = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            return self._summary
        finally:
            self.abort()
