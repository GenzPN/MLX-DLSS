# mlxdlss (Python)

PyTorch inference pipeline and bring-your-own-DLL weight tooling for the
recovered neural-rendering transformer of [MLX-DLSS](../README.md).

```sh
python -m pip install .            # numpy, torch, safetensors, pillow
python -m pip install '.[web,video]' # web UI and default temporal video (OpenCV)
python -m pip install '.[coreml]'  # optional Core ML conversion (macOS, Linux)

mlxdlss-weights all nvngx_dlssnr.dll weights/ --coreml 320x320   # your own DLL -> packed, logical, MLX, Core ML
mlxdlss-torch run --weights weights/dlssnr-small-weights-logical.safetensors --input in.png --output out.png
```

```python
from mlxdlss import NeuralRenderingPipeline, TemporalSession, FrameGenerator
pipeline = NeuralRenderingPipeline.from_safetensors("weights/dlssnr-small-weights-logical.safetensors", device="auto")
result = pipeline.enhance(image_float32_hwc, profile="standard", processing_scale=2, detail_strength=2)
session = TemporalSession(pipeline)
frame = session.process(image_float32_hwc)  # optional motion=engine_uv_offsets
generator = FrameGenerator.from_safetensors("weights/framegen.safetensors", device="auto")
middle = generator.generate(frame_a_uint8, frame_b_uint8, factor=2)[0]
```

`precision="reference"` (default) computes in float32 with the recovered
E4M3/half rounding points; `"fast"` runs the same graph in float16 on CUDA or
MPS. No weights are included or downloaded: the DLL comes from your own NVIDIA
driver or Streamline package and every artifact stays on your machine.

New video jobs and `mlxdlss-video convert` enable temporal rendering by default.
`--no-temporal` keeps independent frames. `TemporalSession` uses bidirectional
DIS flow to reproject rendered history, rejects unreliable correspondence, and
resets on scene changes. Scales 1–4 work on both PyTorch and the rebuilt Metal
stream; history is retained before display detail enhancement. Existing saved
jobs retain their explicit settings. See the [rendering controls](../README.md#cli).

## Video CLI and web UI

The portable video CLI requires `ffmpeg` and `ffprobe` in `PATH`:

```sh
mlxdlss-video convert in.mp4 out.mp4 --weights weights/dlssnr-small-weights-logical.safetensors --device cuda
mlxdlss-video convert in.mp4 out.mp4 --backend mlxdlss --model weights/NeuralRendering-Small.dlssmodel --encode-args "-c:v libx265 -crf 20"
mlxdlss-video framegen in.mp4 out.mp4 --weights weights/framegen.safetensors --backend mlxdlss
mlxdlss-video framegen in.mp4 slow.mp4 --weights weights/framegen.safetensors --mode slowmo --factor 4 --audio stretch
mlxdlss-web
```

`--start-frame` and `--frames` select a range; `--decode-args` and `--encode-args`
pass FFmpeg options. `--pix-fmt rgb48le` retains 16-bit sources. Default encoding
is H.264 CRF 18, medium preset, yuv420p. FG's `fps` mode keeps duration; `slowmo`
stretches it, with `--audio copy|stretch|none` controlling sound.

The web UI has live image/video previews, a frame timeline, batch export,
NR/FG ordering, slow motion, codec/audio/range controls, cancellation and retry.
Preview warms up to three preceding frames; FG runs only on export.

On macOS 26+, Metal previews and exports use the app’s native media pipeline.
MKV/WebM/AVI, explicit OpenCV motion and other backends use the portable path;
Vision/VideoToolbox motion requires native media. FFmpeg is optional for native
exports and required for completed side-by-side comparisons. Rebuild Swift
when updating the web package.

Jobs run one at a time in `~/MLX-DLSS/outputs/<job>/`. Settings can switch the
output folder and its job history; previous files remain untouched. Clearing
finished jobs hides them without deleting results. The HTTP API accepts an
optional JSON `output_options` form field (`codec`, `include_audio`,
`start_frame`, `frame_limit`); retry clones a failed or cancelled job.
`mlxdlss-web --native` is a Python/pywebview window. See [HTTP routes](mlxdlss/web/api.py)
and the [image](../docs/assets/web-image.png), [video](../docs/assets/web-video.png)
and [queue](../docs/assets/web-jobs.png) screenshots.

## Metal streaming

Temporal video prepares one following frame on a CPU worker while rendering
the current frame (`--no-prefetch` disables this). The Metal adapter uses stream
protocol 4, so rebuild the Swift binary when updating Python. The web effect
chain decodes and encodes once and preserves float32 frames between NR and FG
in both orders. Metal temporal NR performs downscale and detail composition on
the GPU, after retaining its processing-size history. A following Metal FG
stage uses those MLX arrays in the same process; only final frames return to
Python, packed for the encoder. This path requires both stages to resolve to
the same Swift binary. Other effect orders and backends retain their existing
float32 host path. OpenCV accelerates CPU detail filtering; the base image
installation retains its NumPy fallback.

`MLXDLSSStreamSession(..., protocol_version=3)` retains the CPU display recipe
for comparisons. Protocol 4 accepts `framegen_weights`, `framegen_factor`,
`framegen_batch`, `framegen_precision`, and `output_format="f32"|"u8"|"u16"`.
With FG enabled, use `push_prepared()` for prepared temporal frames and drain
`finish()` at EOF: the first frame is returned immediately, then complete
windows of generated/original frames in order, including the last short window.
`close()` drains any unread final window and releases the process; `abort()`
terminates it on cancellation or failure.
