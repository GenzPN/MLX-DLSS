# Frame generation: what was recovered and how it was verified

MLX-DLSS's frame generator is a port of the video path of NVIDIA's DLSS
Frame Generation library (`libnvidia-ngx-dlssg.so`, DLSS SDK 310.7.0; the
Windows `nvngx_dlssg.dll` 310.8.0 and 310.9.16 carry the same network, see
[library builds](#library-builds)). This
note records the recovered graph, the evidence behind each piece and the
numbers that tie the port to the vendor's output. Nothing proprietary is
reproduced here: the weights stay in the user's copy of the library and are
extracted locally with `mlxdlss-weights extract-fg`.

## Method

The library only exposes its frame generator through Vulkan (`VK_NVX_binary_import`
launches of CUDA kernels). A headless harness fed it plain video frames with
zero motion vectors and a flat depth, and a hook on the device dispatch table
recorded every kernel launch with its parameter block, every weight upload,
and memory snapshots of the intermediate buffers and images after each launch.
The kernels' PTX was recovered from the library's fat binaries and read
through a small expression printer that folds a kernel's straight-line code
into the expressions it stores. Every stage below was then implemented in
PyTorch and compared with the snapshot of the same buffer on the same frames.

## The graph (one evaluate, 75 launches)

For video the library's motion-vector machinery — motion-vector dilation,
depth-keyed forward splatting into the interpolated time, occlusion weights,
the mask U-Net that judges the splat — all runs but produces constants: the
splatted flows are zero, the occlusion weights are `sigmoid(0)`, and the
warped candidates are the frames themselves. What remains is:

| stage | operation | verification |
| --- | --- | --- |
| candidates | 2×2 box means of the two frames (zero-padded to a multiple of 16) | MAE 1e-4 against the library's candidate buffer |
| error channel | mean-RGB \|a − b\| at half resolution, fetched at full resolution with bilinear filtering and box-averaged back: a separable `[1/8, 3/4, 1/8]` blur | MAE 4e-5 |
| block input | `[a, err, b, err, mask = 0, phase t]`, 10 channels | layout read from `k_initial_merge` |
| block0 | three stem convs (3×3, LeakyReLU 0.01 clamped to ±6, 2×2 mean pool), eight residual convs in pairs, three activated heads on the ×2-upsampled trunk, one linear 8-channel head on the ×2-upsampled heads: flow A (2), flow B (2), mask logit, residual (3) | every layer 0.03–0.3 % against its snapshot |
| refinement input | the candidates warped by twice block0's flows, block0's outputs, the zero mask, the residual and the phase: 18 channels | 0.24 % |
| block1 | the same structure with 16/32 channels; `flow = 2·f0 + f1`, `mask = m0 + m1`, `res = r0 + r1` | export buffer within 1.1–1.7 % (flows), 3.7 % (mask), 0.5 % (residual), correlation 1.000 |
| output | the export upsampled bilinearly (plain half-scale sampling, index-clamped), `sigmoid(mask)`, both full-resolution frames warped by the scaled flows and blended by the mask; the residual is not used by the output kernel | the library's `OutputInterpFrame` at **59.9 dB PSNR**, maximum 3/255 |

The library composes at full resolution inside its `main_kernel` family; the
synthesis networks run at half resolution. The multi-frame mode is the same
graph evaluated at phases k/N.

Layouts worth knowing when reading the kernels: the custom convolutions keep
weights as `[co/8][tap][ci/8][co%8][ci%8]`, the U-Net's `k_conv_fp16_nhwc`
keeps `mma.m16n8k16` fragments with a permuted 16-column order and the bias in
the tail of the same blob, and activations are NHWC fp16 with 8×8 tiles. The
extractor undoes all of it into dense `[Cout, Cin, 3, 3]` tensors.

## Whole-clip results

The README's five clips, even frames in and odd frames withheld, PSNR against
the withheld frames, the vendor's library against this port (PyTorch, M2 Max):

| clip | library | port |
| --- | --- | --- |
| train | 27.38 | 27.39 |
| handheld | 30.90 | 30.91 |
| druid | 33.48 | 33.49 |
| muse | 32.11 | 32.13 |
| fishes | 38.89 | 38.92 |

## Speed

The float16 Metal kernels specialize channel counts and activation/residual/
pooling flags at compilation. RGB8 streaming evaluates synthesis, composition
and quantization together, then writes the shared MLX storage without a host
copy; float32 streaming also borrows the output buffer through the synchronous
write. The public float32 interpolation API keeps its existing behavior.

Paired release measurements on M2 Max with real weights and consecutive video
frames resized to each extent, comparing against `ce3ec3e`. Each value is a
median over 40 batches after four warm-up batches. The current ranges come
from runs before and after the baseline run, with other desktop applications
active. These times include input preparation and both RGB8 pipe directions,
but exclude model startup, video decoding and encoding. Times are for the
**whole batch**, producing `pairs × (factor − 1)` generated frames.

| extent | pairs | factor | before | current |
| --- | --- | --- | --- | --- |
| 960×540 | 1 | 2 | 7.16 ms | 6.00–6.73 ms |
| 960×540 | 4 | 2 | 23.16 ms | 19.77–20.09 ms |
| 1920×1080 | 1 | 2 | 22.75 ms | 21.15–21.52 ms |
| 1920×1080 | 4 | 2 | 87.79 ms | 77.39–81.79 ms |
| 1920×1080 | 1 | 4 | 58.62 ms | 50.15–55.22 ms |
| 1920×1080 | 4 | 4 | 234.12 ms | 215.19–216.26 ms |

The generated RGB8 and float32 bytes matched the baseline in all six
configurations. Tests also cover both model precisions, strided inputs,
broadcast phases, odd and tiny extents, and RGB8 rounding and clamping.

The scalar/SIMD comparison used to select auto dispatch, before specialization,
used real weights and nonconstant synthetic RGB inputs on M2 Max.
These paired warm measurements include synthesis and composition
but exclude process startup and I/O. Times below are for the **whole batch**;
each batch produces `pairs × (factor − 1)` generated frames.

| extent | pairs | factor | scalar | auto |
| --- | --- | --- | --- | --- |
| 960×540 | 1 | 2 | 5.30 ms | 5.30 ms |
| 960×540 | 4 | 2 | 17.78 ms | 16.47 ms |
| 960×540 | 4 | 4 | 49.17 ms | 45.76 ms |
| 1920×1080 | 1 | 2 | 17.51 ms | 16.58 ms |
| 1920×1080 | 4 | 2 | 65.16 ms | 60.68 ms |
| 1920×1080 | 8 | 2 | 131.38 ms | 124.28 ms |
| 1920×1080 | 4 | 3 | 127.93 ms | 122.18 ms |
| 1920×1080 | 4 | 4 | 192.01 ms | 186.51 ms |

Auto dispatch uses SIMD matrix convolutions for eligible unpooled layers
with at least 32 output channels and 16,384 batch-inclusive pixels. Pooled
stems stay scalar: forcing SIMD throughout the network was slower. Batch 4
remains the default; a larger batch does not consistently reduce time per
generated frame. Set `MLXDLSS_FG_SIMD=0` to force scalar or `=1` to use SIMD
where supported when comparing on another Apple GPU; leave it unset for auto.
On the tested real-weight frames, auto/scalar float16 differences stayed
below `3.9e-4` maximum and `7.1e-6` mean absolute error. The float32 path is
unchanged.

Earlier end-to-end and cross-backend measurements, per generated frame after
warm-up (kernel compilation excluded), float16 unless noted:

| | 960×540 | 1920×1080 |
| --- | --- | --- |
| Metal, `mlxdlss framegen`, one frame (N = 1) | 6.3 ms | 21 ms |
| Metal, `mlxdlss framegen --factor 4` (the three phases as one batch, N = 3) | 5.2 ms | 23 ms |
| Metal, `mlxdlss framegen-stream --batch 4` through the uint8 pipe (what `mlxdlss-video framegen --backend mlxdlss` uses) | 6.4 ms | 25 ms |
| Metal, `mlxdlss framegen-stream --batch 1`, float32 pipe (the previous protocol) | 27 ms | 43 ms |
| Metal (float32, MLX convolutions) | 9.2 ms | 32 ms |
| PyTorch / MPS (float16) | 5.3 ms | 17 ms |
| PyTorch / MPS (float32) | 6.3 ms | 22 ms |

Every stage of the Swift port takes a batch (`[N, H, W, 3]` frames, one phase
per sample): the phases of one pair (`--factor 3|4`) and the consecutive pairs
of a video (`--batch`, default 4) run as one pass. Batching does not lower the
GPU time per frame much (the layers are throughput-bound already at N = 1:
540p gains 17 % at N = 3, 1080p nothing), but it lets the frame server overlap
the host work with the GPU: with `--batch 1` the server waits for the host
after every frame.
Standalone FG uses uint8 RGB in both pipe directions (`--format u8`, the
default), which took the host side of the stream from 22 ms to under 2 ms per
540p frame; the frames are converted on the GPU. The web chain keeps float32
between models, with one decode and one final encode. Temporal Metal NR → FG
uses `mlxdlss stream --protocol-version 4 --framegen-weights ...` in one process:
downscale/detail composition and the FG input window remain on Metal, with
final RGB8/RGB16 packing at the encoder boundary. The first original is emitted
immediately; each complete window emits generated frames followed by its next
original, and EOF flushes the remaining pairs. History resets preserve the FG
pair sequence. FG → NR and mixed backends use the float32 host bridge.

Each convolution runs as one Metal kernel with the bias, the clamped
LeakyReLU, the residual add and the 2×2 mean pool in its epilogue (the three
heads as one convolution, the linear head as a block-diagonal one); each
block's input is assembled by a single kernel (box means, blurred error,
upsampled flows, warps) and the output composed by another. The PyTorch port
batches the same way (`FrameGenerator.generate` and `generate_pairs`).
Accuracy: the two ports agree within `1e-7` MAE at float32 and `1.3e-5` at
float16 on real weights (a full 960×540 frame); a batched result equals the
one-frame result exactly at float32 and within float16 rounding on MPS.

## GPU video chain

Paired release measurements against `707d328`, M2 Max, real NR and FG weights,
512×384 video frames, float16 inference, NR detail strength 2, four FG pairs per
batch. Times are the mean per **input frame** over 16 frames after five warm-up
frames, including both models, display composition and float32 pipe I/O;
decode, encode and CPU motion preparation are excluded.

| NR processing scale | FG factor | separate processes + CPU display | GPU display + shared NR → FG |
| --- | --- | --- | --- |
| 1 | 2 | 43.17–45.05 ms | 40.39–43.83 ms |
| 2 | 4 | 158.05–165.32 ms | 140.87–152.42 ms |

Desktop load varies: a later scale-4 comparison drifted substantially even
between baseline runs, so it does not establish a chain speedup at that scale.
The default final-encoder path additionally packs RGB8/RGB16 on Metal; float32
remains intact between the models. On the tested real-weight sequences the
GPU display recipe differed from CPU composition by at most `2.4e-7`. Passing
those differences through float16 FG produced a maximum RGB difference of
`3.34e-4`, mean at most `8.1e-7`; final RGB8 differed by one code value in fewer
than `0.021%` of channel values. Checkpoints and inference precision are unchanged.
Synthetic integration tests also compare the shared path against separate
GPU-display/FG processes exactly, covering both model precisions, empty and
single-frame input, scene resets, multi-phase batches, short EOF windows,
RGB8/RGB16 output, audio/rate preservation and cancellation.

## Native output-head fusion

The linear output head now fetches its bilinearly enlarged input inside the
convolution. It preserves the intermediate half rounding and removes the
upsample allocation/dispatch in both synthesis blocks. Float32 inference keeps
its reference path. `MLXDLSS_FG_FUSED_UPSAMPLE=0` restores separate operations.

Paired release measurements on M2 Max, real weights, fixed nonconstant inputs,
four warm-ups and median of 20 batches, before/after/before. These measure the
complete warm FG graph (synthesis and composition), excluding I/O and startup;
they must not be compared directly with pipe or encoded-video timings.

| extent | generated frames per batch | separate upsample | fused output head |
| --- | --- | --- | --- |
| 960×540 | 1 | 4.44–4.46 ms | 3.77 ms |
| 960×540 | 4 | 14.44–14.50 ms | 12.25 ms |
| 1920×1080 | 1 | 14.26–14.41 ms | 12.10 ms |
| 1920×1080 | 4 | 51.50–51.94 ms | 42.45 ms |

All four complete float32 RGB outputs matched exactly. Kernel checks also
cover strided batches, odd extents and single-pixel edges. The native media
processor uses the same generator, with AVFoundation decode/encode and
float32 Metal RGB between NR and FG in either order.

## Library builds

`mlxdlss-weights extract-fg` reads three builds and writes the same 92 tensors
from each, bit for bit:

| library | weight location |
| --- | --- |
| `libnvidia-ngx-dlssg.so.310.7.0` (Linux) | absolute offsets |
| `nvngx_dlssg.dll` 310.8.0 | one contiguous weight image, found by the SHA-256 of its first 256 bytes at a 16-byte aligned position; offsets relative to it |
| `nvngx_dlssg.dll` 310.9.16 | the same image, 9328 bytes later |

A library without that image is rejected rather than read at guessed offsets.
The PTX of 310.9.16 and 310.8.0 was compared statically; neither library was
executed:

- The synthesis network kernels (custom convolutions, `k_conv_fp16_nhwc`,
  `k_initial_merge`) and all weights are identical, so an existing
  `framegen.safetensors` stays valid and the port's output does not change.
- 310.8.0 strips the names of the composition kernels to `main_kernel`;
  310.9.16 names them (`Kernel_OutputPush`, `Kernel_OutputPull`,
  `Prev2Curr*`, `DL1Net_Input`, `DL1Net_Output` and others), which matches
  the roles recovered from the 310.7.0 captures.
- `BlendCandidatesFused` changed its parameter block layout; its instruction
  mix is nearly the same. `DownsampleRGBAndEstimateUIFused` extends the UI
  estimation, which plain video does not use.
- `Kernel_BlendCandidatesFGLite` is new, probably the blend of the lighter
  frame generation preset. It is not ported: without a capture of the library
  running it, a port could not be verified.

## Not ported

The motion-vector and depth inputs, the HUD-less/UI compositing and the
disocclusion inpainting pass exist in the library and are no-ops for plain
video, so the port leaves them out; the recovered kernel roles are recorded in
the research notes should an engine-integrated caller ever want them.
