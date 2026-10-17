import Foundation
import MLX

/// The attention half of a multi-head window block in one Metal kernel:
/// QKV projection, the window attention core, output projection and the
/// block's final cosine residual (optionally E4M3-published), on the published
/// feed-forward output.
///
/// One threadgroup per 8×8 window, eight simdgroups, each owning one window
/// row (8 tokens) in simdgroup-matrix registers. The heads run one after the
/// other through the same 8 KB of threadgroup memory (that head's normalised
/// keys and published values) plus 512 bytes of softmax scratch per simdgroup,
/// as in the attention core. Each head's attended values wait in the
/// thread's own output slots (each thread reads back only what it wrote) until
/// the output projection consumes them in chunks of eight column tiles. The
/// arithmetic is the per-operation chain's: the QKV and output projections
/// accumulate in float in the GEMMs' k order and round to half, the core is
/// `NeuralRenderingFusedWindowAttention.source`, and the residual rounds like
/// `NeuralRenderingFusedWindowAttention.cosineResidual`.
enum NeuralRenderingFusedWindowAttentionBlock {
  static let supportedChannels: Set<Int> = [64, 128, 256]
  /// `MLXDLSS_FUSED_ATTENTION_BLOCK=0` restores the separate GEMMs and kernels (diagnostics).
  nonisolated(unsafe) static var isEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_ATTENTION_BLOCK"] != "0"
  /// Widths that use the kernel inside the window blocks. Its in-kernel QKV
  /// projection reuses each weight tile for only eight rows, so from 128
  /// channels on the separate GEMM plus attention core is faster (2026-10-17,
  /// M2 Max: 64ch 272x480 1.17 -> 1.02 ms; 128ch 136x240 0.77 -> 0.85 ms; 256ch
  /// 68x120 0.63 -> 1.82 ms). `MLXDLSS_FUSED_ATTENTION_BLOCK_CHANNELS=64,128` overrides it.
  nonisolated(unsafe) static var blockChannels: Set<Int> =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_ATTENTION_BLOCK_CHANNELS"].map {
      Set($0.split(separator: ",").compactMap { Int($0) }).intersection(supportedChannels)
    } ?? [64]

  private static let kernel = makeKernel(name: "mlxdlss_fused_window_attention_block", source: source)

  static func makeKernel(name: String, source: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
      name: name,
      inputNames: ["input", "qkvWeight", "attentionScale", "attentionBias", "projection", "cosine", "params"],
      outputNames: ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  static let source = #"""
      constexpr uint tiles = channels / 8;
      constexpr uint heads = channels / 32;
      constexpr uint qkvStride = channels * 3;
      // Threadgroup memory: the current head's normalised keys (0..2048) and
      // published values (2048..4096) of the window, and 256 halfs of scratch
      // per simdgroup (4096..6144) for the softmax totals.
      threadgroup half4 arena4[6144 / 4];
      threadgroup half* K = (threadgroup half*)arena4;
      threadgroup half* V = K + 2048;
      const uint simd = simdgroup_index_in_threadgroup;
      const ushort lane = thread_index_in_simdgroup;
      // Element (fragmentRow, fragmentColumn + e) of an 8x8 simdgroup matrix is
      // thread_elements()[e] of this lane. Rows are window tokens, columns channels.
      const ushort fragmentRow = ((lane / 4) & 4) + ((lane / 2) % 4);
      const ushort fragmentColumn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;

      const uint height = params[0];
      const uint width = params[1];
      const uint padTop = params[2];
      const uint padLeft = params[3];
      const bool publish = params[4] != 0u;
      const uint windowsX = (width + padLeft + 7) / 8;
      const uint windowIndex = threadgroup_position_in_grid.x;
      const uint windowY = windowIndex / windowsX;
      const uint windowX = windowIndex % windowsX;
      const int y = int(windowY * 8) - int(padTop) + int(simd);
      const int x = int(windowX * 8) - int(padLeft) + int(fragmentRow);
      const bool inside = y >= 0 && y < int(height) && x >= 0 && x < int(width);
      const uint slot = (inside ? uint(y) * width + uint(x) : 0) * channels + fragmentColumn;
      const device half2* input2 = (const device half2*)input;
      device half2* output2 = (device half2*)output;
      const device half2* attentionBias2 = (const device half2*)attentionBias;
      const device half2* cosine2 = (const device half2*)cosine;

      // The window row (zeros outside the image, like the padded GEMM input).
      simdgroup_matrix<half, 8, 8> yt[tiles];
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < tiles; ++k) {
        const half2 value = inside ? input2[(slot + k * 8) / 2] : half2(0.0h);
        yt[k].thread_elements()[0] = value.x;
        yt[k].thread_elements()[1] = value.y;
      }

      threadgroup half* rowScratch = K + 4096 + simd * 256;
      const uint token0 = simd * 8;
      for (uint head = 0; head < heads; ++head) {
        const half scale = attentionScale[head];
        // QKV projection of this head (float accumulation, half result); vendor
        // cosine normalisation of q (scaled) and k, E4M3 of v. A token's 32
        // values sit on the four lanes of its fragment row, so its squared norm
        // is the vendor's xor reduction over those lanes (lane bit 3, then bit 0).
        simdgroup_matrix<half, 8, 8> qt[4];
        for (uint part = 0; part < 3; ++part) {
          simdgroup_matrix<half, 8, 8> result[4];
          const device half* columns = qkvWeight + part * channels + head * 32;
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < 4; ++o) {
            simdgroup_matrix<float, 8, 8> acc;
            acc.thread_elements()[0] = 0.0f;
            acc.thread_elements()[1] = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < tiles; ++k) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, columns + k * 8 * qkvStride + o * 8, qkvStride, ulong2(0), false);
              simdgroup_multiply_accumulate(acc, yt[k], right, acc);
            }
            result[o].thread_elements()[0] = half(acc.thread_elements()[0]);
            result[o].thread_elements()[1] = half(acc.thread_elements()[1]);
          }
          if (part < 2) {
            half partial[2];
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              const half v0 = result[0].thread_elements()[e];
              const half v1 = result[1].thread_elements()[e];
              const half v2 = result[2].thread_elements()[e];
              const half v3 = result[3].thread_elements()[e];
              half first = fma(v1, v1, v0 * v0);
              half second = fma(v3, v3, v2 * v2);
              partial[e] = first + second;
            }
            // Row positions p = (lane bit 3, lane bit 0): the xor sums are
            // (p0 + p2) + (p1 + p3), the same value on all four lanes, swapped through
            // this simdgroup's own value rows (written only after the norms).
            threadgroup half2* partials = (threadgroup half2*)(V + simd * 8 * 32) + fragmentRow * 4;
            partials[((lane >> 2) & 2) | (lane & 1)] = half2(partial[0], partial[1]);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const half2 p0 = partials[0];
            const half2 p1 = partials[1];
            const half2 p2 = partials[2];
            const half2 p3 = partials[3];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            half xorOne[2];
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              xorOne[e] = (p0[e] + p2[e]) + (p1[e] + p3[e]);
            }
            half norm = max(xorOne[0] + xorOne[1], half(0.00006198883056640625));
            const half reciprocalNorm = half(metal::fast::rsqrt(float(norm)));
            _Pragma("clang loop unroll(full)")
            for (uint o = 0; o < 4; ++o) {
              _Pragma("clang loop unroll(full)")
              for (uint e = 0; e < 2; ++e) {
                half normalized = result[o].thread_elements()[e] * reciprocalNorm;
                if (part == 0) { normalized *= scale; }
                result[o].thread_elements()[e] = half(float(mlxdlss_e4m3(normalized)));
              }
              if (part == 0) {
                qt[o] = result[o];
              } else {
                simdgroup_store(result[o], K + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
              }
            }
          } else {
            _Pragma("clang loop unroll(full)")
            for (uint o = 0; o < 4; ++o) {
              _Pragma("clang loop unroll(full)")
              for (uint e = 0; e < 2; ++e) {
                result[o].thread_elements()[e] = half(float(mlxdlss_e4m3(result[o].thread_elements()[e])));
              }
              simdgroup_store(result[o], V + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores (float accumulation), bias and the bit-affine softmax weights
        // in the matmul epilogue, the sequential half total of each query row,
        // E4M3 probabilities, attended = E4M3(P · v), parked in this thread's
        // own output slots.
        simdgroup_matrix<half, 8, 8> weights[8];
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 8; ++j) {
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f;
          acc.thread_elements()[1] = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, K + j * 8 * 32 + k * 8, 32, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, qt[k], right, acc);
          }
          half2 score = half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1]))
            + attentionBias2[((head * 64 + token0 + fragmentRow) * 64 + j * 8 + fragmentColumn) / 2];
          half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
          affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
          // The vendor's (packed << 5) + 0x7FF88000 on the half pair: within the
          // clamp range (bits 0x3C20..0x3E47) the low half always carries one into
          // the high half and shifts 7 into it, so each half becomes
          // (bits << 5) ^ 0x8000 on its own.
          half2 weight = as_type<half2>(ushort2((as_type<ushort2>(affine) << 5) ^ ushort2(0x8000)));
          weights[j].thread_elements()[0] = weight.x;
          weights[j].thread_elements()[1] = weight.y;
        }
        // Every lane reads its query row back from the simdgroup's scratch, 32 keys
        // at a time, and adds it up in key order.
        half total = 0.0h;
        _Pragma("clang loop unroll(full)")
        for (uint pass = 0; pass < 2; ++pass) {
          _Pragma("clang loop unroll(full)")
          for (uint j = 0; j < 4; ++j) {
            simdgroup_store(weights[pass * 4 + j], rowScratch + j * 8, 32, ulong2(0), false);
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
          const threadgroup half4* row4 = (const threadgroup half4*)(rowScratch + fragmentRow * 32);
          _Pragma("clang loop unroll(full)")
          for (uint i = 0; i < 8; ++i) {
            const half4 w = row4[i];
            total += w.x;
            total += w.y;
            total += w.z;
            total += w.w;
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        // The weights lie in [2^-14, 9.75] and the total is at least the largest of
        // them, so the probabilities are non-negative and at most about 1; P · v
        // stays within 64 * 1.125 * 448 and cannot be NaN.
        const half reciprocal = half(1.0f / float(total));
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 8; ++j) {
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            half weight = weights[j].thread_elements()[e] * reciprocal;
            weights[j].thread_elements()[e] = mlxdlss_e4m3_nonnegative(weight);
          }
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f;
          acc.thread_elements()[1] = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 8; ++k) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, V + k * 8 * 32 + o * 8, 32, ulong2(0), false);
            simdgroup_multiply_accumulate(acc, weights[k], right, acc);
          }
          const half2 attended = half2(
            mlxdlss_e4m3_no_nan(half(acc.thread_elements()[0])),
            mlxdlss_e4m3_no_nan(half(acc.thread_elements()[1])));
          if (inside) { output2[(slot + head * 32 + o * 8) / 2] = attended; }
        }
        // The next head overwrites the keys and values.
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Output projection (float accumulation) in chunks of eight column tiles,
      // residual with the feed-forward output and publication; each result
      // replaces the input fragment it consumed until the final store.
      constexpr uint chunkTiles = tiles < 8 ? tiles : 8;
      _Pragma("clang loop unroll(full)")
      for (uint chunk = 0; chunk < tiles / chunkTiles; ++chunk) {
        simdgroup_matrix<float, 8, 8> out[chunkTiles];
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          out[o].thread_elements()[0] = 0.0f;
          out[o].thread_elements()[1] = 0.0f;
        }
        for (uint kk = 0; kk < tiles; ++kk) {
          const half2 parked = inside ? output2[(slot + kk * 8) / 2] : half2(0.0h);
          simdgroup_matrix<half, 8, 8> attended;
          attended.thread_elements()[0] = parked.x;
          attended.thread_elements()[1] = parked.y;
          const device half* rows = projection + kk * 8 * channels + chunk * chunkTiles * 8;
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < chunkTiles; ++o) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, rows + o * 8, channels, ulong2(0), false);
            simdgroup_multiply_accumulate(out[o], attended, right, out[o]);
          }
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          const uint tile = chunk * chunkTiles + o;
          const half2 scale = cosine2[(tile * 8 + fragmentColumn) / 2];
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half branch = half(out[o].thread_elements()[e]);
            const half skip = yt[tile].thread_elements()[e] * scale[e];
            half value = branch + skip;
            if (publish) { value = half(float(mlxdlss_e4m3(value))); }
            yt[tile].thread_elements()[e] = value;
          }
        }
      }
      if (inside) {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < tiles; ++o) {
          output2[(slot + o * 8) / 2] = half2(yt[o].thread_elements()[0], yt[o].thread_elements()[1]);
        }
      }
      """#

  /// `cosineResidual(input, windowAttention(input), cosine)`, E4M3-published
  /// when `publish`, for a `[1, H, W, channels]` half input.
  static func apply(
    _ input: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    projectionWeight: MLXArray,
    cosine: MLXArray,
    windowOrigin: NeuralRenderingWindowOrigin,
    publish: Bool,
    kernel: MLXFast.MLXFastKernel? = nil
  ) -> MLXArray {
    precondition(input.ndim == 4 && input.shape[0] == 1 && input.dtype == .float16)
    let channels = input.shape[3]
    let headCount = channels / 32
    precondition(supportedChannels.contains(channels))
    precondition(qkvWeight.shape == [channels, channels * 3] && projectionWeight.shape == [channels, channels])
    precondition(attentionScale.shape == [headCount] && attentionBias.shape == [headCount, 64, 64])
    precondition(cosine.shape == [channels])
    let height = input.shape[1]
    let width = input.shape[2]
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    let windowCount = ((height + padTop + 7) / 8) * ((width + padLeft + 7) / 8)
    let half = { (array: MLXArray) in array.asType(.float16) }
    let params = NeuralRenderingKernelParameters.array([
      UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), publish ? UInt32(1) : UInt32(0),
    ])
    return (kernel ?? Self.kernel)(
      [
        input, half(qkvWeight), half(attentionScale), half(attentionBias).reshaped([headCount * 64 * 64]),
        half(projectionWeight), half(cosine), params,
      ],
      template: [("channels", channels)],
      grid: (windowCount * 256, 1, 1),
      threadGroup: (256, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }

}
