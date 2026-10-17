import Foundation
import MLX

/// Window attention core for the multi-head window families: one Metal
/// threadgroup per (8×8 window, head) that takes the projected q/k/v rows of
/// the window, applies the vendor cosine normalisation (query scaled per head)
/// and E4M3 publication, the scores with the recovered attention bias, the
/// bit-affine softmax with E4M3 probabilities and the attended values, and
/// writes E4M3-published `[1, H, W, channels]` output rows in image layout.
///
/// It replaces the padded window partition, the per-head reshapes, the two
/// publish kernels, the scores/probabilities matmuls, the softmax kernel, the
/// value publication, the window reverse and the final slice of the
/// per-operation path with one dispatch. Every rounding point is copied from
/// those kernels: float accumulation in the matmuls, sequential per-token norm
/// and per-row softmax totals, half arithmetic everywhere else. Tokens outside
/// the image (shifted origins, ragged edges) behave like the zero padding of
/// the per-operation path: their q/k/v are zero and their outputs are dropped.
///
/// Eight simdgroups share a (window, head); each keeps one window row in
/// simdgroup-matrix registers. Threadgroup memory holds the keys and values
/// (8 KB) and 512 bytes of scratch per simdgroup (12 KB in all). The per-token
/// norms are the vendor's xor reduction over the four lanes of a fragment row,
/// exchanged through the simdgroup's own value rows, and every lane adds up its
/// softmax row in the recovered order after reading it back from the scratch
/// (2026-10-17, M2 Max: 25-60% faster than four simdgroups staging the whole
/// rows; the scratch totals instead of 32 lane shuffles per row: another
/// 16-25% from 68x120 up; bit-identical: `NeuralRenderingKernelVariantTests`).
/// Small integer parameter buffers for the custom kernels, kept alive for
/// the process. A parameter array created per call is released as soon as
/// its consumer is encoded, and MLX's concurrent command encoder only
/// orders dispatches by read-after-write and write-after-write on buffers:
/// a recycled parameter buffer could be overwritten while the kernel that
/// reads it is still running. Caching by value keeps every buffer alive.
enum NeuralRenderingKernelParameters {
  nonisolated(unsafe) private static var cache: [[UInt32]: MLXArray] = [:]
  static func array(_ values: [UInt32]) -> MLXArray {
    if let cached = cache[values] { return cached }
    let created = MLXArray(values)
    eval(created)
    cache[values] = created
    return created
  }
}

enum NeuralRenderingFusedWindowAttention {
  static let windowSize = 8
  /// One simdgroup per window row: eight per (window, head).
  static let simdgroupsPerWindow = 8
  private static let kernel = makeKernel(name: "mlxdlss_window_attention_core", source: source)

  /// Builds the kernel from `source` (earlier versions of it in the A/B probes).
  static func makeKernel(name: String, source: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
      name: name,
      inputNames: ["qkv", "attentionScale", "attentionBias", "params"],
      outputNames: ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  static let source = #"""
      // Threadgroup memory: the window's normalised keys (0..2048) and published
      // values (2048..4096) of this head, and 256 halfs of scratch per simdgroup
      // (4096..6144) for the softmax totals. Everything else stays in registers.
      threadgroup half4 arena4[6144 / 4];
      threadgroup half* arena = (threadgroup half*)arena4;
      threadgroup half* K = arena;
      threadgroup half* V = arena + 2048;
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
      const uint headCount = params[4];
      const uint channels = headCount * 32;
      const uint rowStride = channels * 3;
      const uint windowsX = (width + padLeft + 7) / 8;
      const uint head = threadgroup_position_in_grid.x % headCount;
      const uint windowIndex = threadgroup_position_in_grid.x / headCount;
      const uint windowY = windowIndex / windowsX;
      const uint windowX = windowIndex % windowsX;
      const int y = int(windowY * 8) - int(padTop) + int(simd);
      const int x = int(windowX * 8) - int(padLeft) + int(fragmentRow);
      const bool inside = y >= 0 && y < int(height) && x >= 0 && x < int(width);
      const uint pixel = inside ? uint(y) * width + uint(x) : 0;
      const device half2* qkv2 = (const device half2*)qkv;
      device half2* output2 = (device half2*)output;
      const device half2* attentionBias2 = (const device half2*)attentionBias;
      const half scale = attentionScale[head];

      // Phase 0: this simdgroup's window row (8 tokens) of k, v and q straight
      // into fragments (zeros outside the image). A token's 32 values sit on the
      // four lanes of its fragment row, so the vendor's cosine norm is the
      // two-step xor reduction over those lanes (lane bit 3, then bit 0).
      simdgroup_matrix<half, 8, 8> qt[4];
      _Pragma("clang loop unroll(full)")
      for (uint part = 0; part < 3; ++part) {
        simdgroup_matrix<half, 8, 8> tile[4];
        const uint base = (pixel * rowStride + part * channels + head * 32 + fragmentColumn) / 2;
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          const half2 value = inside ? qkv2[base + o * 4] : half2(0.0h);
          tile[o].thread_elements()[0] = value.x;
          tile[o].thread_elements()[1] = value.y;
        }
        if (part < 2) {
          half partial[2];
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half v0 = tile[0].thread_elements()[e];
            const half v1 = tile[1].thread_elements()[e];
            const half v2 = tile[2].thread_elements()[e];
            const half v3 = tile[3].thread_elements()[e];
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
              half normalized = tile[o].thread_elements()[e] * reciprocalNorm;
              if (part == 0) { normalized *= scale; }
              tile[o].thread_elements()[e] = half(float(mlxdlss_e4m3(normalized)));
            }
          }
        } else {
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < 4; ++o) {
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              tile[o].thread_elements()[e] = half(float(mlxdlss_e4m3(tile[o].thread_elements()[e])));
            }
          }
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          if (part == 0) {
            qt[o] = tile[o];
          } else {
            simdgroup_store(tile[o], (part == 1 ? K : V) + simd * 8 * 32 + o * 8, 32, ulong2(0), false);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Scores (float accumulation), bias and the bit-affine softmax weights in
      // the matmul epilogue, the sequential half total of each query row, E4M3
      // probabilities in registers, attended = E4M3(P · v), store from the
      // fragments.
      threadgroup half* rowScratch = arena + 4096 + simd * 256;
      const uint token0 = simd * 8;
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
        if (inside) {
          output2[(pixel * channels + head * 32 + o * 8 + fragmentColumn) / 2] = attended;
        }
      }
      """#

  /// E4M3-published attended values `[1, H, W, channels]` for projected rows
  /// `qkv` of shape `[1, H, W, 3 * channels]` (half precision).
  static func apply(
    qkv: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    headCount: Int,
    windowOrigin: NeuralRenderingWindowOrigin,
    kernel: MLXFast.MLXFastKernel? = nil,
    simdgroupsPerWindow simdgroups: Int = simdgroupsPerWindow
  ) -> MLXArray {
    precondition(qkv.ndim == 4 && qkv.shape[0] == 1 && qkv.dtype == .float16)
    let channels = qkv.shape[3] / 3
    precondition(channels == headCount * 32 && qkv.shape[3] == channels * 3)
    precondition(attentionScale.shape == [headCount])
    precondition(attentionBias.shape == [headCount, 64, 64])
    let height = qkv.shape[1]
    let width = qkv.shape[2]
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    let windowsY = (height + padTop + windowSize - 1) / windowSize
    let windowsX = (width + padLeft + windowSize - 1) / windowSize
    let threadgroups = windowsY * windowsX * headCount
    let params = NeuralRenderingKernelParameters.array([UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), UInt32(headCount)])
    return (kernel ?? Self.kernel)(
      [
        qkv, attentionScale.asType(.float16), attentionBias.asType(.float16).reshaped([headCount * 64 * 64]),
        params,
      ],
      // MLX 0.31 uses constant pointers for inputs smaller than eight elements.
      // Separate the two scale signatures so its cache never evicts a pipeline
      // still referenced by an encoded dispatch with another head count.
      template: [
        ("constantAttentionScale", headCount < 8),
        ("shuffleReciprocals", NeuralRenderingFusedWindowBlock.shuffleReciprocals),
      ],
      grid: (threadgroups * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [[1, height, width, channels]],
      outputDTypes: [.float16]
    )[0]
  }

  private static let residualKernel = MLXFast.metalKernel(
    name: "mlxdlss_cosine_residual_publish",
    inputNames: ["branch", "skip", "cosine", "params"],
    outputNames: ["output"],
    source: #"""
      const uint index = thread_position_in_grid.x;
      const uint count = params[0];
      if (index >= count) { return; }
      const uint channels = params[1];
      const bool publish = params[2] != 0u;
      const device half4* branch4 = (const device half4*)branch;
      const device half4* skip4 = (const device half4*)skip;
      const device half4* cosine4 = (const device half4*)cosine;
      device half4* output4 = (device half4*)output;
      half4 scaled = skip4[index] * cosine4[index % (channels / 4)];
      half4 value = branch4[index] + scaled;
      if (publish) {
        value = half4(
          half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))),
          half(float(mlxdlss_e4m3(value.z))), half(float(mlxdlss_e4m3(value.w))));
      }
      output4[index] = value;
      """#,
    header: NeuralRenderingTransformerOperations.e4m3MetalHeaderText
  )

  /// `branch + skip * cosine` in half precision, optionally E4M3-published,
  /// in one pass (same roundings as `cosineResidual` followed by `e4m3RoundTrip`).
  static func cosineResidual(skip: MLXArray, branch: MLXArray, cosine: MLXArray, publish: Bool) -> MLXArray {
    precondition(skip.shape == branch.shape && skip.dtype == .float16 && branch.dtype == .float16)
    let channels = skip.shape[skip.ndim - 1]
    precondition(cosine.shape == [channels] && channels.isMultiple(of: 4))
    let count = skip.size / 4
    let params = NeuralRenderingKernelParameters.array([UInt32(count), UInt32(channels), publish ? UInt32(1) : UInt32(0)])
    return residualKernel(
      [branch, skip, cosine.asType(.float16), params],
      grid: (count, 1, 1),
      threadGroup: (min(count, 256), 1, 1),
      outputShapes: [skip.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
