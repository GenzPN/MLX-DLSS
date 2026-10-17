import Foundation
import MLX

/// The multi-head window blocks' branched feed-forward and its published
/// cosine residual in one Metal kernel:
/// `E4M3(FFN(x) + x * cos)`, where `FFN` is the dense arrangement of
/// `NeuralRenderingTransformerOperations.denseBranchedFeedForward`.
///
/// Each simdgroup owns eight rows and keeps them in simdgroup-matrix registers.
/// For every output group it produces the gated hidden tiles one 8×8 tile at a
/// time and feeds each straight into the group's branch projection, so the
/// `[rows, 4C]` hidden activation never leaves the registers. The published
/// group projections wait in the thread's own output slots (each thread reads
/// back only what it wrote) until the output projection consumes them in
/// chunks of eight column tiles, which bounds the live float accumulators. The roundings are
/// those of the GEMM chain it replaces: every product accumulates in float in
/// the GEMMs' sequential k order (half operands into float accumulators,
/// bit-identical to MLX's GEMMs), the GEMM results round to half, the gate and
/// the E4M3 publications sit at the same points, and the residual rounds twice
/// in half like `NeuralRenderingFusedWindowAttention.cosineResidual`.
enum NeuralRenderingFusedBranchedFeedForward {
  static let supportedChannels: Set<Int> = [64, 128, 256]
  /// Widths that use the kernel inside the window blocks. At 256 channels the
  /// expansion weight (512 KB) no longer stays cached for eight-row tiles and
  /// the GEMM chain is faster (2026-10-17, M2 Max, 68x120: 0.81 ms vs 0.94 ms).
  /// Diagnostics override: `MLXDLSS_FUSED_FFN_CHANNELS=64,128,256`.
  nonisolated(unsafe) static var blockChannels: Set<Int> =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN_CHANNELS"].map {
      Set($0.split(separator: ",").compactMap { Int($0) }).intersection(supportedChannels)
    } ?? [64, 128]
  /// Simdgroups (eight rows each) per threadgroup.
  nonisolated(unsafe) static var simdgroupsPerThreadgroup: Int =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN_SIMDGROUPS"].flatMap(Int.init) ?? 4
  /// `MLXDLSS_FUSED_FFN=0` restores the GEMM chain (diagnostics).
  nonisolated(unsafe) static var isEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_FFN"] != "0"

  private static let kernel = makeKernel(name: "mlxdlss_fused_branched_feed_forward", source: source)

  static func makeKernel(name: String, source: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
      name: name,
      inputNames: ["input", "expansion", "projection", "outputProjection", "cosine", "params"],
      outputNames: ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  static let source = #"""
      constexpr uint tiles = channels / 8;
      constexpr uint groups = channels / 32;
      constexpr uint hiddenStride = channels * 4;
      const uint simd = simdgroup_index_in_threadgroup;
      const ushort lane = thread_index_in_simdgroup;
      const ushort fragmentRow = ((lane / 4) & 4) + ((lane / 2) % 4);
      const ushort fragmentColumn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
      const uint rowCount = params[0];
      const uint row = (threadgroup_position_in_grid.x * simdgroups + simd) * 8 + fragmentRow;
      const bool inside = row < rowCount;
      const device half2* input2 = (const device half2*)input;
      device half2* output2 = (device half2*)output;
      const device half2* cosine2 = (const device half2*)cosine;

      simdgroup_matrix<half, 8, 8> xt[tiles];
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < tiles; ++k) {
        const half2 value = inside ? input2[(row * channels + k * 8 + fragmentColumn) / 2] : half2(0.0h);
        xt[k].thread_elements()[0] = value.x;
        xt[k].thread_elements()[1] = value.y;
      }
      for (uint g = 0; g < groups; ++g) {
        simdgroup_matrix<float, 8, 8> branch[4];
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          branch[o].thread_elements()[0] = 0.0f;
          branch[o].thread_elements()[1] = 0.0f;
        }
        for (uint b = 0; b < 4; ++b) {
          _Pragma("clang loop unroll(full)")
          for (uint n = 0; n < 4; ++n) {
            // Hidden tile (columns g, b, n*8..n*8+7) = x · W1, gate, E4M3.
            simdgroup_matrix<float, 8, 8> acc;
            acc.thread_elements()[0] = 0.0f;
            acc.thread_elements()[1] = 0.0f;
            const device half* column = expansion + (g * 4 + b) * 32 + n * 8;
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < tiles; ++k) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, column + k * 8 * hiddenStride, hiddenStride, ulong2(0), false);
              simdgroup_multiply_accumulate(acc, xt[k], right, acc);
            }
            simdgroup_matrix<half, 8, 8> hidden;
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              half value = half(acc.thread_elements()[e]);
              half clamped = clamp(value, half(-4.0), half(4.0));
              half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
              half gate = fma(clamped, linear, half(0.89453125));
              hidden.thread_elements()[e] = half(float(mlxdlss_e4m3(value * gate)));
            }
            // Next k-step (b*32 + n*8) of the group's branch projection.
            const device half* rows = projection + g * 128 * 32 + (b * 32 + n * 8) * 32;
            _Pragma("clang loop unroll(full)")
            for (uint o = 0; o < 4; ++o) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, rows + o * 8, 32, ulong2(0), false);
              simdgroup_multiply_accumulate(branch[o], hidden, right, branch[o]);
            }
          }
        }
        // Published group projection, parked in this thread's own output slots.
        _Pragma("clang loop unroll(full)")
        for (uint kk = 0; kk < 4; ++kk) {
          const half2 published = half2(
            half(float(mlxdlss_e4m3(half(branch[kk].thread_elements()[0])))),
            half(float(mlxdlss_e4m3(half(branch[kk].thread_elements()[1])))));
          if (inside) { output2[(row * channels + g * 32 + kk * 8 + fragmentColumn) / 2] = published; }
        }
      }

      // Output projection in chunks of eight column tiles; each residual result
      // replaces the input fragment it consumed until the final store.
      constexpr uint chunkTiles = 8;
      _Pragma("clang loop unroll(full)")
      for (uint chunk = 0; chunk < tiles / chunkTiles; ++chunk) {
        simdgroup_matrix<float, 8, 8> out[chunkTiles];
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          out[o].thread_elements()[0] = 0.0f;
          out[o].thread_elements()[1] = 0.0f;
        }
        for (uint kk = 0; kk < tiles; ++kk) {
          const half2 parked = inside ? output2[(row * channels + kk * 8 + fragmentColumn) / 2] : half2(0.0h);
          simdgroup_matrix<half, 8, 8> published;
          published.thread_elements()[0] = parked.x;
          published.thread_elements()[1] = parked.y;
          const device half* rows = outputProjection + kk * 8 * channels + chunk * chunkTiles * 8;
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < chunkTiles; ++o) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, rows + o * 8, channels, ulong2(0), false);
            simdgroup_multiply_accumulate(out[o], published, right, out[o]);
          }
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < chunkTiles; ++o) {
          const uint tile = chunk * chunkTiles + o;
          const half2 scale = cosine2[(tile * 8 + fragmentColumn) / 2];
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half branchValue = half(out[o].thread_elements()[e]);
            const half skip = xt[tile].thread_elements()[e] * scale[e];
            xt[tile].thread_elements()[e] = half(float(mlxdlss_e4m3(half(branchValue + skip))));
          }
        }
      }

      if (inside) {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < tiles; ++o) {
          output2[(row * channels + o * 8 + fragmentColumn) / 2] = half2(xt[o].thread_elements()[0], xt[o].thread_elements()[1]);
        }
      }
      """#

  /// `E4M3(denseBranchedFeedForward(input) + input * cosine)` for a half input
  /// with 64, 128 or 256 channels.
  static func apply(
    _ input: MLXArray,
    denseExpansionWeight: MLXArray,
    groupedProjectionWeight: MLXArray,
    outputProjectionWeight: MLXArray,
    cosine: MLXArray,
    kernel: MLXFast.MLXFastKernel? = nil,
    simdgroups: Int? = nil
  ) -> MLXArray {
    let channels = input.shape[input.ndim - 1]
    precondition(supportedChannels.contains(channels) && input.dtype == .float16)
    precondition(denseExpansionWeight.shape == [channels, channels * 4])
    precondition(groupedProjectionWeight.shape == [channels / 32, 128, 32])
    precondition(outputProjectionWeight.shape == [channels, channels] && cosine.shape == [channels])
    let simdgroups = simdgroups ?? simdgroupsPerThreadgroup
    let rowCount = input.size / channels
    let threadgroups = (rowCount + 8 * simdgroups - 1) / (8 * simdgroups)
    let half = { (array: MLXArray) in array.asType(.float16) }
    return (kernel ?? Self.kernel)(
      [
        input, half(denseExpansionWeight), half(groupedProjectionWeight), half(outputProjectionWeight),
        half(cosine), NeuralRenderingKernelParameters.array([UInt32(rowCount)]),
      ],
      template: [("channels", channels), ("simdgroups", simdgroups)],
      grid: (threadgroups * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
