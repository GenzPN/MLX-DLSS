import Foundation
import MLX

/// One Metal kernel per 8×8 window for the single-head 32-channel window
/// blocks (blocks 0–4, 67–70 and the recovered head): FFN, cosine residual,
/// QKV, vendor cosine normalisation, attention with bias and the recovered
/// bit-affine softmax, projection, residual and E4M3 publication.
///
/// Eight simdgroups share one window; each owns one window row (8 tokens) and
/// keeps it in simdgroup-matrix registers from the input load to the output
/// store. Threadgroup memory holds the window's normalised keys and published
/// values (8 KB) and 512 bytes of scratch per simdgroup (12 KB in all). The
/// arithmetic is copied from the per-operation kernels so the rounding points
/// stay identical: the FFN accumulates in half like the fused feed-forward
/// kernels (the hidden activation is produced in two 64-wide chunks feeding
/// one accumulator, which keeps the accumulation order), the attention matmuls
/// accumulate in float like the MLX GEMMs they replace (half operands into
/// float accumulators, bit-identical to converting them first:
/// `testMixedPrecisionMMAMatchesConvertedFloat`), the per-token norms follow
/// the vendor's xor reduction over the four lanes of a fragment row, and the
/// bit-affine softmax totals are summed in the recovered sequential order.
/// The lanes exchange norm partials and softmax rows through threadgroup
/// memory rather than lane shuffles: the 32 shuffles of each row's total took
/// about 12% of the kernel. 2026-10-17, M2 Max, 1088x1920: 13.6 ms (four
/// simdgroups with threadgroup scratch) -> 10.1 ms -> 8.8 ms (scratch totals and
/// norms, cheaper E4M3), bit-identical (`NeuralRenderingKernelVariantTests`).
enum NeuralRenderingFusedWindowBlock {
  static let channels = 32
  static let hiddenChannels = 128
  static let windowSize = 8
  /// Window rows per simdgroup: 1 (eight simdgroups per window, default) or 2
  /// (four simdgroups with twice the registers each, 16-23% slower at 1088x1920).
  /// `MLXDLSS_1H_ROWS` overrides it (diagnostics).
  nonisolated(unsafe) static var rowsPerSimdgroup: Int =
    ProcessInfo.processInfo.environment["MLXDLSS_1H_ROWS"].flatMap(Int.init).flatMap { [1, 2].contains($0) ? $0 : nil } ?? 1
  static var simdgroupsPerWindow: Int { 8 / rowsPerSimdgroup }
  /// The stem kernel (adapter, block 0, publication, pool) and the post kernel
  /// (merge, block, head); `MLXDLSS_FUSED_1H_GLUE=0` restores the separate kernels.
  nonisolated(unsafe) static var fusedGlueEnabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_FUSED_1H_GLUE"] != "0"
  /// The scratch-array reduction toggle of the earlier kernel; this kernel always
  /// reduces in registers, so `MLXDLSS_WINDOW_SHUFFLE` no longer changes it.
  nonisolated(unsafe) static var shuffleReciprocals: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_WINDOW_SHUFFLE"] != "0"

  private static let kernel = makeKernel(name: "mlxdlss_fused_window_block_1h32", source: source)

  /// Builds the kernel from `source` (variants of it in the timing probes).
  static func makeKernel(name: String, source: String) -> MLXFast.MLXFastKernel {
    MLXFast.metalKernel(
      name: name,
      inputNames: [
        "input", "expansion", "projection", "feedForwardCosine", "qkvWeight",
        "attentionScale", "attentionBias", "attentionProjection", "attentionCosine", "params",
      ],
      outputNames: ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  static let source = #"""
      // Threadgroup memory: 6144 halfs, the window's normalised keys (0..2048),
      // published values (2048..4096) and 256 halfs of scratch per simdgroup
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
      const bool publish = params[4] != 0u;
      const uint windowsX = (width + padLeft + 7) / 8;
      const uint windowIndex = threadgroup_position_in_grid.x;
      const uint windowY = windowIndex / windowsX;
      const uint windowX = windowIndex % windowsX;
      const int rowY0 = int(windowY * 8) - int(padTop);
      const int colX0 = int(windowX * 8) - int(padLeft);
      const int tokenX = colX0 + int(fragmentRow);
      const device half2* input2 = (const device half2*)input;
      device half2* output2 = (device half2*)output;
      const device half2* feedForwardCosine2 = (const device half2*)feedForwardCosine;
      const device half2* attentionCosine2 = (const device half2*)attentionCosine;
      const device half2* attentionBias2 = (const device half2*)attentionBias;
      const half scale = attentionScale[0];

      simdgroup_matrix<half, 8, 8> xt[rowsPerSimdgroup][4];
      simdgroup_matrix<half, 8, 8> ft[rowsPerSimdgroup][4];
      simdgroup_matrix<half, 8, 8> qt[rowsPerSimdgroup][4];
      simdgroup_matrix<half, 8, 8> at[rowsPerSimdgroup][4];
      bool inside[rowsPerSimdgroup];

      // Phase 0: window rows straight into the fragments (zeros outside the image).
      _Pragma("clang loop unroll(full)")
      for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
        const int y = rowY0 + int(simd * rowsPerSimdgroup + rr);
        inside[rr] = y >= 0 && y < int(height) && tokenX >= 0 && tokenX < int(width);
        const uint base = inside[rr] ? (uint(y) * width + uint(tokenX)) * 16 + fragmentColumn / 2 : 0;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
          const half2 value = inside[rr] ? input2[base + k * 4] : half2(0.0h);
          xt[rr][k].thread_elements()[0] = value.x;
          xt[rr][k].thread_elements()[1] = value.y;
        }
      }

      // Phase 1: feed-forward. Expansion (half accumulation, gate, E4M3) and
      // projection over two 64-wide hidden chunks into one accumulator per tile.
      // Each gated 8x8 hidden tile is the next k-step operand of the projection
      // (the same accumulation order as staging the chunk), and the weights are
      // read from device memory, where they stay cached.
      simdgroup_matrix<half, 8, 8> pacc[rowsPerSimdgroup][4];
      _Pragma("clang loop unroll(full)")
      for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          pacc[rr][o].thread_elements()[0] = 0.0h;
          pacc[rr][o].thread_elements()[1] = 0.0h;
        }
      }
      for (uint chunk = 0; chunk < 2; ++chunk) {
        _Pragma("clang loop unroll(full)")
        for (uint c = 0; c < 8; ++c) {
          simdgroup_matrix<half, 8, 8> hidden[rowsPerSimdgroup];
          _Pragma("clang loop unroll(full)")
          for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
            hidden[rr].thread_elements()[0] = 0.0h;
            hidden[rr].thread_elements()[1] = 0.0h;
          }
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 4; ++k) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, expansion + k * 8 * 128 + chunk * 64 + c * 8, 128, ulong2(0), false);
            _Pragma("clang loop unroll(full)")
            for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
              simdgroup_multiply_accumulate(hidden[rr], xt[rr][k], right, hidden[rr]);
            }
          }
          _Pragma("clang loop unroll(full)")
          for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 2; ++e) {
              half value = hidden[rr].thread_elements()[e];
              half clamped = clamp(value, half(-4.0), half(4.0));
              half linear = fma(abs(clamped), half(-0.055908203125), half(0.447265625));
              half gate = fma(clamped, linear, half(0.89453125));
              hidden[rr].thread_elements()[e] = half(float(mlxdlss_e4m3(value * gate)));
            }
          }
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < 4; ++o) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, projection + chunk * 2048 + c * 8 * 32 + o * 8, 32, ulong2(0), false);
            _Pragma("clang loop unroll(full)")
            for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
              simdgroup_multiply_accumulate(pacc[rr][o], hidden[rr], right, pacc[rr][o]);
            }
          }
        }
      }
      // residual: F = branch + X * cos (two half roundings, like the MLX path)
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        const half2 cosine = feedForwardCosine2[(o * 8 + fragmentColumn) / 2];
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half skip = xt[rr][o].thread_elements()[e] * cosine[e];
            ft[rr][o].thread_elements()[e] = pacc[rr][o].thread_elements()[e] + skip;
          }
        }
      }

      // Phase 2: QKV projection (float accumulation, half result), one 32-column
      // part at a time; vendor cosine normalisation of q (scaled) and k, E4M3 of v.
      // A token's 32 values sit on the four lanes of its fragment row (two
      // columns of each tile), so its squared norm is the vendor's two-step
      // xor reduction over those lanes (lane bit 3, then bit 0). The lanes swap
      // their partial sums through the simdgroup's own value rows, which part 2
      // overwrites only afterwards; shuffles cost more than these stores.
      for (uint part = 0; part < 3; ++part) {
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
          simdgroup_matrix<half, 8, 8> result[4];
          _Pragma("clang loop unroll(full)")
          for (uint o = 0; o < 4; ++o) {
            simdgroup_matrix<float, 8, 8> acc;
            acc.thread_elements()[0] = 0.0f;
            acc.thread_elements()[1] = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (uint k = 0; k < 4; ++k) {
              simdgroup_matrix<half, 8, 8> right;
              simdgroup_load(right, qkvWeight + k * 8 * 96 + (part * 4 + o) * 8, 96, ulong2(0), false);
              simdgroup_multiply_accumulate(acc, ft[rr][k], right, acc);
            }
            result[o].thread_elements()[0] = half(acc.thread_elements()[0]);
            result[o].thread_elements()[1] = half(acc.thread_elements()[1]);
          }
          threadgroup half* destination = (part == 1 ? K : V) + (simd * rowsPerSimdgroup + rr) * 8 * 32;
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
            // (p0 + p2) + (p1 + p3), the same value on all four lanes.
            threadgroup half2* partials = (threadgroup half2*)(V + (simd * rowsPerSimdgroup + rr) * 8 * 32) + fragmentRow * 4;
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
                qt[rr][o] = result[o];
              } else {
                simdgroup_store(result[o], destination + o * 8, 32, ulong2(0), false);
              }
            }
          } else {
            _Pragma("clang loop unroll(full)")
            for (uint o = 0; o < 4; ++o) {
              _Pragma("clang loop unroll(full)")
              for (uint e = 0; e < 2; ++e) {
                result[o].thread_elements()[e] = half(float(mlxdlss_e4m3(result[o].thread_elements()[e])));
              }
              simdgroup_store(result[o], destination + o * 8, 32, ulong2(0), false);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Phase 3: per window row, scores = q · kᵀ (float accumulation), bias,
      // the bit-affine softmax weights in the matmul epilogue, the sequential
      // half total of each query row, E4M3 probabilities in registers,
      // attended = E4M3(P · v). For the total every lane reads its query row
      // back from the simdgroup's scratch, 32 keys at a time.
      threadgroup half* rowScratch = arena + 4096 + simd * 256;
      _Pragma("clang loop unroll(full)")
      for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
        const uint token0 = (simd * rowsPerSimdgroup + rr) * 8;
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
            simdgroup_multiply_accumulate(acc, qt[rr][k], right, acc);
          }
          half2 score = half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1]))
            + attentionBias2[((token0 + fragmentRow) * 64 + j * 8 + fragmentColumn) / 2];
          half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
          affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
          // The vendor's (packed << 5) + 0x7FF88000 on the half pair: within the
          // clamp range (bits 0x3C20..0x3E47) the low half always carries one
          // into the high half and shifts 7 into it, so each half becomes
          // (bits << 5) ^ 0x8000 on its own.
          half2 weight = as_type<half2>(ushort2((as_type<ushort2>(affine) << 5) ^ ushort2(0x8000)));
          weights[j].thread_elements()[0] = weight.x;
          weights[j].thread_elements()[1] = weight.y;
        }
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
        // The weights lie in [2^-14, 9.75] and the total is at least the
        // largest of them, so the probabilities are non-negative and at most
        // about 1; P · v stays within 64 * 1.125 * 448 and cannot be NaN.
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
          at[rr][o].thread_elements()[0] = mlxdlss_e4m3_no_nan(half(acc.thread_elements()[0]));
          at[rr][o].thread_elements()[1] = mlxdlss_e4m3_no_nan(half(acc.thread_elements()[1]));
        }
      }

      // Phase 4: output projection (float accumulation), residual with the
      // feed-forward output, publication, store from the fragments.
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        simdgroup_matrix<float, 8, 8> acc[rowsPerSimdgroup];
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
          acc[rr].thread_elements()[0] = 0.0f;
          acc[rr].thread_elements()[1] = 0.0f;
        }
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
          simdgroup_matrix<half, 8, 8> right;
          simdgroup_load(right, attentionProjection + k * 8 * 32 + o * 8, 32, ulong2(0), false);
          _Pragma("clang loop unroll(full)")
          for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
            simdgroup_multiply_accumulate(acc[rr], at[rr][k], right, acc[rr]);
          }
        }
        const half2 cosine = attentionCosine2[(o * 8 + fragmentColumn) / 2];
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
          half2 value;
          _Pragma("clang loop unroll(full)")
          for (uint e = 0; e < 2; ++e) {
            const half projected = half(acc[rr].thread_elements()[e]);
            const half skip = ft[rr][o].thread_elements()[e] * cosine[e];
            value[e] = projected + skip;
          }
          if (publish) {
            value = half2(half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))));
          }
          if (inside[rr]) {
            const int y = rowY0 + int(simd * rowsPerSimdgroup + rr);
            output2[(uint(y) * width + uint(tokenX)) * 16 + (o * 8 + fragmentColumn) / 2] = value;
          }
        }
      }
      """#

  // MARK: Variants with the neighbouring glue folded in

  /// The eight weight tensors of a single-head window block, in kernel order.
  struct Weights {
    let expansion: MLXArray
    let feedForwardProjection: MLXArray
    let feedForwardCosine: MLXArray
    let qkv: MLXArray
    let attentionScale: MLXArray
    let attentionBias: MLXArray
    let attentionProjection: MLXArray
    let attentionCosine: MLXArray

    var kernelInputs: [MLXArray] {
      let half = { (array: MLXArray) in array.asType(.float16) }
      return [
        half(expansion), half(feedForwardProjection), half(feedForwardCosine), half(qkv), half(attentionScale),
        half(attentionBias.reshaped([64 * 64])), half(attentionProjection), half(attentionCosine),
      ]
    }
  }

  /// How a variant produces the window rows.
  enum Prologue {
    /// `input` `[1, H, W, 32]`, as `source`.
    case load
    /// `input` `[1, H, W, 16]` times `adapter` `[16, 32]` (the block-0 input adapter).
    case adapter
    /// `low[y / 2, x / 2] (* lowScale) + skip * skipScale`, optionally E4M3-published
    /// (`NeuralRenderingFusedGlue.upsampleMerge`).
    case merge
  }

  /// What a variant does with the block output.
  enum Epilogue {
    /// Store it, E4M3-published when asked, as `source`.
    case store
    /// The published 2×2 mean (`NeuralRenderingFusedGlue.publishAndPool`), and
    /// the published output itself when asked.
    case pool
    /// The post block's output head (`NeuralRenderingPostBlock.head`, fused form).
    case head
  }

  private static let weightInputNames = [
    "expansion", "projection", "feedForwardCosine", "qkvWeight", "attentionScale", "attentionBias",
    "attentionProjection", "attentionCosine",
  ]

  /// Replaces one exact piece of `source`; the variants fail loudly if it drifts.
  private static func replacing(_ text: String, _ target: String, with replacement: String) -> String {
    precondition(text.components(separatedBy: target).count == 2, "fused window block variant: anchor not unique")
    return text.replacingOccurrences(of: target, with: replacement)
  }

  // `source` as the kernel sees it: the literal's six-space margin is stripped.
  private static let inputDeclaration = "const device half2* input2 = (const device half2*)input;\n"
  private static let loadPrologue: String = {
    let start = source.range(of: "// Phase 0: window rows straight into the fragments")!.lowerBound
    let end = source.range(of: "// Phase 1: feed-forward.")!.lowerBound
    return String(source[start..<end])
  }()
  private static let storeTail = #"""
          if (publish) {
            value = half2(half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))));
          }
          if (inside[rr]) {
            const int y = rowY0 + int(simd * rowsPerSimdgroup + rr);
            output2[(uint(y) * width + uint(tokenX)) * 16 + (o * 8 + fragmentColumn) / 2] = value;
          }

      """#
  /// Phase 4 keeps the block output in the dead feed-forward fragments.
  private static let keepTail = #"""
          ft[rr][o].thread_elements()[0] = value.x;
          ft[rr][o].thread_elements()[1] = value.y;

      """#

  private static let adapterPrologue = #"""
      // Phase 0 (adapter): [8 tokens, 16] x [16, 32] with float accumulation
      // and a half result like the MLX GEMM (zeros outside the image).
      const device half2* input2 = (const device half2*)input;
      _Pragma("clang loop unroll(full)")
      for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
        const int y = rowY0 + int(simd * rowsPerSimdgroup + rr);
        inside[rr] = y >= 0 && y < int(height) && tokenX >= 0 && tokenX < int(width);
        const uint base = inside[rr] ? (uint(y) * width + uint(tokenX)) * 8 + fragmentColumn / 2 : 0;
        simdgroup_matrix<half, 8, 8> raw[2];
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 2; ++k) {
          const half2 value = inside[rr] ? input2[base + k * 4] : half2(0.0h);
          raw[k].thread_elements()[0] = value.x;
          raw[k].thread_elements()[1] = value.y;
        }
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f;
          acc.thread_elements()[1] = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (uint k = 0; k < 2; ++k) {
            simdgroup_matrix<half, 8, 8> right;
            simdgroup_load(right, adapter + k * 8 * 32 + o * 8, 32, ulong2(0), false);
            simdgroup_multiply_accumulate(acc, raw[k], right, acc);
          }
          xt[rr][o].thread_elements()[0] = half(acc.thread_elements()[0]);
          xt[rr][o].thread_elements()[1] = half(acc.thread_elements()[1]);
        }
      }


      """#

  private static let mergePrologue = #"""
      // Phase 0 (merge): low[y / 2, x / 2] (* lowScale) + skip * skipScale in
      // half, E4M3-published when asked (zeros outside the image).
      const device half2* low2 = (const device half2*)low;
      const device half2* skip2 = (const device half2*)skip;
      const device half2* lowScale2 = (const device half2*)lowScale;
      const device half2* skipScale2 = (const device half2*)skipScale;
      const uint lowWidth = params[5];
      const bool hasLowScale = (params[6] & 1u) != 0u;
      const bool publishMerged = (params[6] & 2u) != 0u;
      _Pragma("clang loop unroll(full)")
      for (uint rr = 0; rr < rowsPerSimdgroup; ++rr) {
        const int y = rowY0 + int(simd * rowsPerSimdgroup + rr);
        inside[rr] = y >= 0 && y < int(height) && tokenX >= 0 && tokenX < int(width);
        const uint skipBase = inside[rr] ? (uint(y) * width + uint(tokenX)) * 16 + fragmentColumn / 2 : 0;
        const uint lowBase = inside[rr] ? (uint(y / 2) * lowWidth + uint(tokenX / 2)) * 16 + fragmentColumn / 2 : 0;
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < 4; ++k) {
          half2 value = half2(0.0h);
          if (inside[rr]) {
            half2 upsampled = low2[lowBase + k * 4];
            if (hasLowScale) { upsampled = upsampled * lowScale2[(k * 8 + fragmentColumn) / 2]; }
            half2 scaledSkip = skip2[skipBase + k * 4] * skipScale2[(k * 8 + fragmentColumn) / 2];
            value = upsampled + scaledSkip;
            if (publishMerged) {
              value = half2(half(float(mlxdlss_e4m3(value.x))), half(float(mlxdlss_e4m3(value.y))));
            }
          }
          xt[rr][k].thread_elements()[0] = value.x;
          xt[rr][k].thread_elements()[1] = value.y;
        }
      }


      """#

  private static let poolEpilogue = #"""

      // Pool epilogue (one window row per simdgroup, window origins and image
      // extents even): the published output when asked, then the published
      // 2x2 mean in MLX's float16 order (row-major half sums, then x 0.25).
      // The window goes to the idle arena; each thread then pools one
      // channel pair of one output token.
      device half2* pooled2 = (device half2*)pooled;
      const bool emitFull = params[7] != 0u;
      const int y = rowY0 + int(simd);
      if (emitFull && inside[0]) {
        _Pragma("clang loop unroll(full)")
        for (uint o = 0; o < 4; ++o) {
          output2[(uint(y) * width + uint(tokenX)) * 16 + (o * 8 + fragmentColumn) / 2] = half2(
            half(float(mlxdlss_e4m3(ft[0][o].thread_elements()[0]))),
            half(float(mlxdlss_e4m3(ft[0][o].thread_elements()[1]))));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      _Pragma("clang loop unroll(full)")
      for (uint o = 0; o < 4; ++o) {
        simdgroup_store(ft[0][o], arena + simd * 256 + o * 8, 32, ulong2(0), false);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint pooledToken = simd * 2 + lane / 16;
      const uint pooledY = pooledToken / 4;
      const uint pooledX = pooledToken % 4;
      const uint pair = lane % 16;
      const int sourceY = rowY0 + int(pooledY * 2);
      const int sourceX = colX0 + int(pooledX * 2);
      if (sourceY >= 0 && sourceY < int(height) && sourceX >= 0 && sourceX < int(width)) {
        const threadgroup half2* window2 = (const threadgroup half2*)arena;
        const uint source = (pooledY * 2 * 8 + pooledX * 2) * 16 + pair;
        const half2 v00 = window2[source];
        const half2 v01 = window2[source + 16];
        const half2 v10 = window2[source + 8 * 16];
        const half2 v11 = window2[source + 8 * 16 + 16];
        half2 sum = v00;
        sum = sum + v01;
        sum = sum + v10;
        sum = sum + v11;
        sum = sum * 0.25h;
        pooled2[(uint(sourceY / 2) * (width / 2) + uint(sourceX / 2)) * 16 + pair] =
          half2(half(float(mlxdlss_e4m3(sum.x))), half(float(mlxdlss_e4m3(sum.y))));
      }
      """#

  private static let headEpilogue = #"""

      // Head epilogue: features x [32, 8] (float accumulation, half result),
      // output = columns 0..3 + columns 4..7 (lanes with bit 3 hold the latter).
      simdgroup_matrix<float, 8, 8> headAcc;
      headAcc.thread_elements()[0] = 0.0f;
      headAcc.thread_elements()[1] = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint k = 0; k < 4; ++k) {
        simdgroup_matrix<half, 8, 8> right;
        simdgroup_load(right, headWeight + k * 8 * 8, 8, ulong2(0), false);
        simdgroup_multiply_accumulate(headAcc, ft[0][k], right, headAcc);
      }
      const half2 projected = half2(half(headAcc.thread_elements()[0]), half(headAcc.thread_elements()[1]));
      const half2 upper = simd_shuffle_xor(projected, ushort(8));
      if ((lane & 8) == 0 && inside[0]) {
        const int y = rowY0 + int(simd);
        output2[(uint(y) * width + uint(tokenX)) * 2 + fragmentColumn / 2] = projected + upper;
      }
      """#

  /// `source` with the prologue and epilogue swapped (one window row per simdgroup).
  static func variantSource(_ prologue: Prologue, _ epilogue: Epilogue) -> String {
    var text = source
    switch prologue {
    case .load: break
    case .adapter:
      text = replacing(replacing(text, inputDeclaration, with: ""), loadPrologue, with: adapterPrologue)
    case .merge:
      text = replacing(replacing(text, inputDeclaration, with: ""), loadPrologue, with: mergePrologue)
    }
    switch epilogue {
    case .store: break
    case .pool: text = replacing(text, storeTail, with: keepTail) + poolEpilogue
    case .head: text = replacing(text, storeTail, with: keepTail) + headEpilogue
    }
    return text
  }

  private struct VariantKey: Hashable {
    let prologue: Prologue
    let epilogue: Epilogue
  }

  nonisolated(unsafe) private static var variantKernels: [VariantKey: MLXFast.MLXFastKernel] = [:]
  private static let variantLock = NSLock()

  private static func variantKernel(_ prologue: Prologue, _ epilogue: Epilogue) -> MLXFast.MLXFastKernel {
    variantLock.lock()
    defer { variantLock.unlock() }
    let key = VariantKey(prologue: prologue, epilogue: epilogue)
    if let kernel = variantKernels[key] { return kernel }
    let kernel = makeVariantKernel(
      prologue, epilogue, name: "mlxdlss_fused_window_block_1h32_\(prologue)_\(epilogue)",
      source: variantSource(prologue, epilogue))
    variantKernels[key] = kernel
    return kernel
  }

  /// Builds a variant kernel from `source` (earlier versions of it in the A/B probes).
  static func makeVariantKernel(
    _ prologue: Prologue, _ epilogue: Epilogue, name: String, source: String
  ) -> MLXFast.MLXFastKernel {
    let inputs: [String]
    switch prologue {
    case .load: inputs = ["input"]
    case .adapter: inputs = ["input", "adapter"]
    case .merge: inputs = ["low", "skip", "lowScale", "skipScale"]
    }
    return MLXFast.metalKernel(
      name: name,
      inputNames: inputs + weightInputNames + (epilogue == .head ? ["headWeight"] : []) + ["params"],
      outputNames: epilogue == .pool ? ["output", "pooled"] : ["output"],
      source: source,
      header: "#include <metal_simdgroup_matrix>\n" + NeuralRenderingTransformerOperations.e4m3MetalHeaderText
    )
  }

  /// The rows a variant reads: the block input or the network input (`.load`,
  /// `.adapter`), or the decoder tensor and its skip (`.merge`).
  enum Rows {
    case input(MLXArray)
    case adapter(input: MLXArray, weight: MLXArray)
    case merge(low: MLXArray, skip: MLXArray, lowScale: MLXArray?, skipScale: MLXArray, publish: Bool)
  }

  /// A single-head window block with its glue: `rows` as `Rows` describes,
  /// then `.store` (`[1, H, W, 32]`, published when `publish`), `.pool`
  /// (`[pooled, full]`: the published `[1, H/2, W/2, 32]` mean and, when
  /// `publish`, the published block output, else a one-element placeholder)
  /// or `.head` (`[1, H, W, 4]` with `headWeight` `[32, 8]`).
  static func applyVariant(
    _ rows: Rows, weights: Weights, windowOrigin: NeuralRenderingWindowOrigin, epilogue: Epilogue,
    publish: Bool, headWeight: MLXArray? = nil, kernel: MLXFast.MLXFastKernel? = nil
  ) -> [MLXArray] {
    let half = { (array: MLXArray) in array.asType(.float16) }
    let prologue: Prologue
    let rowInputs: [MLXArray]
    let height: Int, width: Int
    var lowWidth = 0, mergeFlags = 0
    switch rows {
    case .input(let input):
      precondition(input.ndim == 4 && input.shape[0] == 1 && input.shape[3] == channels && input.dtype == .float16)
      prologue = .load
      rowInputs = [input]
      (height, width) = (input.shape[1], input.shape[2])
    case .adapter(let input, let weight):
      precondition(input.ndim == 4 && input.shape[0] == 1 && input.shape[3] == 16 && input.dtype == .float16)
      precondition(weight.shape == [16, channels])
      prologue = .adapter
      rowInputs = [input, half(weight)]
      (height, width) = (input.shape[1], input.shape[2])
    case .merge(let low, let skip, let lowScale, let skipScale, let publishMerged):
      precondition(low.ndim == 4 && skip.ndim == 4 && low.shape[0] == 1 && skip.shape[0] == 1)
      precondition(low.dtype == .float16 && skip.dtype == .float16)
      precondition(low.shape[3] == channels && skip.shape[3] == channels && skipScale.shape == [channels])
      precondition(low.shape[1] * 2 >= skip.shape[1] && low.shape[2] * 2 >= skip.shape[2])
      prologue = .merge
      rowInputs = [low, skip, half(lowScale ?? skipScale), half(skipScale)]
      (height, width) = (skip.shape[1], skip.shape[2])
      lowWidth = low.shape[2]
      mergeFlags = (lowScale != nil ? 1 : 0) | (publishMerged ? 2 : 0)
    }
    let padTop = -windowOrigin.y, padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    if epilogue == .pool {
      precondition(height.isMultiple(of: 2) && width.isMultiple(of: 2) && padTop.isMultiple(of: 2) && padLeft.isMultiple(of: 2))
    }
    if epilogue == .head { precondition(headWeight?.shape == [channels, 8]) }
    let windowCount = ((height + padTop + windowSize - 1) / windowSize) * ((width + padLeft + windowSize - 1) / windowSize)
    let params = NeuralRenderingKernelParameters.array([
      UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), publish && epilogue == .store ? 1 : 0,
      UInt32(lowWidth), UInt32(mergeFlags), publish && epilogue == .pool ? 1 : 0,
    ])
    let outputShapes: [[Int]]
    switch epilogue {
    case .store: outputShapes = [[1, height, width, channels]]
    case .pool: outputShapes = [publish ? [1, height, width, channels] : [1], [1, height / 2, width / 2, channels]]
    case .head: outputShapes = [[1, height, width, 4]]
    }
    let outputs = (kernel ?? variantKernel(prologue, epilogue))(
      rowInputs + weights.kernelInputs + (headWeight.map { [half($0)] } ?? []) + [params],
      template: [("shuffleReciprocals", shuffleReciprocals), ("rowsPerSimdgroup", 1)],
      grid: (windowCount * 256, 1, 1),
      threadGroup: (256, 1, 1),
      outputShapes: outputShapes,
      outputDTypes: outputShapes.map { _ in .float16 }
    )
    return epilogue == .pool ? [outputs[1], outputs[0]] : outputs
  }

  /// Whole window block for a `[1, H, W, 32]` half-precision input.
  static func apply(
    _ input: MLXArray,
    expansionWeight: MLXArray,
    feedForwardProjectionWeight: MLXArray,
    feedForwardCosine: MLXArray,
    qkvWeight: MLXArray,
    attentionScale: MLXArray,
    attentionBias: MLXArray,
    attentionProjectionWeight: MLXArray,
    attentionCosine: MLXArray,
    windowOrigin: NeuralRenderingWindowOrigin,
    publish: Bool,
    kernel: MLXFast.MLXFastKernel? = nil,
    rowsPerSimdgroup rows: Int? = nil
  ) -> MLXArray {
    let rows = rows ?? rowsPerSimdgroup
    let simdgroups = 8 / rows
    precondition(input.ndim == 4 && input.shape[0] == 1 && input.shape[3] == channels)
    precondition(input.dtype == .float16)
    let height = input.shape[1]
    let width = input.shape[2]
    let padTop = -windowOrigin.y
    let padLeft = -windowOrigin.x
    precondition(padTop >= 0 && padLeft >= 0)
    let windowsY = (height + padTop + windowSize - 1) / windowSize
    let windowsX = (width + padLeft + windowSize - 1) / windowSize
    let windowCount = windowsY * windowsX
    let half = { (array: MLXArray) in array.asType(.float16) }
    // Geometry travels in a buffer rather than template values so the kernel is
    // compiled once per process instead of once per shape and window phase.
    let inputs = [
      input, half(expansionWeight), half(feedForwardProjectionWeight), half(feedForwardCosine),
      half(qkvWeight), half(attentionScale), half(attentionBias.reshaped([64 * 64])),
      half(attentionProjectionWeight), half(attentionCosine),
    ]
    let params = NeuralRenderingKernelParameters.array([UInt32(height), UInt32(width), UInt32(padTop), UInt32(padLeft), publish ? UInt32(1) : UInt32(0)])
    return (kernel ?? Self.kernel)(
      inputs + [params],
      template: [("shuffleReciprocals", shuffleReciprocals), ("rowsPerSimdgroup", rows)],
      grid: (windowCount * 32 * simdgroups, 1, 1),
      threadGroup: (32 * simdgroups, 1, 1),
      outputShapes: [input.shape],
      outputDTypes: [.float16]
    )[0]
  }
}
