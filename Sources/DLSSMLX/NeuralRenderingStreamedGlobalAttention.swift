import Foundation
import MLX

/// Keeps an eight-query tile on-chip for short sequences. Larger sequences use
/// two passes over K, retaining the sequential half denominator and E4M3
/// probabilities without allocating quadratic score/probability arrays.
enum NeuralRenderingStreamedGlobalAttention {
  private static let mode = ProcessInfo.processInfo.environment["MLXDLSS_STREAMED_GLOBAL_ATTENTION"]
  /// The v2 kernels convert scores to vendor weights and probabilities with
  /// every thread and keep only the half denominator sum sequential, so the
  /// resident tile serves up to 1024 tokens; `MLXDLSS_GLOBAL_ATTENTION_V2=0`
  /// restores the earlier kernels and their 512-token limit (diagnostics).
  nonisolated(unsafe) static var v2Enabled: Bool =
    ProcessInfo.processInfo.environment["MLXDLSS_GLOBAL_ATTENTION_V2"] != "0"
  static var residentMaxTokens: Int { v2Enabled ? 1024 : 512 }

  /// Longer sequences keep the materialized attention unless streaming is forced.
  static func isEnabled(tokens: Int) -> Bool {
    mode == "1" || (mode != "0" && tokens <= residentMaxTokens)
  }

  static func apply(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    precondition(query.ndim == 4 && query.shape.last == 32)
    precondition(query.shape == key.shape && query.shape == value.shape)
    precondition(query.dtype == .float16 && key.dtype == .float16 && value.dtype == .float16)
    let tokens = query.dim(2)
    precondition(tokens > 0 && tokens.isMultiple(of: 2))
    if v2Enabled { return applyV2(query: query, key: key, value: value) }
    if tokens <= 512 {
      let paddedTokens = (tokens + 7) / 8 * 8
      func pad(_ x: MLXArray) -> MLXArray {
        paddedTokens == tokens ? x : padded(x, widths: [[0, 0], [0, 0], [0, paddedTokens - tokens], [0, 0]])
      }
      let output = resident([pad(query), pad(key), pad(value)],
        template: [("tokens", tokens), ("pitch", paddedTokens)],
        grid: (128, paddedTokens / 8, query.dim(0) * query.dim(1)), threadGroup: (128, 1, 1),
        outputShapes: [[query.dim(0), query.dim(1), paddedTokens, 32]], outputDTypes: [.float16])[0]
      return output[0..., 0..., 0..<tokens, 0...]
    }
    let grid = (128, (tokens + 7) / 8, query.dim(0) * query.dim(1))
    let reciprocal = denominator([query, key], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [Array(query.shape.dropLast())], outputDTypes: [.float16])[0]
    return attention([query, key, value, reciprocal], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [query.shape], outputDTypes: [.float16])[0]
  }

  private static let header = NeuralRenderingTransformerOperations.e4m3MetalHeaderText + "\n" + #"""
    #include <metal_simdgroup_matrix>
    using namespace metal;
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)

    METAL_FUNC half2 vendor_weights(half2 score) {
      half2 affine = fma(score, half2(0.044921875h), half2(1.30078125h));
      affine = clamp(affine, half2(1.03125h), half2(1.5693359375h));
      ushort2 bits = as_type<ushort2>(affine);
      uint packed = uint(bits.x) | (uint(bits.y) << 16);
      uint transformed = (packed << 5) + 0x7FF88000;
      return as_type<half2>(ushort2(ushort(transformed), ushort(transformed >> 16)));
    }

    METAL_FUNC void score_tile(
      threadgroup half* Q, threadgroup half* K, threadgroup half* S,
      const device half* key, uint headBase, uint token0, uint tokens, uint tid, uint simd
    ) {
      for (uint i = tid; i < 32 * 32; i += 128) {
        uint token = token0 + i / 32;
        K[i] = token < tokens ? key[headBase + token * 32 + i % 32] : 0.0h;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f;
      acc.thread_elements()[1] = 0.0f;
      for (uint c = 0; c < 32; c += 8) {
        simdgroup_matrix<half, 8, 8> q, k;
        simdgroup_load(q, Q + c, 32, ulong2(0), false);
        simdgroup_load(k, K + simd * 8 * 32 + c, 32, ulong2(0), true);
        simdgroup_matrix<float, 8, 8> left, right;
        left.thread_elements()[0] = float(q.thread_elements()[0]);
        left.thread_elements()[1] = float(q.thread_elements()[1]);
        right.thread_elements()[0] = float(k.thread_elements()[0]);
        right.thread_elements()[1] = float(k.thread_elements()[1]);
        simdgroup_multiply_accumulate(acc, left, right, acc);
      }
      simdgroup_matrix<half, 8, 8> scores;
      scores.thread_elements()[0] = clamp(half(acc.thread_elements()[0]), -3.0h, 3.0h);
      scores.thread_elements()[1] = clamp(half(acc.thread_elements()[1]), -3.0h, 3.0h);
      simdgroup_store(scores, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """#

  private static let setup = #"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint tokens = uint(query_shape[2]);
    const uint head = threadgroup_position_in_grid.z;
    const uint row0 = threadgroup_position_in_grid.y * 8;
    const uint headBase = head * tokens * 32;
    threadgroup half Q[8 * 32];
    threadgroup half K[32 * 32];
    threadgroup half S[8 * 32];
    for (uint i = tid; i < 8 * 32; i += 128) {
      uint row = row0 + i / 32;
      Q[i] = row < tokens ? query[headBase + row * 32 + i % 32] : 0.0h;
    }
    """#

  private static let resident = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_resident", inputNames: ["query", "key", "value"], outputNames: ["output"],
    source: #"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint headBase = threadgroup_position_in_grid.z * pitch * 32;
      const uint row0 = threadgroup_position_in_grid.y * 8;
      threadgroup half S[8 * pitch];
      for (uint token0 = simd * 8; token0 < pitch; token0 += 32) {
        simdgroup_matrix<float, 8, 8> acc;
        acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> q, k;
          simdgroup_load(q, query + headBase + row0 * 32 + c, 32, ulong2(0), false);
          simdgroup_load(k, key + headBase + token0 * 32 + c, 32, ulong2(0), true);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(q.thread_elements()[0]);
          left.thread_elements()[1] = float(q.thread_elements()[1]);
          right.thread_elements()[0] = float(k.thread_elements()[0]);
          right.thread_elements()[1] = float(k.thread_elements()[1]);
          simdgroup_multiply_accumulate(acc, left, right, acc);
        }
        simdgroup_matrix<half, 8, 8> s;
        s.thread_elements()[0] = clamp(half(acc.thread_elements()[0]), -3.0h, 3.0h);
        s.thread_elements()[1] = clamp(half(acc.thread_elements()[1]), -3.0h, 3.0h);
        simdgroup_store(s, S + token0, pitch, ulong2(0), false);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 8) {
        half total = 0.0h;
        for (uint j = 0; j < tokens; j += 2) {
          half2 weight = vendor_weights(half2(S[tid * pitch + j], S[tid * pitch + j + 1]));
          total += weight.x; total += weight.y;
          S[tid * pitch + j] = weight.x; S[tid * pitch + j + 1] = weight.y;
        }
        half reciprocal = half(1.0f / float(total));
        for (uint j = 0; j < pitch; ++j) {
          S[tid * pitch + j] = j < tokens ? half(float(mlxdlss_e4m3(S[tid * pitch + j] * reciprocal))) : 0.0h;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
      for (uint c = 0; c < pitch; c += 8) {
        simdgroup_matrix<half, 8, 8> p, v;
        simdgroup_load(p, S + c, pitch, ulong2(0), false);
        simdgroup_load(v, value + headBase + c * 32 + simd * 8, 32, ulong2(0), false);
        simdgroup_matrix<float, 8, 8> left, right;
        left.thread_elements()[0] = float(p.thread_elements()[0]);
        left.thread_elements()[1] = float(p.thread_elements()[1]);
        right.thread_elements()[0] = float(v.thread_elements()[0]);
        right.thread_elements()[1] = float(v.thread_elements()[1]);
        simdgroup_multiply_accumulate(acc, left, right, acc);
      }
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(acc.thread_elements()[1]))));
      simdgroup_store(result, output + headBase + row0 * 32 + simd * 8, 32, ulong2(0), false);
      """#, header: header)

  private static let denominator = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_denominator", inputNames: ["query", "key"], outputNames: ["reciprocal"],
    source: setup + #"""
      half total = 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        if (tid < 8) {
          for (uint j = 0; j < min(32u, tokens - token0); j += 2) {
            half2 weight = vendor_weights(half2(S[tid * 32 + j], S[tid * 32 + j + 1]));
            total += weight.x;
            total += weight.y;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid < 8 && row0 + tid < tokens) {
        reciprocal[head * tokens + row0 + tid] = half(1.0f / float(total));
      }
      """#, header: header)

  private static let attention = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_values", inputNames: ["query", "key", "value", "reciprocal"], outputNames: ["output"],
    source: setup + #"""
      simdgroup_matrix<float, 8, 8> attended;
      attended.thread_elements()[0] = 0.0f;
      attended.thread_elements()[1] = 0.0f;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        if (tid < 8) {
          half r = row0 + tid < tokens ? reciprocal[head * tokens + row0 + tid] : 0.0h;
          for (uint j = 0; j < 32; j += 2) {
            half2 weight = vendor_weights(half2(S[tid * 32 + j], S[tid * 32 + j + 1]));
            S[tid * 32 + j] = token0 + j < tokens ? half(float(mlxdlss_e4m3(weight.x * r))) : 0.0h;
            S[tid * 32 + j + 1] = token0 + j + 1 < tokens ? half(float(mlxdlss_e4m3(weight.y * r))) : 0.0h;
          }
        }
        // K's storage is now free for the matching value rows.
        for (uint i = tid; i < 32 * 32; i += 128) {
          uint token = token0 + i / 32;
          K[i] = token < tokens ? value[headBase + token * 32 + i % 32] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> p, v;
          simdgroup_load(p, S + c, 32, ulong2(0), false);
          simdgroup_load(v, K + c * 32 + simd * 8, 32, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(p.thread_elements()[0]);
          left.thread_elements()[1] = float(p.thread_elements()[1]);
          right.thread_elements()[0] = float(v.thread_elements()[0]);
          right.thread_elements()[1] = float(v.thread_elements()[1]);
          simdgroup_multiply_accumulate(attended, left, right, attended);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[1]))));
      simdgroup_store(result, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < 8 * 32; i += 128) {
        uint row = row0 + i / 32;
        if (row < tokens) output[headBase + row * 32 + i % 32] = S[i];
      }
      """#, header: header)

  // MARK: - v2

  static func applyV2(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    let tokens = query.dim(2)
    if tokens <= residentMaxTokens {
      let batchHeads = query.dim(0) * query.dim(1)
      let pitch = (tokens + 7) / 8 * 8, groups = (pitch / 8 + 3) / 4
      func pad(_ x: MLXArray, _ length: Int) -> MLXArray {
        length == tokens ? x : padded(x, widths: [[0, 0], [0, 0], [0, length - tokens], [0, 0]])
      }
      // The resident kernel's lane-major key and value layouts (see `publishQKV`):
      // key [tile][m2 f2 m1 f1][c e] from token t*8 + f2*4 + f1*2 + e, channel c*8 + m2*4 + m1;
      // value [group][d][m2 f2 m1 f1][i e] from token g*32 + i*8 + m2*4 + m1, channel d*8 + f2*4 + f1*2 + e.
      let keyLanes = contiguous(pad(key, pitch).reshaped([batchHeads, pitch / 8, 2, 2, 2, 4, 2, 4])
        .transposed(0, 1, 6, 2, 7, 3, 5, 4))
      let valueLanes = contiguous(pad(value, groups * 32).reshaped([batchHeads, groups, 4, 2, 4, 4, 2, 2, 2])
        .transposed(0, 1, 5, 3, 6, 4, 7, 2, 8))
      let output = residentV2([pad(query, pitch), keyLanes, valueLanes],
        template: [("tokens", tokens), ("pitch", pitch), ("heads", query.dim(1)), ("tokenMajor", false)],
        grid: (256, pitch / 8, batchHeads), threadGroup: (256, 1, 1),
        outputShapes: [[query.dim(0), query.dim(1), pitch, 32]], outputDTypes: [.float16])[0]
      return output[0..., 0..., 0..<tokens, 0...]
    }
    let grid = (128, (tokens + 7) / 8, query.dim(0) * query.dim(1))
    let reciprocal = denominatorV2([query, key], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [Array(query.shape.dropLast())], outputDTypes: [.float16])[0]
    return attentionV2([query, key, value, reciprocal], grid: grid, threadGroup: (128, 1, 1),
      outputShapes: [query.shape], outputDTypes: [.float16])[0]
  }

  /// Whether `applyProjected` serves `tokens` (the resident v2 kernel).
  static func projectedEnabled(tokens: Int) -> Bool {
    v2Enabled && isEnabled(tokens: tokens) && tokens <= residentMaxTokens
  }

  /// Attention straight from the QKV projection `[B, T, 3 * heads * 32]` to the
  /// token-major result `[B, T, heads * 32]`. One kernel publishes the scaled
  /// cosine queries, cosine keys and E4M3 values (the roundings of
  /// `vendorCosinePublish` and `e4m3RoundTrip`) in the resident kernel's
  /// layouts, and the resident kernel writes its rows in token order, so the
  /// head transposes, the padding and the transpose back are gone.
  static func applyProjected(_ projected: MLXArray, heads: Int, queryScale: MLXArray) -> MLXArray {
    precondition(projected.ndim == 3 && projected.dtype == .float16 && projected.dim(2) == 3 * heads * 32)
    precondition(queryScale.shape == [heads] && heads.isMultiple(of: 8))
    let batch = projected.dim(0), tokens = projected.dim(1)
    precondition(tokens > 0 && tokens.isMultiple(of: 2) && tokens <= residentMaxTokens)
    let batchHeads = batch * heads
    let pitch = (tokens + 7) / 8 * 8, groups = (pitch / 8 + 3) / 4
    let published = publishQKV([projected, queryScale.asType(.float16)],
      template: [("tokens", tokens), ("pitch", pitch), ("heads", heads), ("batchHeads", batchHeads)],
      grid: (256, groups, 3 * batchHeads / 8), threadGroup: (256, 1, 1),
      outputShapes: [[batchHeads, pitch, 32], [batchHeads, pitch * 32], [batchHeads, groups * 1024]],
      outputDTypes: [.float16, .float16, .float16])
    return residentV2(published,
      template: [("tokens", tokens), ("pitch", pitch), ("heads", heads), ("tokenMajor", true)],
      grid: (256, pitch / 8, batchHeads), threadGroup: (256, 1, 1),
      outputShapes: [[batch, tokens, heads * 32]], outputDTypes: [.float16])[0]
  }

  /// A threadgroup publishes one 32-token group of eight heads for one of the
  /// query, key and value parts: every thread runs the per-vector arithmetic of
  /// `vendorCosineNormalize32Kernel` (publish, with the head scale for queries)
  /// or the E4M3 round trip into threadgroup memory, then the threads write
  /// 16-byte chunks of the resident kernel's layouts. A fragment lane `l` holds
  /// row `fm(l)` and the column pair `fn(l)`, `fn(l) + 1`; queries stay
  /// `[head][token][channel]`, keys become `[head][tile][lane][c e]` (the four
  /// transposed 8-channel chunks of a tile), values `[head][group][d][lane][i e]`
  /// (the four token tiles of a group for one 8-channel slice).
  private static let publishQKV = MLXFast.metalKernel(
    name: "mlxdlss_global_publish_qkv", inputNames: ["projected", "scale"], outputNames: ["query", "key", "value"],
    source: #"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint g = threadgroup_position_in_grid.y;
      constexpr uint blocks = uint(batchHeads) / 8;
      constexpr uint tiles = uint(pitch) / 8;
      constexpr uint groups = (tiles + 3) / 4;
      const uint part = threadgroup_position_in_grid.z / blocks;
      const uint block = threadgroup_position_in_grid.z % blocks;
      threadgroup half T[8 * 32 * 32];
      {
        const uint local = tid / 32, r = tid % 32;
        const uint batchHead = block * 8 + local;
        const uint head = batchHead % uint(heads);
        const uint token = g * 32 + r;
        threadgroup half* mine = T + tid * 32;
        if (token >= uint(tokens)) {
          for (uint channel = 0; channel < 32; ++channel) { mine[channel] = 0.0h; }
        } else {
          const uint row = (batchHead / uint(heads)) * uint(tokens) + token;
          const device half4* input4 = (const device half4*)(projected + (row * 3 + part) * uint(heads) * 32 + head * 32);
          half value[32];
          for (uint i = 0; i < 8; ++i) {
            const half4 v = input4[i];
            value[4 * i] = v.x; value[4 * i + 1] = v.y; value[4 * i + 2] = v.z; value[4 * i + 3] = v.w;
          }
          if (part == 2) {
            for (uint channel = 0; channel < 32; ++channel) { mine[channel] = half(float(mlxdlss_e4m3(value[channel]))); }
          } else {
            half partial[4][2];
            for (uint lane = 0; lane < 4; ++lane) {
              for (uint parity = 0; parity < 2; ++parity) {
                uint channel = lane * 2 + parity;
                half first = fma(value[channel + 8], value[channel + 8], value[channel] * value[channel]);
                half second = fma(value[channel + 24], value[channel + 24], value[channel + 16] * value[channel + 16]);
                partial[lane][parity] = first + second;
              }
            }
            half xorTwo[4][2];
            for (uint lane = 0; lane < 4; ++lane) {
              for (uint parity = 0; parity < 2; ++parity) {
                xorTwo[lane][parity] = partial[lane][parity] + partial[lane ^ 2][parity];
              }
            }
            half xorOne[4][2];
            for (uint lane = 0; lane < 4; ++lane) {
              for (uint parity = 0; parity < 2; ++parity) {
                xorOne[lane][parity] = xorTwo[lane][parity] + xorTwo[lane ^ 1][parity];
              }
            }
            half norm = max(xorOne[0][0] + xorOne[0][1], half(0.00006198883056640625));
            half reciprocal = half(metal::fast::rsqrt(float(norm)));
            for (uint channel = 0; channel < 32; ++channel) {
              half normalized = value[channel] * reciprocal;
              if (part == 0) {
                normalized *= half(scale[head]);
              }
              mine[channel] = half(float(mlxdlss_e4m3(normalized)));
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint n = 0; n < 4; ++n) {
        const uint chunk = tid * 4 + n;
        const uint local = chunk / 128, within = chunk % 128;
        const uint batchHead = block * 8 + local;
        const threadgroup half* source = T + local * 1024;
        const uint l = within % 32;
        const uint fm = ((l >> 4) << 2) + ((l >> 1) & 3);
        const uint fn = ((l >> 2) & 2) * 2 + (l & 1) * 2;
        half v[8];
        device half4* target;
        if (part == 0) {
          const uint r = within / 4, quarter = within % 4;
          if (g * 32 + r >= uint(pitch)) { continue; }
          for (uint k = 0; k < 8; ++k) { v[k] = source[r * 32 + quarter * 8 + k]; }
          target = (device half4*)(query + (batchHead * pitch + g * 32 + r) * 32 + quarter * 8);
        } else if (part == 1) {
          const uint i = within / 32;
          if (g * 4 + i >= tiles) { continue; }
          for (uint c = 0; c < 4; ++c) {
            v[2 * c] = source[(i * 8 + fn) * 32 + c * 8 + fm];
            v[2 * c + 1] = source[(i * 8 + fn + 1) * 32 + c * 8 + fm];
          }
          target = (device half4*)(key + ((batchHead * tiles + g * 4 + i) * 32 + l) * 8);
        } else {
          const uint d = within / 32;
          for (uint i = 0; i < 4; ++i) {
            v[2 * i] = source[(i * 8 + fm) * 32 + d * 8 + fn];
            v[2 * i + 1] = source[(i * 8 + fm) * 32 + d * 8 + fn + 1];
          }
          target = (device half4*)(value + ((batchHead * groups + g) * 4 + d) * 256 + l * 8);
        }
        target[0] = half4(v[0], v[1], v[2], v[3]);
        target[1] = half4(v[4], v[5], v[6], v[7]);
      }
      """#, header: NeuralRenderingTransformerOperations.e4m3MetalHeaderText)

  /// An eight-query tile per threadgroup of eight simdgroups over the
  /// lane-major key and value layouts of `publishQKV`, so a lane fetches its
  /// fragment pairs for four tiles with two 8-byte loads. Each simdgroup scores
  /// whole 32-token groups (four independent accumulators) and converts them
  /// to vendor weights in registers (a lane holds an even/odd column pair, the
  /// transform's unit), stored lane-major; one lane per row keeps the
  /// sequential half denominator in token order; all threads publish E4M3
  /// probabilities; four simdgroups run the value chains over every 8-token
  /// tile in order. Half operands accumulate in float directly, bit-identical
  /// to converting them first.
  private static let residentV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_resident_v2", inputNames: ["query", "key", "value"], outputNames: ["output"],
    source: #"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint simd = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      constexpr uint tiles = uint(pitch) / 8;
      constexpr uint groups = (tiles + 3) / 4;
      const uint z = threadgroup_position_in_grid.z;
      const uint headBase = z * pitch * 32;
      const uint row0 = threadgroup_position_in_grid.y * 8;
      threadgroup half S[groups * 256];
      threadgroup half R[8];
      simdgroup_matrix<half, 8, 8> q[4];
      for (uint c = 0; c < 4; ++c) {
        simdgroup_load(q[c], query + headBase + row0 * 32 + c * 8, 32, ulong2(0), false);
      }
      for (uint g = simd; g < groups; g += 8) {
        half weights[8];
        for (uint i = 0; i < 4; ++i) {
          const uint t = g * 4 + i;
          simdgroup_matrix<float, 8, 8> acc;
          acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
          if (t < tiles) {
            const device half4* chunk = (const device half4*)(key + headBase + (t * 32 + lane) * 8);
            const half4 a = chunk[0], b = chunk[1];
            simdgroup_matrix<half, 8, 8> k;
            k.thread_elements()[0] = a.x; k.thread_elements()[1] = a.y;
            simdgroup_multiply_accumulate(acc, q[0], k, acc);
            k.thread_elements()[0] = a.z; k.thread_elements()[1] = a.w;
            simdgroup_multiply_accumulate(acc, q[1], k, acc);
            k.thread_elements()[0] = b.x; k.thread_elements()[1] = b.y;
            simdgroup_multiply_accumulate(acc, q[2], k, acc);
            k.thread_elements()[0] = b.z; k.thread_elements()[1] = b.w;
            simdgroup_multiply_accumulate(acc, q[3], k, acc);
          }
          const half2 w = vendor_weights(clamp(
            half2(half(acc.thread_elements()[0]), half(acc.thread_elements()[1])), half2(-3.0h), half2(3.0h)));
          weights[2 * i] = w.x; weights[2 * i + 1] = w.y;
        }
        threadgroup half4* stored = (threadgroup half4*)(S + (g * 32 + lane) * 8);
        stored[0] = half4(weights[0], weights[1], weights[2], weights[3]);
        stored[1] = half4(weights[4], weights[5], weights[6], weights[7]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup half4* S4 = (const threadgroup half4*)S;
      if (tid < 8) {
        // The lanes holding row tid's column pairs 0, 2, 4 and 6.
        uint L[4];
        for (uint p = 0; p < 4; ++p) { L[p] = ((tid >> 2) << 4) | ((p >> 1) << 3) | ((tid & 3) << 1) | (p & 1); }
        half total = 0.0h;
        constexpr uint fullGroups = uint(tokens) / 32;
        for (uint g = 0; g < fullGroups; ++g) {
          half4 c[4][2];
          for (uint p = 0; p < 4; ++p) { c[p][0] = S4[(g * 32 + L[p]) * 2]; c[p][1] = S4[(g * 32 + L[p]) * 2 + 1]; }
          for (uint p = 0; p < 4; ++p) { total += c[p][0].x; total += c[p][0].y; }
          for (uint p = 0; p < 4; ++p) { total += c[p][0].z; total += c[p][0].w; }
          for (uint p = 0; p < 4; ++p) { total += c[p][1].x; total += c[p][1].y; }
          for (uint p = 0; p < 4; ++p) { total += c[p][1].z; total += c[p][1].w; }
        }
        if (uint(tokens) % 32 != 0) {
          const uint g = fullGroups;
          half c[4][8];
          for (uint p = 0; p < 4; ++p) {
            const half4 a = S4[(g * 32 + L[p]) * 2], b = S4[(g * 32 + L[p]) * 2 + 1];
            c[p][0] = a.x; c[p][1] = a.y; c[p][2] = a.z; c[p][3] = a.w;
            c[p][4] = b.x; c[p][5] = b.y; c[p][6] = b.z; c[p][7] = b.w;
          }
          for (uint i = 0; i < 4; ++i) {
            for (uint p = 0; p < 4; ++p) {
              if (g * 32 + i * 8 + 2 * p < uint(tokens)) { total += c[p][2 * i]; total += c[p][2 * i + 1]; }
            }
          }
        }
        R[tid] = half(1.0f / float(total));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint chunk = tid; chunk < groups * 32; chunk += 256) {
        const uint g = chunk / 32, l = chunk % 32;
        const uint fm = ((l >> 4) << 2) + ((l >> 1) & 3);
        const uint fn = ((l >> 2) & 2) * 2 + (l & 1) * 2;
        const half r = R[fm];
        threadgroup half4* cell = (threadgroup half4*)(S + chunk * 8);
        for (uint h = 0; h < 2; ++h) {
          const half4 p = cell[h] * r;
          const uint j0 = g * 32 + (2 * h) * 8 + fn, j1 = j0 + 8;
          cell[h] = half4(
            j0 < uint(tokens) ? mlxdlss_e4m3_nonnegative(p.x) : 0.0h,
            j0 + 1 < uint(tokens) ? mlxdlss_e4m3_nonnegative(p.y) : 0.0h,
            j1 < uint(tokens) ? mlxdlss_e4m3_nonnegative(p.z) : 0.0h,
            j1 + 1 < uint(tokens) ? mlxdlss_e4m3_nonnegative(p.w) : 0.0h);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd >= 4) { return; }
      simdgroup_matrix<float, 8, 8> acc;
      acc.thread_elements()[0] = 0.0f; acc.thread_elements()[1] = 0.0f;
      const device half4* values = (const device half4*)(value + z * groups * 1024);
      for (uint g = 0; g < groups; ++g) {
        const half4 pa = S4[(g * 32 + lane) * 2], pb = S4[(g * 32 + lane) * 2 + 1];
        const half4 va = values[((g * 4 + simd) * 32 + lane) * 2], vb = values[((g * 4 + simd) * 32 + lane) * 2 + 1];
        simdgroup_matrix<half, 8, 8> p, v;
        p.thread_elements()[0] = pa.x; p.thread_elements()[1] = pa.y;
        v.thread_elements()[0] = va.x; v.thread_elements()[1] = va.y;
        simdgroup_multiply_accumulate(acc, p, v, acc);
        if (g * 4 + 1 < tiles) {
          p.thread_elements()[0] = pa.z; p.thread_elements()[1] = pa.w;
          v.thread_elements()[0] = va.z; v.thread_elements()[1] = va.w;
          simdgroup_multiply_accumulate(acc, p, v, acc);
        }
        if (g * 4 + 2 < tiles) {
          p.thread_elements()[0] = pb.x; p.thread_elements()[1] = pb.y;
          v.thread_elements()[0] = vb.x; v.thread_elements()[1] = vb.y;
          simdgroup_multiply_accumulate(acc, p, v, acc);
        }
        if (g * 4 + 3 < tiles) {
          p.thread_elements()[0] = pb.z; p.thread_elements()[1] = pb.w;
          v.thread_elements()[0] = vb.z; v.thread_elements()[1] = vb.w;
          simdgroup_multiply_accumulate(acc, p, v, acc);
        }
      }
      if (tokenMajor) {
        const uint fragmentRow = ((lane >> 4) << 2) + ((lane >> 1) & 3);
        const uint fragmentColumn = ((lane >> 2) & 2) * 2 + (lane & 1) * 2;
        const uint row = row0 + fragmentRow;
        if (row < uint(tokens)) {
          const uint batchRow = (z / uint(heads)) * uint(tokens) + row;
          *(device half2*)(output + batchRow * uint(heads) * 32 + (z % uint(heads)) * 32 + simd * 8 + fragmentColumn) = half2(
            mlxdlss_e4m3_no_nan(half(acc.thread_elements()[0])), mlxdlss_e4m3_no_nan(half(acc.thread_elements()[1])));
        }
      } else {
        simdgroup_matrix<half, 8, 8> result;
        result.thread_elements()[0] = mlxdlss_e4m3_no_nan(half(acc.thread_elements()[0]));
        result.thread_elements()[1] = mlxdlss_e4m3_no_nan(half(acc.thread_elements()[1]));
        simdgroup_store(result, output + headBase + row0 * 32 + simd * 8, 32, ulong2(0), false);
      }
      """#, header: header)

  /// Score tile followed by the parallel weight conversion of its valid columns.
  private static let weightTileV2 = #"""
    METAL_FUNC void weight_tile_v2(threadgroup half* S, uint token0, uint tokens, uint tid) {
      const uint valid = min(32u, tokens - token0);
      for (uint i = tid; i < 8 * 16; i += 128) {
        const uint row = i / 16;
        const uint j = (i % 16) * 2;
        if (j < valid) {
          half2 weight = vendor_weights(half2(S[row * 32 + j], S[row * 32 + j + 1]));
          S[row * 32 + j] = weight.x;
          S[row * 32 + j + 1] = weight.y;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    """#

  private static let denominatorV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_denominator_v2", inputNames: ["query", "key"], outputNames: ["reciprocal"],
    source: setup + #"""
      half total = 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        weight_tile_v2(S, token0, tokens, tid);
        if (tid < 8) {
          const uint valid = min(32u, tokens - token0);
          for (uint j = 0; j < valid; ++j) { total += S[tid * 32 + j]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (tid < 8 && row0 + tid < tokens) {
        reciprocal[head * tokens + row0 + tid] = half(1.0f / float(total));
      }
      """#, header: header + weightTileV2)

  private static let attentionV2 = MLXFast.metalKernel(
    name: "mlxdlss_global_attention_values_v2", inputNames: ["query", "key", "value", "reciprocal"], outputNames: ["output"],
    source: setup + #"""
      simdgroup_matrix<float, 8, 8> attended;
      attended.thread_elements()[0] = 0.0f;
      attended.thread_elements()[1] = 0.0f;
      const half r = row0 + (tid % 8) < tokens ? reciprocal[head * tokens + row0 + (tid % 8)] : 0.0h;
      for (uint token0 = 0; token0 < tokens; token0 += 32) {
        score_tile(Q, K, S, key, headBase, token0, tokens, tid, simd);
        weight_tile_v2(S, token0, tokens, tid);
        for (uint i = tid; i < 8 * 32; i += 128) {
          const uint row = i / 32;
          const uint j = i % 32;
          const half rr = row0 + row < tokens ? reciprocal[head * tokens + row0 + row] : 0.0h;
          S[i] = token0 + j < tokens ? half(float(mlxdlss_e4m3(S[i] * rr))) : 0.0h;
        }
        // K's storage is now free for the matching value rows.
        for (uint i = tid; i < 32 * 32; i += 128) {
          uint token = token0 + i / 32;
          K[i] = token < tokens ? value[headBase + token * 32 + i % 32] : 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < 32; c += 8) {
          simdgroup_matrix<half, 8, 8> p, v;
          simdgroup_load(p, S + c, 32, ulong2(0), false);
          simdgroup_load(v, K + c * 32 + simd * 8, 32, ulong2(0), false);
          simdgroup_matrix<float, 8, 8> left, right;
          left.thread_elements()[0] = float(p.thread_elements()[0]);
          left.thread_elements()[1] = float(p.thread_elements()[1]);
          right.thread_elements()[0] = float(v.thread_elements()[0]);
          right.thread_elements()[1] = float(v.thread_elements()[1]);
          simdgroup_multiply_accumulate(attended, left, right, attended);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      (void)r;
      simdgroup_matrix<half, 8, 8> result;
      result.thread_elements()[0] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[0]))));
      result.thread_elements()[1] = half(float(mlxdlss_e4m3(half(attended.thread_elements()[1]))));
      simdgroup_store(result, S + simd * 8, 32, ulong2(0), false);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < 8 * 32; i += 128) {
        uint row = row0 + i / 32;
        if (row < tokens) output[headBase + row * 32 + i % 32] = S[i];
      }
      """#, header: header + weightTileV2)
}
