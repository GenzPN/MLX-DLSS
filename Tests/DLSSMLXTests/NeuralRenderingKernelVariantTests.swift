import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in A/B of a fused kernel against an earlier version of its source:
/// `MLXDLSS_BASELINE_1H_SOURCE` names a file with the previous
/// `NeuralRenderingFusedWindowBlock.source` (`MLXDLSS_BASELINE_1H_ROWS`: its window
/// rows per simdgroup, 2 by default). The current kernel must reproduce the
/// baseline bit for bit; both are timed in the same process. Diagnostics only.
final class NeuralRenderingKernelVariantTests: XCTestCase {
  private func ms(_ body: () -> MLXArray, warm: Int = 3, runs: Int = 9) -> Double {
    for _ in 0..<warm { eval(Device.withDefaultDevice(.gpu) { body() }) }
    var best = Double.infinity
    for _ in 0..<runs {
      let start = ContinuousClock.now
      eval(Device.withDefaultDevice(.gpu) { body() })
      let d = start.duration(to: .now)
      best = min(best, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
    }
    return best
  }

  func testFusedWindowBlockMatchesBaselineSource() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_BASELINE_1H_SOURCE"] else {
      throw XCTSkip("set MLXDLSS_BASELINE_1H_SOURCE")
    }
    let baseline = NeuralRenderingFusedWindowBlock.makeKernel(
      name: "mlxdlss_baseline_window_block_1h32", source: try String(contentsOfFile: path, encoding: .utf8))
    let baselineRows = ProcessInfo.processInfo.environment["MLXDLSS_BASELINE_1H_ROWS"].flatMap(Int.init) ?? 2
    MLXRandom.seed(7)
    let w1 = (MLXRandom.normal([32, 128]) * 0.2).asType(.float16)
    let w2 = (MLXRandom.normal([128, 32]) * 0.2).asType(.float16)
    let cos1 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    let qkv = (MLXRandom.normal([32, 96]) * 0.3).asType(.float16)
    let scale = MLXArray([Float(7.5)]).asType(.float16)
    let bias = (MLXRandom.normal([1, 64, 64]) * 2.0).asType(.float16)
    let proj = (MLXRandom.normal([32, 32]) * 0.2).asType(.float16)
    let cos2 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    eval(w1, w2, cos1, qkv, scale, bias, proj, cos2)
    for (height, width) in [(136, 248), (544, 960), (1088, 1920)] {
      let x = (MLXRandom.normal([1, height, width, 32]) * 2).asType(.float16)
      eval(x)
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4)] {
        for publish in [true, false] {
          func run(_ kernel: MLXFast.MLXFastKernel?) -> MLXArray {
            NeuralRenderingFusedWindowBlock.apply(
              x, expansionWeight: w1, feedForwardProjectionWeight: w2, feedForwardCosine: cos1, qkvWeight: qkv,
              attentionScale: scale, attentionBias: bias, attentionProjectionWeight: proj, attentionCosine: cos2,
              windowOrigin: origin, publish: publish, kernel: kernel, rowsPerSimdgroup: kernel == nil ? nil : baselineRows)
          }
          let reference = Device.withDefaultDevice(.gpu) { run(baseline) }
          let current = Device.withDefaultDevice(.gpu) { run(nil) }
          let differing = (reference.view(dtype: .uint16) .!= current.view(dtype: .uint16)).sum().item(Int32.self)
          XCTAssertEqual(differing, 0, "\(height)x\(width) origin \(origin) publish \(publish)")
          if origin == .zero, publish {
            // Alternating rounds keep clock ramps and background load out of the comparison.
            var tBaseline = Double.infinity
            var tCurrent = Double.infinity
            for _ in 0..<3 {
              tBaseline = min(tBaseline, ms { run(baseline) })
              tCurrent = min(tCurrent, ms { run(nil) })
            }
            print("kernel-variant 1h block \(height)x\(width): baseline \(String(format: "%.3f", tBaseline)) ms, current \(String(format: "%.3f", tCurrent)) ms (\(String(format: "%+.1f", 100 * (tCurrent / tBaseline - 1)))%), \(differing) differing values")
          }
        }
      }
    }
  }

  /// `MLXDLSS_BASELINE_1H_VARIANTS_DIR`: earlier sources of the stem and post
  /// variants (`adapter_pool.metal`, `merge_head.metal`). The current variants
  /// must reproduce them bit for bit at 1088x1920 (all four window phases for
  /// the post kernel); both are timed in alternating rounds.
  func testStemAndPostVariantsMatchBaselineSources() throws {
    guard let directory = ProcessInfo.processInfo.environment["MLXDLSS_BASELINE_1H_VARIANTS_DIR"] else {
      throw XCTSkip("set MLXDLSS_BASELINE_1H_VARIANTS_DIR")
    }
    typealias Block = NeuralRenderingFusedWindowBlock
    MLXRandom.seed(5)
    let weights = Block.Weights(
      expansion: (MLXRandom.normal([32, 128]) * 0.2).asType(.float16),
      feedForwardProjection: (MLXRandom.normal([128, 32]) * 0.2).asType(.float16),
      feedForwardCosine: MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16),
      qkv: (MLXRandom.normal([32, 96]) * 0.3).asType(.float16),
      attentionScale: MLXArray([Float(7.5)]).asType(.float16),
      attentionBias: (MLXRandom.normal([1, 64, 64]) * 2.0).asType(.float16),
      attentionProjection: (MLXRandom.normal([32, 32]) * 0.2).asType(.float16),
      attentionCosine: MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16))
    let (height, width) = (1088, 1920)
    let network = (MLXRandom.normal([1, height, width, 16]) * 1.5).asType(.float16)
    let adapter = (MLXRandom.normal([16, 32]) * 0.3).asType(.float16)
    let low = NeuralRenderingTransformerOperations.e4m3RoundTrip((MLXRandom.normal([1, height / 2, width / 2, 32]) * 1.5).asType(.float16))
    let skip = NeuralRenderingTransformerOperations.e4m3RoundTrip((MLXRandom.normal([1, height, width, 32]) * 1.5).asType(.float16))
    let lowScale = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    let skipScale = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    let headWeight = (MLXRandom.normal([32, 8]) * 0.3).asType(.float16)
    eval(network, adapter, low, skip, lowScale, skipScale, headWeight)
    let stemBaseline = Block.makeVariantKernel(
      .adapter, .pool, name: "mlxdlss_baseline_1h_adapter_pool",
      source: try String(contentsOfFile: directory + "/adapter_pool.metal", encoding: .utf8))
    let postBaseline = Block.makeVariantKernel(
      .merge, .head, name: "mlxdlss_baseline_1h_merge_head",
      source: try String(contentsOfFile: directory + "/merge_head.metal", encoding: .utf8))
    func stem(_ kernel: MLXFast.MLXFastKernel?) -> [MLXArray] {
      Device.withDefaultDevice(.gpu) {
        Block.applyVariant(
          .adapter(input: network, weight: adapter), weights: weights, windowOrigin: .zero, epilogue: .pool,
          publish: true, kernel: kernel)
      }
    }
    func post(_ kernel: MLXFast.MLXFastKernel?, _ origin: NeuralRenderingWindowOrigin) -> [MLXArray] {
      Device.withDefaultDevice(.gpu) {
        Block.applyVariant(
          .merge(low: low, skip: skip, lowScale: lowScale, skipScale: skipScale, publish: false), weights: weights,
          windowOrigin: origin, epilogue: .head, publish: false, headWeight: headWeight, kernel: kernel)
      }
    }
    func differing(_ a: [MLXArray], _ b: [MLXArray]) -> Int {
      zip(a, b).map { (($0.view(dtype: .uint16) .!= $1.view(dtype: .uint16)).sum().item(Int32.self)) }.reduce(0) { $0 + Int($1) }
    }
    XCTAssertEqual(differing(stem(stemBaseline), stem(nil)), 0, "stem")
    for origin in [NeuralRenderingWindowOrigin.zero, .init(y: -4, x: -4), .init(y: 0, x: -4), .init(y: -4, x: 0)] {
      XCTAssertEqual(differing(post(postBaseline, origin), post(nil, origin)), 0, "post \(origin)")
    }
    let rounds = ProcessInfo.processInfo.environment["MLXDLSS_1H_VARIANT_ROUNDS"].flatMap(Int.init) ?? 40
    var samples = [[Double]](repeating: [], count: 4)
    let bodies: [() -> MLXArray] = [
      { stem(stemBaseline)[0] }, { stem(nil)[0] }, { post(postBaseline, .zero)[0] }, { post(nil, .zero)[0] },
    ]
    for body in bodies { _ = ms(body, warm: 3, runs: 1) }
    for _ in 0..<rounds {
      for (index, body) in bodies.enumerated() { samples[index].append(ms(body, warm: 0, runs: 1)) }
    }
    let best = samples.map { $0.min()! }
    let median = samples.map { $0.sorted()[$0.count / 2] }
    for (index, label) in ["stem", "post"].enumerated() {
      let (b, c) = (2 * index, 2 * index + 1)
      print("kernel-variant 1h \(label) \(height)x\(width): baseline min \(String(format: "%.3f", best[b])) median \(String(format: "%.3f", median[b])) ms, current min \(String(format: "%.3f", best[c])) (\(String(format: "%+.1f", 100 * (best[c] / best[b] - 1)))%) median \(String(format: "%.3f", median[c])) ms (\(String(format: "%+.1f", 100 * (median[c] / median[b] - 1)))%)")
    }
  }

  /// Times every `*.metal` variant of the 1h kernel source in
  /// `MLXDLSS_1H_VARIANTS_DIR` in alternating rounds (`MLXDLSS_1H_VARIANT_ROUNDS`,
  /// 40 by default) and counts the output values that differ from the first file
  /// (ablations differ on purpose; nothing is asserted).
  func testFusedWindowBlockVariantTiming() throws {
    guard let directory = ProcessInfo.processInfo.environment["MLXDLSS_1H_VARIANTS_DIR"] else {
      throw XCTSkip("set MLXDLSS_1H_VARIANTS_DIR")
    }
    let rounds = ProcessInfo.processInfo.environment["MLXDLSS_1H_VARIANT_ROUNDS"].flatMap(Int.init) ?? 40
    let files = try FileManager.default.contentsOfDirectory(atPath: directory).filter { $0.hasSuffix(".metal") }.sorted()
    let x = (MLXRandom.normal([1, 1088, 1920, 32]) * 2).asType(.float16)
    let w1 = (MLXRandom.normal([32, 128]) * 0.2).asType(.float16)
    let w2 = (MLXRandom.normal([128, 32]) * 0.2).asType(.float16)
    let cos1 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    let qkv = (MLXRandom.normal([32, 96]) * 0.3).asType(.float16)
    let scale = MLXArray([Float(7.5)]).asType(.float16)
    let bias = (MLXRandom.normal([1, 64, 64]) * 2.0).asType(.float16)
    let proj = (MLXRandom.normal([32, 32]) * 0.2).asType(.float16)
    let cos2 = MLXRandom.uniform(low: 0.5, high: 1.0, [32]).asType(.float16)
    eval(x, w1, w2, cos1, qkv, scale, bias, proj, cos2)
    let kernels = try files.enumerated().map { index, file in
      NeuralRenderingFusedWindowBlock.makeKernel(
        name: "mlxdlss_variant_\(index)", source: try String(contentsOfFile: directory + "/" + file, encoding: .utf8))
    }
    // A file name containing `rows2` runs with two window rows per simdgroup.
    // Each sample chains `MLXDLSS_1H_VARIANT_CHAIN` launches (4 by default), so
    // background GPU load averages over a longer stretch.
    let chain = ProcessInfo.processInfo.environment["MLXDLSS_1H_VARIANT_CHAIN"].flatMap(Int.init) ?? 4
    func run(_ index: Int) -> MLXArray {
      var value = x
      for _ in 0..<chain {
        value = NeuralRenderingFusedWindowBlock.apply(
          value, expansionWeight: w1, feedForwardProjectionWeight: w2, feedForwardCosine: cos1, qkvWeight: qkv,
          attentionScale: scale, attentionBias: bias, attentionProjectionWeight: proj, attentionCosine: cos2,
          windowOrigin: .zero, publish: true, kernel: kernels[index],
          rowsPerSimdgroup: files[index].contains("rows2") ? 2 : 1)
      }
      return value
    }
    guard !kernels.isEmpty else { return }
    let reference = Device.withDefaultDevice(.gpu) { run(0) }
    eval(reference)
    // One launch per variant per round, so every variant samples the same
    // stretch of background GPU load; minimum and median over the rounds.
    for index in kernels.indices { _ = ms({ run(index) }, warm: 3, runs: 1) }
    var samples = [[Double]](repeating: [], count: kernels.count)
    for _ in 0..<rounds {
      for index in kernels.indices {
        samples[index].append(ms({ run(index) }, warm: 0, runs: 1))
      }
    }
    let best = samples.map { $0.min()! }
    let median = samples.map { $0.sorted()[$0.count / 2] }
    for (index, file) in files.enumerated() {
      let output = Device.withDefaultDevice(.gpu) { run(index) }
      let differing = (reference.view(dtype: .uint16) .!= output.view(dtype: .uint16)).sum().item(Int32.self)
      print("kernel-variant \(file): min \(String(format: "%.3f", best[index] / Double(chain))) ms (\(String(format: "%+.1f", 100 * (best[index] / best[0] - 1)))%), median \(String(format: "%.3f", median[index] / Double(chain))) ms (\(String(format: "%+.1f", 100 * (median[index] / median[0] - 1)))%), \(differing) values differ from \(files[0])")
    }
  }

  /// `MLXDLSS_BASELINE_ATTENTION_SOURCE`: an earlier
  /// `NeuralRenderingFusedWindowAttention.source` (`MLXDLSS_BASELINE_ATTENTION_SIMDGROUPS`
  /// simdgroups per window, 4 by default).
  func testFusedWindowAttentionMatchesBaselineSource() throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_BASELINE_ATTENTION_SOURCE"] else {
      throw XCTSkip("set MLXDLSS_BASELINE_ATTENTION_SOURCE")
    }
    let baseline = NeuralRenderingFusedWindowAttention.makeKernel(
      name: "mlxdlss_baseline_window_attention_core", source: try String(contentsOfFile: path, encoding: .utf8))
    let baselineSimdgroups = ProcessInfo.processInfo.environment["MLXDLSS_BASELINE_ATTENTION_SIMDGROUPS"].flatMap(Int.init) ?? 4
    MLXRandom.seed(11)
    for (heads, height, width) in [(2, 272, 480), (4, 136, 240), (8, 68, 120), (16, 36, 60), (2, 37, 53)] {
      let channels = heads * 32
      let qkv = (MLXRandom.normal([1, height, width, channels * 3]) * 2).asType(.float16)
      let scale = MLXRandom.uniform(low: 2, high: 12, [heads]).asType(.float16)
      let bias = (MLXRandom.normal([heads, 64, 64]) * 2.0).asType(.float16)
      eval(qkv, scale, bias)
      for origin in [NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4), NeuralRenderingWindowOrigin(y: 0, x: -4)] {
        func run(_ kernel: MLXFast.MLXFastKernel?) -> MLXArray {
          NeuralRenderingFusedWindowAttention.apply(
            qkv: qkv, attentionScale: scale, attentionBias: bias, headCount: heads, windowOrigin: origin,
            kernel: kernel, simdgroupsPerWindow: kernel == nil ? NeuralRenderingFusedWindowAttention.simdgroupsPerWindow : baselineSimdgroups)
        }
        let reference = Device.withDefaultDevice(.gpu) { run(baseline) }
        let current = Device.withDefaultDevice(.gpu) { run(nil) }
        let differing = (reference.view(dtype: .uint16) .!= current.view(dtype: .uint16)).sum().item(Int32.self)
        XCTAssertEqual(differing, 0, "\(heads) heads \(height)x\(width) origin \(origin)")
        if origin == .zero {
          var tBaseline = Double.infinity
          var tCurrent = Double.infinity
          for _ in 0..<3 {
            tBaseline = min(tBaseline, ms { run(baseline) })
            tCurrent = min(tCurrent, ms { run(nil) })
          }
          print("kernel-variant attention \(heads)h \(height)x\(width): baseline \(String(format: "%.3f", tBaseline)) ms, current \(String(format: "%.3f", tCurrent)) ms (\(String(format: "%+.1f", 100 * (tCurrent / tBaseline - 1)))%), \(differing) differing values")
        }
      }
    }
  }
}
