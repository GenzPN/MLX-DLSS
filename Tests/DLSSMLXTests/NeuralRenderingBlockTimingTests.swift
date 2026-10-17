import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// Opt-in block-by-block timing of the fused float16 path on external weights:
/// `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_BLOCK_TIMING=HEIGHTxWIDTH`. The layout
/// follows the topology of the weights (71-block HT or 31-block small).
/// Every piece is evaluated on its own, so the sum exceeds a whole frame by the
/// per-launch synchronization. Diagnostics only.
final class NeuralRenderingBlockTimingTests: XCTestCase {
  func testExternalBlockTimingWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"], let shape = environment["MLXDLSS_BLOCK_TIMING"] else {
      throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_BLOCK_TIMING=HEIGHTxWIDTH")
    }
    let parts = shape.split(separator: "x").compactMap { Int($0) }
    let height = parts[0], width = parts[1]
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let topology = try NeuralRenderingTopology.infer(weights: weights)
    typealias Layout = NeuralRenderingTopology
    func window(_ index: Int, _ channels: Int) throws -> NeuralRenderingWindowBlock {
      try NeuralRenderingWindowBlock(
        weights: weights, blockIndex: index, channels: channels,
        hiddenChannels: Layout.hiddenChannels(channels: channels), headCount: Layout.headCount(channels: channels),
        compileBlock: true, topology: topology)
    }
    let input = MLXArray((0..<(height * width * 16)).map { sin(Float($0) * 0.001) }, [1, height, width, 16]).asType(.float16)
    eval(input)
    var total = 0.0
    var stageTotals: [(String, Double)] = []
    func best(_ label: String, stage: String, _ body: () -> [MLXArray]) -> [MLXArray] {
      var bestMs = Double.infinity
      var outputs: [MLXArray] = []
      for _ in 0..<3 {
        let clock = ContinuousClock(); let start = clock.now
        outputs = Device.withDefaultDevice(.gpu) { body() }
        eval(outputs)
        let d = start.duration(to: clock.now)
        bestMs = min(bestMs, Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15)
      }
      total += bestMs
      if let last = stageTotals.last, last.0 == stage {
        stageTotals[stageTotals.count - 1].1 += bestMs
      } else {
        stageTotals.append((stage, bestMs))
      }
      print("block-timing \(label): \(String(format: "%.2f", bestMs)) ms  shape \(outputs[0].shape)")
      return outputs
    }
    print("block-timing topology \(topology) at \(height)x\(width)")
    let stem = try NeuralRenderingEncoderStem(weights: weights, compileBlocks: true, topology: topology)
    let stemOutput = best("block 0 adapter + window + pool, blocks \(topology.encoderStage(0).dropLast().map(String.init).joined(separator: ","))", stage: "stem") {
      let output = stem(input)
      return [output.skip, output.fullResolutionSkip]
    }
    var value = stemOutput[0]
    let fullSkip = stemOutput[1]
    var skips: [MLXArray] = []
    for stage in 0..<4 {
      let channels = Layout.encoderChannels[stage]
      let blocks = topology.encoderStage(stage)
      if stage > 0 {
        for index in blocks.dropLast() {
          let block = try window(index, channels)
          let x = value
          value = best("block \(index) window \(channels)ch", stage: "encoder \(channels)") { [block(x)] }[0]
        }
      }
      skips.append(value)
      let downsample = try NeuralRenderingDownsampleBlock(
        weights: weights, blockIndex: blocks.upperBound, channels: channels,
        hiddenChannels: Layout.hiddenChannels(channels: channels), headCount: Layout.headCount(channels: channels),
        compileBlocks: true, topology: topology)
      let x = value
      value = best("block \(blocks.upperBound) downsample \(channels)ch", stage: "encoder \(channels)") { [downsample(x)] }[0]
    }
    let splitEncoder = try NeuralRenderingSplitEncoderStage(
      weights: weights, preciseAttention: true, fusedFeedForward: true, topology: topology)
    let encoderLatent = value
    let split = best("split encoder \(topology.splitEncoder)", stage: "split 512") {
      let output = splitEncoder(encoderLatent)
      return [output.latent, output.skip]
    }
    let globalStage = try NeuralRenderingGlobalStage(
      weights: weights, quantizeFFN: false, preciseAttention: true, fusedOperations: true, compileGraph: true,
      topology: topology)
    let latent = best("global \(topology.globalStage)", stage: "global 1024") { [globalStage(split[0])] }
    let decoderInput = try NeuralRenderingDecoderInput(weights: weights, topology: topology)
    value = best("bridge \(topology.bridge)", stage: "bridge") { [decoderInput(latent[0], skip: split[1])] }[0]
    for index in topology.splitDecoder {
      let block = try NeuralRenderingSplitWindowBlock(
        weights: weights, blockIndex: index, preciseAttention: true, fusedFeedForward: true, topology: topology)
      let x = value
      value = best("block \(index) split window", stage: "split 512") { [block(x)] }[0]
    }
    for stage in (0..<4).reversed() {
      let channels = Layout.encoderChannels[stage]
      let blocks = topology.decoderStage(stage)
      let upsample = try NeuralRenderingUpsampleBlock(
        weights: weights, blockIndex: blocks.lowerBound, channels: channels,
        hiddenChannels: Layout.hiddenChannels(channels: channels), headCount: Layout.headCount(channels: channels),
        compileBlocks: true, topology: topology)
      let x = value
      let skip = skips[stage]
      value = best("block \(blocks.lowerBound) upsample \(channels)ch", stage: "decoder \(channels)") { [upsample(x, skip: skip)] }[0]
      for index in blocks.dropFirst() {
        let block = try window(index, channels)
        let y = value
        value = best("block \(index) window \(channels)ch", stage: "decoder \(channels)") { [block(y)] }[0]
      }
    }
    let post = try NeuralRenderingPostBlock(weights: weights, compileBlocks: true, topology: topology)
    let decoded = value
    _ = best("post block \(topology.post) (upsample+merge+window+head)", stage: "post") { [post(decoded, skip: fullSkip)] }
    for (stage, ms) in stageTotals {
      print("block-timing stage \(stage): \(String(format: "%.2f", ms)) ms (\(String(format: "%.0f", 100 * ms / total))%)")
    }
    print("block-timing total of pieces: \(String(format: "%.1f", total)) ms at \(height)x\(width)")
  }
}
