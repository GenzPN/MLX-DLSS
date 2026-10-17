import Foundation
import MLX
import XCTest

@testable import DLSSMLX

final class NeuralRenderingTopologyTests: XCTestCase {
  /// Window phases of `python/mlxdlss/topology.py` (`None` → `nil`).
  private static let htPhases: [Int?] =
    [0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    + Array(repeating: nil, count: 9)
    + [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]
  private static let smallPhases: [Int?] =
    [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] + Array(repeating: nil, count: 9)
    + [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 1]

  func testHTLayoutMatchesTheRecoveredSeventyOneBlockGraph() {
    let topology = NeuralRenderingTopology.ht
    XCTAssertEqual(topology.blockCount, 71)
    XCTAssertEqual(topology.post, 70)
    XCTAssertEqual((0..<4).map(topology.encoderStage), [1...4, 5...8, 9...14, 15...22])
    XCTAssertEqual(topology.splitEncoder, 23...30)
    XCTAssertEqual(topology.globalStage, 31...38)
    XCTAssertEqual(topology.bridge, 39)
    XCTAssertEqual(topology.splitDecoder, 40...47)
    XCTAssertEqual((0..<4).reversed().map(topology.decoderStage), [48...55, 56...61, 62...65, 66...69])
    XCTAssertEqual(topology.lastEncoderDownsample, 22)
    XCTAssertEqual((0..<71).map(topology.phase(of:)), Self.htPhases)
  }

  func testSmallLayoutMirrorsTwoBlocksPerStage() {
    let topology = NeuralRenderingTopology.small
    XCTAssertEqual(topology.blockCount, 31)
    XCTAssertEqual(topology.post, 30)
    XCTAssertEqual((0..<4).map(topology.encoderStage), [1...2, 3...4, 5...6, 7...8])
    XCTAssertEqual(topology.splitEncoder, 9...10)
    XCTAssertEqual(topology.globalStage, 11...18)
    XCTAssertEqual(topology.bridge, 19)
    XCTAssertEqual(topology.splitDecoder, 20...21)
    XCTAssertEqual((0..<4).reversed().map(topology.decoderStage), [22...23, 24...25, 26...27, 28...29])
    XCTAssertEqual(topology.lastEncoderDownsample, 8)
    XCTAssertEqual((0..<31).map(topology.phase(of:)), Self.smallPhases)
    XCTAssertEqual(
      topology.blocks.map(\.family),
      [.pre, .window, .downsample, .window, .downsample, .window, .downsample, .window, .downsample,
       .splitWindow, .splitDownsample]
        + Array(repeating: .globalAttention, count: 8)
        + [.bridge, .splitWindow, .splitWindow, .upsample, .window, .upsample, .window, .upsample, .window,
           .upsample, .window, .post])
  }

  func testInferenceRecoversBothNetworksFromTensorShapes() throws {
    for topology in [NeuralRenderingTopology.ht, .small, NeuralRenderingTopology(depths: [1, 3, 2, 1, 1], globalBlockCount: 2)] {
      XCTAssertEqual(try NeuralRenderingTopology.infer(shapes: Self.shapes(for: topology)), topology)
    }
  }

  func testInferenceRejectsAMissingBridge() {
    var shapes = Self.shapes(for: .small)
    shapes = shapes.filter { !$0.key.hasPrefix("block19.") }
    XCTAssertThrowsError(try NeuralRenderingTopology.infer(shapes: shapes))
  }

  /// The tensor names and shapes `NeuralRenderingTopology.classify` reads, per block family.
  private static func shapes(for topology: NeuralRenderingTopology) -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    for block in topology.blocks {
      let prefix = "block\(block.index)"
      let channels = block.family == .upsample ? block.outputChannels : block.family == .pre ? 32 : block.inputChannels
      switch block.family {
      case .splitWindow, .splitDownsample:
        shapes["\(prefix).layer0.first_projection_weight"] = [512, 512]
        if block.family == .splitDownsample { shapes["\(prefix).layer4.weight"] = [512, 1024] }
      case .globalAttention:
        shapes["\(prefix).layer0.weight"] = [1024, 4096]
      case .bridge:
        shapes["\(prefix).layer0.conv_weight"] = [1024, 512]
      default:
        shapes["\(prefix).layer0.qkv_weight"] = [channels, channels * 3]
        switch block.family {
        case .pre: shapes["\(prefix).layer0.input_adapter_weight"] = [16, 32]
        case .post: shapes["\(prefix).layer0.inp_merge_sin"] = [32]
        case .upsample: shapes["\(prefix).layer0.sin"] = [channels]
        case .downsample: shapes["\(prefix).layer0.weight0"] = [channels, channels * 2]
        default: break
        }
      }
    }
    return shapes
  }

  /// Compares the MLX graph with the PyTorch reference (`python/mlxdlss/model.py`)
  /// on the same weights: `MLXDLSS_LOGICAL_WEIGHTS` and `MLXDLSS_NR_REFERENCE`, a
  /// safetensors file with the `input` and `output` of the reference.
  ///
  /// The E4M3 publications make the head chaotic: 1e-6 Gaussian input noise
  /// moves the reference output by 4.0% (71 blocks) and 12.6% (31 blocks)
  /// relative MAE on the 256x256 sine input, so the bound sits above that
  /// floor; `testExternalStagesMatchTheTorchReference` checks each stage tightly.
  func testExternalModelMatchesTheTorchReference() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"],
      let referencePath = environment["MLXDLSS_NR_REFERENCE"]
    else { throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_NR_REFERENCE") }
    let arrays = try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu)
    let reference = try loadArrays(url: URL(fileURLWithPath: referencePath), stream: .cpu)
    let input = try XCTUnwrap(reference["input"])
    let expected = try XCTUnwrap(reference["output"])
    let weights = ValidatedWeights(arrays: arrays)
    let eager = try NeuralRenderingTransformerModel(weights: weights)
    let fused = try NeuralRenderingTransformerModel(weights: weights.cast(to: .float16), compileBlocks: true)
    let scale = abs(expected).mean().item(Float.self)
    for (label, output, tolerance) in [
      ("float32 graph", Device.withDefaultDevice(.gpu) { eager(input) }, Float(0.25)),
      ("fused float16", Device.withDefaultDevice(.gpu) { fused(input.asType(.float16)) }.asType(.float32), Float(0.25)),
    ] {
      let error = abs(output - expected)
      let relative = error.mean().item(Float.self) / scale
      print("nr-reference \(eager.topology) \(label): mean |ref| \(scale), MAE \(error.mean().item(Float.self)), relative \(relative), max \(error.max().item(Float.self))")
      XCTAssertLessThan(relative, tolerance, label)
    }
  }

  /// Teacher-forced stage comparison with the PyTorch reference: every stage
  /// runs on the reference input of that stage (`MLXDLSS_NR_STAGE_REFERENCE`,
  /// written by a script that captures the intermediates of `model.forward`).
  func testExternalStagesMatchTheTorchReference() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let weightsPath = environment["MLXDLSS_LOGICAL_WEIGHTS"],
      let referencePath = environment["MLXDLSS_NR_STAGE_REFERENCE"]
    else { throw XCTSkip("set MLXDLSS_LOGICAL_WEIGHTS and MLXDLSS_NR_STAGE_REFERENCE") }
    let weights = ValidatedWeights(arrays: try loadArrays(url: URL(fileURLWithPath: weightsPath), stream: .cpu))
    let reference = try loadArrays(url: URL(fileURLWithPath: referencePath), stream: .cpu)
    let topology = try NeuralRenderingTopology.infer(weights: weights)
    func ref(_ name: String) throws -> MLXArray { try XCTUnwrap(reference[name], name) }
    func compare(_ label: String, _ actual: MLXArray, _ name: String) throws {
      let expected = try ref(name)
      XCTAssertEqual(actual.shape, expected.shape, label)
      let error = abs(actual - expected).mean().item(Float.self)
      let scale = abs(expected).mean().item(Float.self)
      print("nr-stage \(topology.name) \(label): relative MAE \(error / scale) (MAE \(error), mean |ref| \(scale))")
      // 2026-10-17, 256x256: at most 0.066 (71 blocks) and 0.020 (31 blocks).
      XCTAssertLessThan(error / scale, 0.1, label)
    }
    try Device.withDefaultDevice(.gpu) {
      let stem = try NeuralRenderingEncoderStem(weights: weights, topology: topology)(try ref("input"))
      try compare("stem full skip", stem.fullResolutionSkip, "full_skip")
      try compare("stem skip0", stem.skip, "skip0")
      let first = try NeuralRenderingDownsampleBlock(
        weights: weights, blockIndex: topology.encoderStage(0).upperBound, channels: 32, hiddenChannels: 128,
        headCount: 1, topology: topology)
      try compare("downsample \(topology.encoderStage(0).upperBound)", first(try ref("skip0")), "enc1_in")
      for stage in 1..<4 {
        let blocks = topology.encoderStage(stage)
        let channels = NeuralRenderingTopology.encoderChannels[stage]
        let encoder = try NeuralRenderingEncoderStage(
          weights: weights, regularBlocks: blocks.dropLast(), transitionBlock: blocks.upperBound, channels: channels,
          hiddenChannels: NeuralRenderingTopology.hiddenChannels(channels: channels),
          headCount: NeuralRenderingTopology.headCount(channels: channels), topology: topology)
        let output = encoder(try ref("enc\(stage)_in"))
        try compare("encoder \(stage) skip", output.skip, "skip\(stage)")
        try compare("encoder \(stage) downsample", output.downsampled, stage == 3 ? "enc_latent" : "enc\(stage + 1)_in")
      }
      let split = try NeuralRenderingSplitEncoderStage(weights: weights, topology: topology)(try ref("enc_latent"))
      try compare("split encoder skip", split.skip, "split_skip")
      try compare("split encoder latent", split.latent, "split_latent")
      let global = try NeuralRenderingGlobalStage(weights: weights, topology: topology)
      try compare("global stage", global(try ref("split_latent")), "global_out")
      let bridge = try NeuralRenderingDecoderInput(weights: weights, topology: topology)
      try compare("bridge", bridge(try ref("global_out"), skip: try ref("split_skip")), "bridge_out")
      let splitDecoder = try topology.splitDecoder.map {
        try NeuralRenderingSplitWindowBlock(weights: weights, blockIndex: $0, topology: topology)
      }
      try compare("split decoder", splitDecoder.reduce(try ref("bridge_out")) { $1($0) }, "split_decoder_out")
      var previous = "split_decoder_out"
      for stage in (0..<4).reversed() {
        let blocks = topology.decoderStage(stage)
        let channels = NeuralRenderingTopology.encoderChannels[stage]
        let decoder = try NeuralRenderingDecoderStage(
          weights: weights, upsampleBlock: blocks.lowerBound, regularBlocks: blocks.dropFirst(), channels: channels,
          hiddenChannels: NeuralRenderingTopology.hiddenChannels(channels: channels),
          headCount: NeuralRenderingTopology.headCount(channels: channels), topology: topology)
        try compare("decoder \(stage)", decoder(try ref(previous), skip: try ref("skip\(stage)")), "dec\(stage)_out")
        previous = "dec\(stage)_out"
      }
      let post = try NeuralRenderingPostBlock(weights: weights, topology: topology)
      try compare("post", post(try ref("dec0_out"), skip: try ref("full_skip")), "output")
    }
  }
}
