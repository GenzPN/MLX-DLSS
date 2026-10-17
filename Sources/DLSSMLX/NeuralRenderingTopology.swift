import Foundation

/// Block layout of the recovered neural-rendering U-Net.
///
/// The vendor builds every network of this family from five stage depths, one
/// per encoder width (32, 64, 128 and 256 channels) plus the 512-channel split
/// stage, mirrored by the decoder, and a run of 1024-channel global blocks:
///
///     pre | encoder 32 | 64 | 128 | 256 | split 512 | global 1024 | bridge
///         | split 512 | decoder 256 | 128 | 64 | 32 | post
///
/// `nvngx_dlssnr.dll` 310.9.16 ships two configs of this factory: `ht`
/// (depths 4,4,6,8,8, 71 blocks, preset 1, the only network of 310.8.0) and
/// `small` (2,2,2,2,2, 31 blocks, preset 2, the shipping default). Both use
/// the same layer packing and kernels; `python/mlxdlss/topology.py` is the
/// reference of this layout.
struct NeuralRenderingTopology: Equatable, Sendable, CustomStringConvertible {
  static let encoderChannels = [32, 64, 128, 256]
  static let splitChannels = 512
  static let globalChannels = 1024
  static let defaultGlobalBlockCount = 8
  /// Window origin of each phase: unshifted, both, x only, y only.
  static let windowOrigins = [
    NeuralRenderingWindowOrigin.zero,
    NeuralRenderingWindowOrigin(y: -4, x: -4),
    NeuralRenderingWindowOrigin(y: 0, x: -4),
    NeuralRenderingWindowOrigin(y: -4, x: 0),
  ]
  static let postPhase = 1

  static let ht = NeuralRenderingTopology(depths: [4, 4, 6, 8, 8])
  static let small = NeuralRenderingTopology(depths: [2, 2, 2, 2, 2])

  /// Blocks per stage: the four encoder widths, then the split stage.
  let depths: [Int]
  let globalBlockCount: Int

  init(depths: [Int], globalBlockCount: Int = Self.defaultGlobalBlockCount) {
    precondition(depths.count == 5 && depths.allSatisfy { $0 >= 1 }, "five stage depths of at least 1")
    precondition(globalBlockCount >= 1, "at least one global block")
    self.depths = depths
    self.globalBlockCount = globalBlockCount
  }

  static func headCount(channels: Int) -> Int { channels / 32 }

  static func hiddenChannels(channels: Int) -> Int {
    switch channels {
    case 32: 128
    case 64: 224
    case 128: 384
    case 256: 704
    case splitChannels: 512
    default: 4096
    }
  }

  var name: String {
    switch self {
    case .ht: "ht"
    case .small: "small"
    default: "custom"
    }
  }

  /// `"4,4,6,8,8"`, with `/G` appended for a non-default global block count.
  var spec: String {
    let text = depths.map(String.init).joined(separator: ",")
    return globalBlockCount == Self.defaultGlobalBlockCount ? text : "\(text)/\(globalBlockCount)"
  }

  var description: String { "\(name) (\(spec), \(blockCount) blocks)" }

  var blockCount: Int { 3 + globalBlockCount + 2 * depths.reduce(0, +) }
  var post: Int { blockCount - 1 }

  /// Encoder stage `stage` (0-3), its downsample block last; stage 0 excludes the pre block.
  func encoderStage(_ stage: Int) -> ClosedRange<Int> {
    let start = 1 + depths[..<stage].reduce(0, +)
    return start...(start + depths[stage] - 1)
  }

  var splitEncoder: ClosedRange<Int> {
    let start = 1 + depths[..<4].reduce(0, +)
    return start...(start + depths[4] - 1)
  }

  var globalStage: ClosedRange<Int> {
    let start = splitEncoder.upperBound + 1
    return start...(start + globalBlockCount - 1)
  }

  var bridge: Int { globalStage.upperBound + 1 }

  var splitDecoder: ClosedRange<Int> { (bridge + 1)...(bridge + depths[4]) }

  /// Decoder stage mirroring encoder `stage`, its upsample block first.
  func decoderStage(_ stage: Int) -> ClosedRange<Int> {
    let start = splitDecoder.upperBound + 1 + depths[(stage + 1)..<4].reduce(0, +)
    return start...(start + depths[stage] - 1)
  }

  /// The 256-channel downsample that pads its output to whole windows for the split stage.
  var lastEncoderDownsample: Int { encoderStage(3).upperBound }

  /// Window-origin phase of a block, `nil` for the blocks without window attention.
  func phase(of blockIndex: Int) -> Int? {
    if blockIndex == 0 { return 0 }
    if blockIndex == post { return Self.postPhase }
    for stage in 0..<4 {
      let encoder = encoderStage(stage)
      if encoder.contains(blockIndex) { return (blockIndex - encoder.lowerBound) & 3 }
      let decoder = decoderStage(stage)
      if decoder.contains(blockIndex) { return (depths[stage] + blockIndex - decoder.lowerBound) & 3 }
    }
    if splitEncoder.contains(blockIndex) { return (blockIndex - splitEncoder.lowerBound) & 3 }
    if splitDecoder.contains(blockIndex) { return (depths[4] + blockIndex - splitDecoder.lowerBound) & 3 }
    return nil
  }

  func windowOrigin(for blockIndex: Int) -> NeuralRenderingWindowOrigin {
    phase(of: blockIndex).map { Self.windowOrigins[$0 % 4] } ?? .zero
  }

  var encoderStages: [NeuralRenderingStageSpec] {
    (0..<4).map { stage in
      let range = encoderStage(stage)
      return NeuralRenderingStageSpec(
        blockRange: (stage == 0 ? 0 : range.lowerBound)...range.upperBound,
        channels: Self.encoderChannels[stage])
    }
  }

  var decoderStages: [NeuralRenderingStageSpec] {
    (0..<4).reversed().map { stage in
      NeuralRenderingStageSpec(blockRange: decoderStage(stage), channels: Self.encoderChannels[stage])
    }
  }

  var blocks: [NeuralRenderingBlockSpec] {
    var result: [NeuralRenderingBlockSpec] = []
    func append(_ family: NeuralRenderingBlockFamily, _ channels: Int, output: Int? = nil, input: Int? = nil, attention: Bool = true) {
      result.append(
        NeuralRenderingBlockSpec(
          index: result.count, family: family, inputChannels: input ?? channels,
          outputChannels: output ?? channels,
          hiddenChannels: attention ? Self.hiddenChannels(channels: channels) : nil,
          headCount: attention ? Self.headCount(channels: channels) : nil))
    }
    append(.pre, 32, input: 16)
    for (stage, channels) in Self.encoderChannels.enumerated() {
      for i in 0..<depths[stage] {
        let last = i == depths[stage] - 1
        append(last ? .downsample : .window, channels, output: last ? channels * 2 : nil)
      }
    }
    for i in 0..<depths[4] {
      let last = i == depths[4] - 1
      append(last ? .splitDownsample : .splitWindow, Self.splitChannels, output: last ? Self.globalChannels : nil)
    }
    for _ in 0..<globalBlockCount { append(.globalAttention, Self.globalChannels) }
    append(.bridge, Self.globalChannels, output: Self.splitChannels, attention: false)
    for _ in 0..<depths[4] { append(.splitWindow, Self.splitChannels) }
    for stage in (0..<4).reversed() {
      let channels = Self.encoderChannels[stage]
      for j in 0..<depths[stage] {
        // The upsample block reads the wider low-resolution tensor and runs at the stage width.
        if j == 0 {
          result.append(
            NeuralRenderingBlockSpec(
              index: result.count, family: .upsample, inputChannels: channels * 2, outputChannels: channels,
              hiddenChannels: Self.hiddenChannels(channels: channels), headCount: Self.headCount(channels: channels)))
        } else {
          append(.window, channels)
        }
      }
    }
    append(.post, 32, output: 4)
    precondition(result.map(\.index) == Array(0..<blockCount))
    return result
  }

  // MARK: Inference from weights

  /// Recovers the stage depths from the tensor names and shapes of a logical weight map.
  static func infer(shapes: [String: [Int]]) throws -> NeuralRenderingTopology {
    let indices = shapes.keys.compactMap(Self.blockIndex(of:))
    guard let last = indices.max() else {
      throw MLXBackendError.unsupportedTopology("no blockN tensors")
    }
    let families = (0...last).map { classify(shapes: shapes, block: $0) }
    return try infer(families: families)
  }

  static func infer(weights: ValidatedWeights) throws -> NeuralRenderingTopology {
    var shapes: [String: [Int]] = [:]
    for name in weights.names {
      if let value = weights[name] { shapes[name] = value.shape }
    }
    return try infer(shapes: shapes)
  }

  private static func blockIndex(of name: String) -> Int? {
    guard name.hasPrefix("block"), let dot = name.firstIndex(of: ".") else { return nil }
    return Int(name[name.index(name.startIndex, offsetBy: 5)..<dot])
  }

  struct Family: Equatable {
    let family: NeuralRenderingBlockFamily
    let channels: Int
  }

  static func classify(shapes: [String: [Int]], block: Int) -> Family? {
    func layer0(_ name: String) -> [Int]? { shapes["block\(block).layer0.\(name)"] }
    if layer0("first_projection_weight") != nil {
      return Family(
        family: shapes["block\(block).layer4.weight"] != nil ? .splitDownsample : .splitWindow,
        channels: splitChannels)
    }
    if layer0("weight") == [globalChannels, hiddenChannels(channels: globalChannels)] {
      return Family(family: .globalAttention, channels: globalChannels)
    }
    if layer0("conv_weight") != nil { return Family(family: .bridge, channels: globalChannels) }
    guard let qkv = layer0("qkv_weight"), let channels = qkv.first else { return nil }
    if layer0("input_adapter_weight") != nil { return Family(family: .pre, channels: channels) }
    if layer0("inp_merge_sin") != nil { return Family(family: .post, channels: channels) }
    if layer0("sin") != nil { return Family(family: .upsample, channels: channels) }
    if layer0("weight0") == [channels, 2 * channels] { return Family(family: .downsample, channels: channels) }
    return Family(family: .window, channels: channels)
  }

  static func infer(families: [Family?]) throws -> NeuralRenderingTopology {
    var position = 0
    func take(_ predicate: (Family) -> Bool) -> Int {
      let start = position
      while position < families.count, let family = families[position], predicate(family) { position += 1 }
      return position - start
    }
    guard families.first == Family(family: .pre, channels: 32) else {
      throw MLXBackendError.unsupportedTopology("block 0 is not the 32-channel pre block")
    }
    position = 1
    var depths: [Int] = []
    for channels in encoderChannels {
      let regular = take { $0 == Family(family: .window, channels: channels) }
      guard take({ $0 == Family(family: .downsample, channels: channels) }) == 1 else {
        throw MLXBackendError.unsupportedTopology("encoder stage \(channels) has no single downsample block at \(position)")
      }
      depths.append(regular + 1)
    }
    let split = take { $0 == Family(family: .splitWindow, channels: splitChannels) }
    guard take({ $0 == Family(family: .splitDownsample, channels: splitChannels) }) == 1 else {
      throw MLXBackendError.unsupportedTopology("split encoder has no single transition block at \(position)")
    }
    depths.append(split + 1)
    let globalBlockCount = take { $0.family == .globalAttention }
    guard globalBlockCount >= 1, take({ $0.family == .bridge }) == 1 else {
      throw MLXBackendError.unsupportedTopology("no global stage and bridge block at \(position)")
    }
    let topology = NeuralRenderingTopology(depths: depths, globalBlockCount: globalBlockCount)
    let expected = topology.blocks.map { spec -> Family in
      Family(family: spec.family, channels: spec.family == .upsample ? spec.outputChannels : spec.inputChannels == 16 ? 32 : spec.inputChannels)
    }
    guard families.count == expected.count, zip(families, expected).allSatisfy({ $0 == $1 }) else {
      let common = min(families.count, expected.count)
      let mismatch = (0..<common).first { families[$0] != expected[$0] } ?? common
      throw MLXBackendError.unsupportedTopology(
        "weights do not follow the mirrored topology \(topology.spec) at block \(mismatch)")
    }
    return topology
  }
}
