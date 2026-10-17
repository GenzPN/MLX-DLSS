import Foundation
import MLX
import DLSSCore
import XCTest

@testable import DLSSMLX

final class MLXNeuralRenderingTemporalBackendTests: XCTestCase {
  private let deviceFeatureDisplayTolerance: Float = 0.000_2

  func testConfidenceFeaturesMatchAnalyticalTranslationForFusedAndPaddedPaths() throws {
    let width = 5
    let height = 3
    let count = width * height
    let currentValues = (0..<count * 3).map { Float(($0 * 7) % 31) / 32 }
    let historyValues = (0..<count * 3).map { Float(($0 * 11) % 29) / 32 }
    let current = try smallTensor(
      name: "color", channels: 3, width: width, height: height, values: currentValues)
    let history = try smallTensor(
      name: "history", channels: 3, width: width, height: height, values: historyValues)
    let motion = MLXArray(
      Array(repeating: [Float(-1) / Float(width), 0], count: count).flatMap { $0 },
      [1, height, width, 2])
    let depth = MLXArray.ones([1, height, width, 1])
    let processor = MLXTemporalFeatureProcessor()
    let geometry = try NeuralRenderingNetworkGeometryPolicy.vendorAligned.resolve(
      outputWidth: width, outputHeight: height)
    let paddedBase = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: current, noiseFrameIndex: 3, geometry: geometry)
    let logicalBase = array(paddedBase)[0..., 0..<height, 0..<width, 0...]
    func scaled(_ value: Float) -> Float {
      Float(Float16(Float(Float16(Float(Float16(value)) - 0.5)) * 0.125))
    }
    for mask in [
      Array(repeating: Float(0), count: count), Array(repeating: Float(1), count: count),
      Array(repeating: Float(0.375), count: count),
      (0..<count).map { [Float(0), 1, 0.375][$0 % 3] },
    ] {
      let confidence = MLXArray(mask, [1, height, width, 1])
      let fused = processor(
        color: array(current), noiseFrameIndex: 3, history: array(history), motion: motion,
        depth: depth, depthInverted: false, historyConfidence: confidence)
      let padded = processor(
        baseFeatures: logicalBase, history: array(history), motion: motion, depth: depth,
        depthInverted: false, historyConfidence: confidence)
      eval(fused, padded)
      for actual in [fused.asArray(Float.self), padded.asArray(Float.self)] {
        for pixel in 0..<count {
          let previousPixel = pixel / width * width + max(0, pixel % width - 1)
          for channel in 0..<3 {
            let currentFeature = scaled(currentValues[pixel * 3 + channel])
            let previousFeature = scaled(historyValues[previousPixel * 3 + channel])
            let c = mask[pixel]
            let expected =
              c == 0
              ? currentFeature
              : c == 1 ? previousFeature : currentFeature + c * (previousFeature - currentFeature)
            if c == 0 || c == 1 {
              XCTAssertEqual(actual[pixel * 16 + 7 + channel], expected)
            } else {
              XCTAssertEqual(actual[pixel * 16 + 7 + channel], expected, accuracy: 0.000_001)
            }
          }
        }
      }
    }
  }

  func testConfidenceFullMaskMatchesOmittedMaskExactly() throws {
    let currentRequest = try request(frameIndex: 1, phase: 31)
    let current = array(currentRequest.input(named: "color")!)
    let history = array(try request(frameIndex: 0, phase: 0).input(named: "color")!)
    let motion = array(currentRequest.input(named: "motion")!)
    let depth = array(currentRequest.input(named: "depth")!)
    let confidence = MLXArray.ones([1, 128, 128, 1])
    let processor = MLXTemporalFeatureProcessor()
    let omitted = processor(
      color: current, noiseFrameIndex: 1, history: history, motion: motion, depth: depth,
      depthInverted: false)
    let full = processor(
      color: current, noiseFrameIndex: 1, history: history, motion: motion, depth: depth,
      depthInverted: false, historyConfidence: confidence)
    eval(omitted, full)
    XCTAssertEqual(full.asArray(Float.self), omitted.asArray(Float.self))
    let head = MLXArray.zeros([1, 128, 128, 4])
    let postprocessor = MLXTemporalPostprocessor()
    let omittedOutput = postprocessor(
      head: head, currentColor: current, features: omitted, hasHistory: true)
    let fullOutput = postprocessor(
      head: head, currentColor: current, features: full, hasHistory: true,
      historyConfidence: confidence)
    eval(omittedOutput, fullOutput)
    XCTAssertEqual(fullOutput.asArray(Float.self), omittedOutput.asArray(Float.self))
  }

  func testDeviceBackendRejectsConfidenceShapeAndValuesBeforeRendering() async throws {
    guard let packagePath = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run confidence request validation")
    }
    let device = try MLXNeuralRenderingDeviceTemporalBackend(
      packageURL: URL(fileURLWithPath: packagePath))
    let frame = try request(frameIndex: 0, phase: 0)
    let inputs = [
      frame.input(named: "color")!, frame.input(named: "motion")!, frame.input(named: "depth")!,
    ]
    let malformed = [
      try smallTensor(name: "historyConfidence", channels: 1, width: 1, height: 1, values: [1]),
      try tensor(
        name: "historyConfidence", channels: 2, values: Array(repeating: 1, count: 128 * 128 * 2)),
      try tensor(
        name: "historyConfidence", channels: 1, values: Array(repeating: .nan, count: 128 * 128)),
    ]
    for confidence in malformed {
      do {
        _ = try await device.render(
          NeuralRenderRequest(
            sequenceID: 0, temporalContext: frame.temporalContext, inputs: inputs + [confidence]))
        XCTFail("expected malformed confidence rejection")
      } catch {
        XCTAssertTrue(
          error is NeuralRenderingTemporalPreprocessorError || error is MLXTemporalConfidenceError)
      }
    }
  }

  func testConfidenceValueValidationRejectsMalformedMaps() throws {
    for value: Float in [.nan, .infinity, -.infinity, -0.001, 1.001] {
      let tensor = try smallTensor(
        name: "historyConfidence", channels: 1, width: 1, height: 1, values: [value])
      XCTAssertThrowsError(try validateHistoryConfidenceValues(tensor)) { error in
        XCTAssertEqual(error as? MLXTemporalConfidenceError, .invalidValues)
      }
    }
    let valid = try smallTensor(
      name: "historyConfidence", channels: 1, width: 3, height: 1, values: [0, 0.375, 1])
    XCTAssertNoThrow(try validateHistoryConfidenceValues(valid))
  }

  func testConfidencePostprocessorHasIndependentBlendScaleAndMaskExpectation() throws {
    let current = MLXArray([Float(0.25), 0.5, 0.75], [1, 1, 1, 3])
    let head = MLXArray([Float(0.5), -0.5, 0.25, 0], [1, 1, 1, 4])
    var features = [Float](repeating: 0, count: 16)
    features[7] = 0.0625
    features[8] = -0.0625
    features[9] = 0
    for scale: Float in [0.5, 0.739_746_093_75] {
      for c: Float in [0, 0.375, 1] {
        let output = MLXTemporalPostprocessor(blendScale: scale)(
          head: head, currentColor: current, features: MLXArray(features, [1, 1, 1, 16]),
          hasHistory: true, historyConfidence: MLXArray([c], [1, 1, 1, 1]))
        eval(output)
        let predictions: [Float] = [0.375, 0.375, 0.8125]
        let history: [Float] = [1, 0, 0.5]
        let alpha = 0.5 * Float(Float16(scale)) * c
        let expected = zip(predictions, history).map { $0 + alpha * ($1 - $0) }
        XCTAssertEqual(output.asArray(Float.self), expected)
      }
    }
  }

  func testLiveIntensityValuesCanShareOnePendingGraph() {
    let values: [Float] = [0, 0.2, 0.8, 1, 2, 0.4]
    let current: [Float] = [0.25, 0.5, 0.75]
    let predictions: [Float] = [0.375, 0.375, 0.8125]
    let input = MLXArray(current, [1, 1, 1, 3])
    let head = MLXArray([Float(0.5), -0.5, 0.25, 0], [1, 1, 1, 4])
    let features = MLXArray.zeros([1, 1, 1, 16])
    let processor = MLXTemporalPostprocessor()
    let outputs = values.map {
      processor(head: head, currentColor: input, features: features, hasHistory: false, intensity: $0)
    }
    eval(outputs)
    for (value, output) in zip(values, outputs) {
      let expected = zip(current, predictions).map { $0 + min(1, value) * ($1 - $0) }
      XCTAssertEqual(output.asArray(Float.self), expected)
    }
  }

  func testDeviceBaseFeatureKernelMatchesPortableFeatureContract() throws {
    let color = try smallTensor(
      name: "color",
      channels: 3,
      width: 2,
      height: 2,
      values: [
        0, 0.25, 0.5,
        0.75, 1, 0.125,
        0.375, 0.625, 0.875,
        1, 0.5, 0,
      ]
    )
    let controlMask = try smallTensor(
      name: "controlMask",
      channels: 3,
      width: 2,
      height: 2,
      values: [
        1, 0.25, 0.75,
        0.5, 1, 0,
        0, 0.5, 0.25,
        0.75, 0, 1,
      ]
    )
    let expected = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      noiseFrameIndex: 7,
      controlMask: controlMask
    )

    let output = MLXFirstFrameFeatureProcessor()(
      color: array(color),
      controlMask: array(controlMask),
      noiseFrameIndex: 7
    )
    eval(output)

    let actual = output.asArray(Float.self)
    let reference = values(expected)
    for pixel in 0..<4 {
      for channel in 0..<16 {
        let offset = pixel * 16 + channel
        XCTAssertEqual(
          actual[offset],
          reference[offset],
          "feature pixel \(pixel) channel \(channel)"
        )
      }
    }
  }

  func testDeviceBaseFeatureKernelMatchesPortableFeatureControls() throws {
    let color = try smallTensor(
      name: "color",
      channels: 3,
      width: 1,
      height: 1,
      values: [0.25, 0.5, 0.75]
    )
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )
    let expected = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: color,
      noiseFrameIndex: 7,
      normalizedStyle: controls.normalizedStyle,
      localToneStrength: controls.localToneStrength,
      localStructureStrength: controls.localStructureStrength,
      automaticMask: controls.automaticMask
    )

    let output = MLXFirstFrameFeatureProcessor()(
      color: array(color),
      noiseFrameIndex: 7,
      featureControls: controls
    )
    eval(output)

    XCTAssertEqual(
      Array(output.asArray(Float.self)[10...15]),
      Array(values(expected)[10...15])
    )
  }

  /// The padded extent's features in one kernel equal the logical features
  /// gathered onto the extent with the noise channels of the network pixels,
  /// and the postprocessor reads the logical corner of the padded head output
  /// and features exactly like their crops.
  func testExtendedFeatureKernelsMatchTheGatheredLogicalFeatures() throws {
    let (width, height) = (330, 341)
    let geometry = try NeuralRenderingNetworkGeometryPolicy.vendorAligned.resolve(outputWidth: width, outputHeight: height)
    XCTAssertFalse(geometry.isIdentity)
    let indices = NeuralRenderingExtensionIndices(geometry)
    MLXRandom.seed(17)
    let color = MLXRandom.uniform(low: -0.1, high: 1.1, [1, height, width, 3])
    let mask = MLXRandom.uniform(low: 0, high: 1, [1, height, width, 3])
    let history = MLXRandom.uniform(low: 0, high: 1, [1, height, width, 3])
    let motion = MLXRandom.uniform(low: -0.02, high: 0.02, [1, height, width, 2])
    let depth = MLXRandom.uniform(low: 0, high: 1, [1, height, width, 1])
    let confidence = which(
      MLXRandom.uniform(low: 0, high: 1, [1, height, width, 1]) .< 0.3, MLXArray(Float(0)),
      minimum(MLXRandom.uniform(low: 0, high: 1.6, [1, height, width, 1]), MLXArray(Float(1))))
    let base = MLXFirstFrameFeatureProcessor()
    let temporal = MLXTemporalFeatureProcessor()
    func gathered(_ array: MLXArray) -> MLXArray {
      array.take(indices.rows, axis: 1).take(indices.columns, axis: 2)
    }
    func bits(_ array: MLXArray) -> [UInt32] { array.asType(.float32).view(dtype: .uint32).asArray(UInt32.self) }
    let transform = try NeuralRenderingTextureTransform(
      baseX: 3, baseY: 2, extentWidth: width - 7, extentHeight: height - 5, resourceWidth: width, resourceHeight: height)
    for (controlMask, controls, historyTransform) in [
      (nil, NeuralRenderingFeatureControls(), nil),
      (mask, NeuralRenderingFeatureControls(normalizedStyle: 0.5, localToneStrength: 0.25, localStructureStrength: 0.75), transform),
      (nil, NeuralRenderingFeatureControls(
        localStructureStrength: 0.5,
        automaticMask: NeuralRenderingAutomaticMaskConfiguration(skinStructureStrength: -1, automaticMaskStructureStrength: 0.125)),
       nil),
    ] as [(MLXArray?, NeuralRenderingFeatureControls, NeuralRenderingTextureTransform?)] {
      let networkBase = base(
        color: gathered(color), controlMask: controlMask.map(gathered), noiseFrameIndex: 5, featureControls: controls)
      let extendedBase = base(color: color, controlMask: controlMask, noiseFrameIndex: 5, featureControls: controls,
        extension: indices)
      XCTAssertEqual(bits(extendedBase), bits(networkBase), "first frame")

      for historyConfidence in [nil, confidence] {
        let logical = temporal(
          baseFeatures: networkBase[0..., 0..<height, 0..<width, 0...], history: history,
          historyTransform: historyTransform, motion: motion, depth: depth, depthInverted: false,
          historyConfidence: historyConfidence, featureControls: controls)
        let reference = concatenated(
          [networkBase[0..., 0..., 0..., 0..<3], gathered(logical)[0..., 0..., 0..., 3...]], axis: -1)
        let extended = temporal(
          color: color, controlMask: controlMask, noiseFrameIndex: 5, history: history,
          historyTransform: historyTransform, motion: motion, depth: depth, depthInverted: false,
          historyConfidence: historyConfidence, featureControls: controls, extension: indices)
        XCTAssertEqual(bits(extended), bits(reference), "history, confidence \(historyConfidence != nil)")

        let postprocessor = MLXTemporalPostprocessor()
        let head = MLXRandom.normal([1, geometry.networkHeight, geometry.networkWidth, 4]).asType(.float16)
        for intensity in [Float(1), 0.6] {
          let cropped = postprocessor(
            head: head[0..., 0..<height, 0..<width, 0...], currentColor: color, features: logical, hasHistory: true,
            historyConfidence: historyConfidence, controlMask: controlMask, intensity: intensity)
          let padded = postprocessor(
            head: head, currentColor: color, features: extended, hasHistory: true,
            historyConfidence: historyConfidence, controlMask: controlMask, intensity: intensity)
          XCTAssertEqual(bits(padded), bits(cropped), "postprocessor, intensity \(intensity)")
        }
      }
    }
  }

  func testFusedDeviceFeatureKernelMatchesSequentialDeviceKernels() throws {
    let current = try smallTensor(
      name: "color", channels: 3, width: 2, height: 2,
      values: [
        0, 0.25, 0.5, 0.75, 1, 0.125,
        0.375, 0.625, 0.875, 1, 0.5, 0,
      ]
    )
    let history = try smallTensor(
      name: "history", channels: 3, width: 2, height: 2,
      values: [
        1, 0, 0.5, 0, 1, 0.25,
        0.5, 0.25, 1, 0.125, 0.75, 0.375,
      ]
    )
    let motion = try smallTensor(
      name: "motion", channels: 2, width: 2, height: 2,
      values: [0, 0, 0.25, 0, 0, -0.25, -0.25, 0]
    )
    let depth = try smallTensor(
      name: "depth", channels: 1, width: 2, height: 2,
      values: [1, 0.75, 0.5, 0.25]
    )
    let controlMask = try smallTensor(
      name: "controlMask", channels: 3, width: 2, height: 2,
      values: [
        1, 0.25, 0.75, 0.5, 1, 0,
        0, 0.5, 0.25, 0.75, 0, 1,
      ]
    )
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )
    let baseProcessor = MLXFirstFrameFeatureProcessor()
    let temporalProcessor = MLXTemporalFeatureProcessor()
    let currentArray = array(current)
    let maskArray = array(controlMask)
    let base = baseProcessor(
      color: currentArray,
      controlMask: maskArray,
      noiseFrameIndex: 11,
      featureControls: controls
    )
    let sequential = temporalProcessor(
      baseFeatures: base,
      history: array(history),
      motion: array(motion),
      depth: array(depth),
      depthInverted: false,
      featureControls: controls
    )

    let fused = temporalProcessor(
      color: currentArray,
      controlMask: maskArray,
      noiseFrameIndex: 11,
      history: array(history),
      motion: array(motion),
      depth: array(depth),
      depthInverted: false,
      featureControls: controls
    )
    eval(sequential, fused)

    XCTAssertEqual(fused.asArray(Float.self), sequential.asArray(Float.self))
  }

  func testDeviceFeatureKernelMatchesRenderedHistoryReference() async throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the rendered-history probe")
    }
    let reference = NeuralRenderingTemporalReferenceBackend(
      backend: try MLXNeuralRenderer(packageURL: URL(fileURLWithPath: packagePath))
    )
    let first = try await reference.render(request(frameIndex: 0, phase: 0))
    let currentRequest = try request(frameIndex: 1, phase: 31)
    let current = currentRequest.input(named: "color")!
    let history = first.output(named: "color")!
    let motion = currentRequest.input(named: "motion")!
    let depth = currentRequest.input(named: "depth")!
    let base = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: current,
      noiseFrameIndex: 1
    )
    let expected = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      normalizedMotion: motion,
      depth: depth,
      noiseFrameIndex: 1
    )

    let output = MLXTemporalFeatureProcessor()(
      baseFeatures: array(base),
      history: array(history),
      motion: array(motion),
      depth: array(depth),
      depthInverted: false
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), values(expected))
  }

  func testDeviceFeatureKernelMatchesRenderedHistoryTransformsAndControls() async throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the rendered-history probe")
    }
    let historyTransform = try NeuralRenderingTextureTransform(
      baseX: 8,
      baseY: 4,
      extentWidth: 112,
      extentHeight: 120,
      resourceWidth: 128,
      resourceHeight: 128
    )
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 3,
      baseY: 5,
      extentWidth: 120,
      extentHeight: 116,
      resourceWidth: 128,
      resourceHeight: 128
    )
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )
    let preprocessor = NeuralRenderingCPUTemporalFeaturePreprocessor(
      depthGuideMode: .observedZeroDescriptor,
      historyTransform: historyTransform,
      motionTransform: motionTransform,
      featureControls: controls
    )
    let reference = NeuralRenderingTemporalReferenceBackend(
      backend: try MLXNeuralRenderer(
        packageURL: URL(fileURLWithPath: packagePath)
      ),
      temporalPreprocessor: preprocessor
    )
    let first = try await reference.render(request(frameIndex: 0, phase: 0))
    let currentRequest = try request(frameIndex: 1, phase: 31)
    let current = currentRequest.input(named: "color")!
    let history = first.output(named: "color")!
    let motion = currentRequest.input(named: "motion")!
    let depth = currentRequest.input(named: "depth")!
    let expected = try preprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      controlMask: nil,
      normalizedMotion: motion,
      depth: depth,
      depthInverted: false,
      noiseFrameIndex: 1
    )

    let output = MLXTemporalFeatureProcessor()(
      color: array(current),
      noiseFrameIndex: 1,
      history: array(history),
      historyTransform: historyTransform,
      motion: array(motion),
      motionTransform: motionTransform,
      depth: array(depth),
      depthInverted: false,
      featureControls: controls
    )
    eval(output)

    let actual = output.asArray(Float.self)
    let referenceValues = values(expected)
    for pixel in 0..<(128 * 128) {
      for channel in 3..<16 {
        let offset = pixel * 16 + channel
        XCTAssertEqual(actual[offset], referenceValues[offset])
      }
    }
  }

  func testDevicePostprocessorMatchesPortableReferenceBytes() throws {
    let width = 8
    let height = 8
    let current = try smallTensor(
      name: "color",
      channels: 3,
      width: width,
      height: height,
      values: (0..<(width * height * 3)).map { Float($0 % 29) / 28 }
    )
    let head = try smallTensor(
      name: "color",
      channels: 4,
      width: width,
      height: height,
      values: (0..<(width * height * 4)).map { Float(($0 * 7) % 31) / 15 - 1 }
    )
    var featureValues = [Float](repeating: 0, count: width * height * 16)
    for pixel in 0..<(width * height) {
      featureValues[pixel * 16 + 7] = Float(pixel % 9) / 64 - 0.0625
      featureValues[pixel * 16 + 8] = Float(pixel % 7) / 64 - 0.046875
      featureValues[pixel * 16 + 9] = Float(pixel % 5) / 64 - 0.03125
    }
    let features = try smallTensor(
      name: "color",
      channels: 16,
      width: width,
      height: height,
      values: featureValues
    )
    let expected = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: current,
      features: features
    )

    let output = MLXTemporalPostprocessor()(
      head: array(head),
      currentColor: array(current),
      features: array(features),
      hasHistory: true
    )
    eval(output)

    let errors = zip(output.asArray(Float.self), values(expected)).map {
      abs($0.0 - $0.1)
    }
    XCTAssertEqual(errors.max() ?? 0, 0)
  }

  func testDevicePostprocessorMatchesTemporalControlMaskReference() throws {
    let current = try smallTensor(
      name: "color",
      channels: 3,
      width: 1,
      height: 1,
      values: [0.2, 0.4, 0.6]
    )
    let head = try smallTensor(
      name: "color",
      channels: 4,
      width: 1,
      height: 1,
      values: [1, -1, 0, 0]
    )
    var featureValues = [Float](repeating: 0, count: 16)
    featureValues[7] = 0.0625
    featureValues[8] = -0.0625
    let features = try smallTensor(
      name: "color",
      channels: 16,
      width: 1,
      height: 1,
      values: featureValues
    )
    let controlMask = try smallTensor(
      name: "controlMask",
      channels: 3,
      width: 1,
      height: 1,
      values: [0.25, 0.5, 0.75]
    )
    let expected = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: current,
      features: features,
      blendScale: 0.5,
      controlMask: controlMask,
      intensity: 0.8
    )

    let output = MLXTemporalPostprocessor(blendScale: 0.5)(
      head: array(head),
      currentColor: array(current),
      features: array(features),
      hasHistory: true,
      controlMask: array(controlMask),
      intensity: 0.8
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), values(expected))
  }

  func testDevicePostprocessorMatchesPortableSigmoidBytes() throws {
    let width = 8
    let height = 8
    let pixelCount = width * height
    let current = try smallTensor(
      name: "color",
      channels: 3,
      width: width,
      height: height,
      values: [Float](repeating: 0, count: pixelCount * 3)
    )
    var headValues = [Float](repeating: 0, count: pixelCount * 4)
    var featureValues = [Float](repeating: 0, count: pixelCount * 16)
    for pixel in 0..<pixelCount {
      headValues[pixel * 4 + 3] = Float((pixel * 7) % 31) / 15 - 1
      for channel in 7...9 {
        featureValues[pixel * 16 + channel] = 0.0625
      }
    }
    let head = try smallTensor(
      name: "color",
      channels: 4,
      width: width,
      height: height,
      values: headValues
    )
    let features = try smallTensor(
      name: "color",
      channels: 16,
      width: width,
      height: height,
      values: featureValues
    )
    let expected = try NeuralRenderingTemporalReferencePostprocessor.compose(
      head: head,
      currentColor: current,
      features: features
    )

    let output = MLXTemporalPostprocessor()(
      head: array(head),
      currentColor: array(current),
      features: array(features),
      hasHistory: true
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), values(expected))
  }

  func testDeviceFeatureKernelMatchesPortableReferenceBytes() throws {
    let currentRequest = try request(frameIndex: 1, phase: 31)
    let historyRequest = try request(frameIndex: 0, phase: 0)
    let current = currentRequest.input(named: "color")!
    let history = historyRequest.input(named: "color")!
    let motion = currentRequest.input(named: "motion")!
    let depth = currentRequest.input(named: "depth")!
    let base = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: current,
      noiseFrameIndex: 1
    )
    for mode in [
      NeuralRenderingDepthGuideMode.observedZeroDescriptor,
      .closestDepth,
    ] {
      let expected = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
        currentColor: current,
        historyColor: history,
        normalizedMotion: motion,
        depth: depth,
        depthGuideMode: mode,
        noiseFrameIndex: 1
      )
      let output = MLXTemporalFeatureProcessor()(
        baseFeatures: array(base),
        history: array(history),
        motion: array(motion),
        depth: array(depth),
        depthInverted: false,
        depthGuideMode: mode
      )
      eval(output)

      XCTAssertEqual(
        output.asArray(Float.self),
        values(expected),
        "depth mode: \(mode.rawValue)"
      )
    }
  }

  func testDeviceFeatureKernelMatchesFractionalTextureTransforms() throws {
    let current = try smallTensor(
      name: "color",
      channels: 3,
      width: 2,
      height: 1,
      values: [Float](repeating: 0.5, count: 6)
    )
    let history = try smallTensor(
      name: "history",
      channels: 3,
      width: 4,
      height: 1,
      values: [
        0, 0.5, 0.5,
        0.5, 0.5, 0.5,
        1, 0.5, 0.5,
        0.5, 0.5, 0.5,
      ]
    )
    let motion = try smallTensor(
      name: "motion",
      channels: 2,
      width: 4,
      height: 1,
      values: [
        0, 0,
        0.5, 0,
        0, 0,
        -0.5, 0,
      ]
    )
    let depth = try smallTensor(
      name: "depth",
      channels: 1,
      width: 2,
      height: 1,
      values: [1, 1]
    )
    let historyTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 1,
      resourceWidth: 4,
      resourceHeight: 1
    )
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 4,
      extentHeight: 1,
      resourceWidth: 4,
      resourceHeight: 1
    )
    let base = try NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(
      from: current,
      noiseFrameIndex: 1
    )
    let expected = try NeuralRenderingTemporalReferencePreprocessor.makeFeatureTensor(
      currentColor: current,
      historyColor: history,
      historyTransform: historyTransform,
      normalizedMotion: motion,
      motionTransform: motionTransform,
      depth: depth,
      noiseFrameIndex: 1
    )

    let output = MLXTemporalFeatureProcessor()(
      baseFeatures: array(base),
      history: array(history),
      historyTransform: historyTransform,
      motion: array(motion),
      motionTransform: motionTransform,
      depth: array(depth),
      depthInverted: false
    )
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), values(expected))
  }

  func testDeviceResidentBackendMatchesPortableTwoFrameReference() async throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the device pipeline probe")
    }
    let packageURL = URL(fileURLWithPath: packagePath)
    let historyTransform = try NeuralRenderingTextureTransform(
      baseX: 8,
      baseY: 4,
      extentWidth: 112,
      extentHeight: 120,
      resourceWidth: 128,
      resourceHeight: 128
    )
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 3,
      baseY: 5,
      extentWidth: 120,
      extentHeight: 116,
      resourceWidth: 128,
      resourceHeight: 128
    )
    let controls = NeuralRenderingFeatureControls(
      normalizedStyle: 0.5,
      localToneStrength: 0.25,
      localStructureStrength: 0.75,
      automaticMask: NeuralRenderingAutomaticMaskConfiguration(
        skinStructureStrength: -1,
        automaticMaskStructureStrength: 0.125
      )
    )
    let renderer = try MLXNeuralRenderer(packageURL: packageURL)
    let reference = NeuralRenderingTemporalReferenceBackend(
      backend: renderer,
      blendScale: renderer.blendScale ?? MLXTemporalPostprocessor.defaultBlendScale,
      temporalPreprocessor: NeuralRenderingCPUTemporalFeaturePreprocessor(
        depthGuideMode: .observedZeroDescriptor,
        historyTransform: historyTransform,
        motionTransform: motionTransform,
        featureControls: controls
      )
    )
    let device = try MLXNeuralRenderingDeviceTemporalBackend(
      packageURL: packageURL,
      historyTransform: historyTransform,
      motionTransform: motionTransform,
      featureControls: controls
    )

    let referenceFirst = try await reference.render(request(frameIndex: 0, phase: 0))
    let deviceFirst = try await device.render(request(frameIndex: 0, phase: 0))
    let referenceSecond = try await reference.render(request(frameIndex: 1, phase: 31))
    let deviceSecond = try await device.render(request(frameIndex: 1, phase: 31))

    assertClose(
      deviceFirst,
      referenceFirst,
      maximumAbsoluteError: deviceFeatureDisplayTolerance
    )
    assertClose(
      deviceSecond,
      referenceSecond,
      maximumAbsoluteError: deviceFeatureDisplayTolerance
    )
  }

  func testDeviceResidentBackendMatchesPortableTemporalControlMask() async throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the device pipeline probe")
    }
    let packageURL = URL(fileURLWithPath: packagePath)
    let renderer = try MLXNeuralRenderer(packageURL: packageURL)
    let reference = NeuralRenderingTemporalReferenceBackend(
      backend: renderer,
      blendScale: renderer.blendScale ?? MLXTemporalPostprocessor.defaultBlendScale,
      controlMaskIntensity: 0.8
    )
    let device = try MLXNeuralRenderingDeviceTemporalBackend(
      packageURL: packageURL,
      controlMaskIntensity: 0.8
    )
    var mask = [Float](repeating: 0, count: 128 * 128 * 3)
    for pixel in 0..<(128 * 128) {
      mask[pixel * 3] = 0.25
      mask[pixel * 3 + 1] = 0.5
      mask[pixel * 3 + 2] = 0.75
    }

    let referenceFirst = try await reference.render(request(frameIndex: 0, phase: 0))
    let deviceFirst = try await device.render(request(frameIndex: 0, phase: 0))
    let referenceSecond = try await reference.render(
      request(frameIndex: 1, phase: 31, controlMask: mask)
    )
    let deviceSecond = try await device.render(
      request(frameIndex: 1, phase: 31, controlMask: mask)
    )

    assertClose(
      deviceFirst,
      referenceFirst,
      maximumAbsoluteError: deviceFeatureDisplayTolerance
    )
    assertClose(
      deviceSecond,
      referenceSecond,
      maximumAbsoluteError: deviceFeatureDisplayTolerance
    )
  }

  func testDeviceBackendRejectsMismatchedMotionTransformResource() async throws {
    guard
      let packagePath = ProcessInfo.processInfo.environment[
        "MLXDLSS_NEURAL_RENDERING_PACKAGE"
      ]
    else {
      throw XCTSkip("set MLXDLSS_NEURAL_RENDERING_PACKAGE to run the device pipeline probe")
    }
    let motionTransform = try NeuralRenderingTextureTransform(
      baseX: 0,
      baseY: 0,
      extentWidth: 64,
      extentHeight: 128,
      resourceWidth: 64,
      resourceHeight: 128
    )
    let device = try MLXNeuralRenderingDeviceTemporalBackend(
      packageURL: URL(fileURLWithPath: packagePath),
      motionTransform: motionTransform
    )

    do {
      _ = try await device.render(request(frameIndex: 0, phase: 0))
      XCTFail("expected the mismatched motion resource to be rejected")
    } catch {
      XCTAssertEqual(
        error as? NeuralRenderingTextureTransformError,
        .resourceShapeMismatch(
          name: "motion",
          expectedWidth: 64,
          expectedHeight: 128,
          actualWidth: 128,
          actualHeight: 128
        )
      )
    }
  }

  private func request(
    frameIndex: UInt64,
    phase: Int,
    controlMask: [Float]? = nil
  ) throws -> NeuralRenderRequest {
    let pixelCount = 128 * 128
    let color = (0..<(pixelCount * 3)).map {
      Float(($0 + phase) % 257) / 256
    }
    let motion = (0..<(pixelCount * 2)).map { index in
      let pixel = index / 2
      if index.isMultiple(of: 2) {
        return Float((pixel * 3) % 11 - 5) / 512
      }
      return Float((pixel * 5) % 13 - 6) / 512
    }
    let depth = (0..<pixelCount).map {
      Float(($0 * 11 + phase) % 251) / 250
    }
    var inputs = [
      try tensor(name: "color", channels: 3, values: color),
      try tensor(name: "motion", channels: 2, values: motion),
      try tensor(name: "depth", channels: 1, values: depth),
    ]
    if let controlMask {
      inputs.append(
        try tensor(name: "controlMask", channels: 3, values: controlMask)
      )
    }
    return try NeuralRenderRequest(
      sequenceID: frameIndex,
      temporalContext: NeuralRenderFrameContext(streamID: 17, frameIndex: frameIndex),
      inputs: inputs
    )
  }

  private func tensor(
    name: String,
    channels: Int,
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: [1, 128, 128, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func smallTensor(
    name: String,
    channels: Int,
    width: Int,
    height: Int,
    values: [Float]
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: name,
        shape: [1, height, width, channels],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func assertClose(
    _ actual: NeuralRenderResult,
    _ expected: NeuralRenderResult,
    maximumAbsoluteError: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    let actualValues = values(actual.output(named: "color")!)
    let expectedValues = values(expected.output(named: "color")!)
    XCTAssertEqual(actualValues.count, expectedValues.count, file: file, line: line)
    let maximum = zip(actualValues, expectedValues).enumerated().max {
      abs($0.element.0 - $0.element.1) < abs($1.element.0 - $1.element.1)
    }
    let error = maximum.map { abs($0.element.0 - $0.element.1) } ?? 0
    let diagnostic =
      maximum.map {
        "index \($0.offset): actual=\($0.element.0), expected=\($0.element.1)"
      } ?? "empty output"
    XCTAssertLessThanOrEqual(
      error,
      maximumAbsoluteError,
      diagnostic,
      file: file,
      line: line
    )
  }

  private func values(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private func array(_ tensor: HostTensor) -> MLXArray {
    MLXArray(tensor.bytes, tensor.descriptor.shape, dtype: .float32)
  }
}
