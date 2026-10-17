import Foundation
import MLX
import XCTest

@testable import DLSSMLX

final class NeuralRenderingExecutionTests: XCTestCase {
  func testHalfPrecisionChunkedFeedForwardMatchesUnchunked() {
    let input = sin(arange(40 * 32, dtype: .float32) * 0.01)
      .reshaped([1, 5, 8, 32]).asType(.float16)
    let expansion = (cos(arange(32 * 128, dtype: .float32) * 0.003) * 0.05)
      .reshaped([32, 128]).asType(.float16)
    let projection = (sin(arange(128 * 32, dtype: .float32) * 0.005) * 0.05)
      .reshaped([128, 32]).asType(.float16)
    let expected = NeuralRenderingTransformerOperations.fusedSimpleFeedForward(
      input, expansionWeight: expansion, projectionWeight: projection)
    let actual = NeuralRenderingTransformerOperations.fusedSimpleFeedForward(
      input, expansionWeight: expansion, projectionWeight: projection,
      maximumIntermediateBytes: 4_096)
    XCTAssertEqual(actual.shape, input.shape)
    XCTAssertEqual(actual.dtype, .float16)
    XCTAssertLessThanOrEqual(
      abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self), 0.001)
  }

  func testExternalGlobalStageMatchesEvaluationAfterEveryBlockWhenConfigured() throws {
    let path = try htLogicalWeightsPath("set MLXDLSS_LOGICAL_WEIGHTS to check recovered global-stage parity")
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let weights = ValidatedWeights(arrays: arrays).cast(to: .float16)
    let stage = try NeuralRenderingGlobalStage(weights: weights)
    let blocks = try (31...38).map {
      try NeuralRenderingGlobalBlock(weights: weights, blockIndex: $0)
    }
    for (h, w) in [(8, 8), (16, 32)] {
      let input = sin(arange(h * w * 1024, dtype: .float32) * 0.03)
        .reshaped([1, h, w, 1024]).asType(.float16)
      var expected = input
      for block in blocks {
        expected = block(expected)
        eval(expected)
        Memory.clearCache()
      }
      let actual = stage(input)
      XCTAssertEqual(actual.shape, input.shape)
      XCTAssertEqual(actual.dtype, .float16)
      XCTAssertEqual(
        abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self), 0)
    }
  }
}
