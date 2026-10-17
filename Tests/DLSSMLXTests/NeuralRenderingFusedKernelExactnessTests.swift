import Foundation
import MLX
import XCTest

@testable import DLSSMLX

/// The fused kernels that replace operation chains must reproduce those chains
/// bit for bit: the branched feed-forward kernel, the multi-head attention block
/// kernel and the single-head window block variants with their glue.
final class NeuralRenderingFusedKernelExactnessTests: XCTestCase {
  private typealias Ops = NeuralRenderingTransformerOperations

  private func normal(_ shape: [Int], _ scale: Float) -> MLXArray {
    (MLXRandom.normal(shape) * scale).asType(.float16)
  }

  private func cosine(_ count: Int) -> MLXArray {
    MLXRandom.uniform(low: 0.5, high: 1.0, [count]).asType(.float16)
  }

  private func assertBitIdentical(_ expected: MLXArray, _ actual: MLXArray, _ label: String) {
    XCTAssertEqual(expected.shape, actual.shape, label)
    guard expected.shape == actual.shape else { return }
    let differing = (expected.view(dtype: .uint16) .!= actual.view(dtype: .uint16)).sum().item(Int32.self)
    XCTAssertEqual(differing, 0, "\(label): \(differing) of \(expected.size) values differ")
  }

  private static let origins = [
    NeuralRenderingWindowOrigin.zero, NeuralRenderingWindowOrigin(y: -4, x: -4),
    NeuralRenderingWindowOrigin(y: 0, x: -4), NeuralRenderingWindowOrigin(y: -4, x: 0),
  ]

  func testBranchedFeedForwardKernelMatchesTheGEMMChain() {
    MLXRandom.seed(3)
    for (groups, height, width) in [(2, 24, 40), (4, 9, 13), (8, 5, 3)] {
      let channels = groups * 32
      let input = Ops.e4m3RoundTrip(normal([1, height, width, channels], 1.5))
      let dense = Ops.denseExpansionWeight(normal([groups, 4, groups, 32, 32], 0.08))
      let grouped = Ops.groupedProjectionWeight(normal([groups, 4, 32, 32], 0.08))
      let output = normal([channels, channels], 0.08)
      let scale = cosine(channels)
      let expected = Device.withDefaultDevice(.gpu) {
        NeuralRenderingFusedWindowAttention.cosineResidual(
          skip: input,
          branch: Ops.denseBranchedFeedForward(
            input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output),
          cosine: scale, publish: true)
      }
      let actual = Device.withDefaultDevice(.gpu) {
        NeuralRenderingFusedBranchedFeedForward.apply(
          input, denseExpansionWeight: dense, groupedProjectionWeight: grouped, outputProjectionWeight: output,
          cosine: scale)
      }
      assertBitIdentical(expected, actual, "feed-forward \(channels)ch \(height)x\(width)")
    }
  }

  func testAttentionBlockKernelMatchesTheKernelChain() {
    MLXRandom.seed(5)
    for (heads, height, width) in [(2, 37, 53), (4, 16, 24), (8, 13, 21)] {
      let channels = heads * 32
      let input = Ops.e4m3RoundTrip(normal([1, height, width, channels], 1.5))
      let qkv = normal([channels, channels * 3], 0.08)
      let scale = MLXRandom.uniform(low: 2, high: 12, [heads]).asType(.float16)
      let bias = normal([heads, 64, 64], 2.0)
      let projection = normal([channels, channels], 0.08)
      let residualCosine = cosine(channels)
      for origin in Self.origins {
        for publish in [true, false] {
          let expected = Device.withDefaultDevice(.gpu) {
            Ops.outputResidual(
              skip: input,
              branch: Ops.windowAttention(
                input, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, projectionWeight: projection,
                headCount: heads, windowSize: 8, windowOrigin: origin, preciseSoftmax: true, fusedAttention: true),
              cosine: residualCosine, publishedOutput: publish)
          }
          let actual = Device.withDefaultDevice(.gpu) {
            NeuralRenderingFusedWindowAttentionBlock.apply(
              input, qkvWeight: qkv, attentionScale: scale, attentionBias: bias, projectionWeight: projection,
              cosine: residualCosine, windowOrigin: origin, publish: publish)
          }
          assertBitIdentical(expected, actual, "attention \(heads)h \(height)x\(width) \(origin) publish \(publish)")
        }
      }
    }
  }

  private func singleHeadWeights() -> NeuralRenderingFusedWindowBlock.Weights {
    NeuralRenderingFusedWindowBlock.Weights(
      expansion: normal([32, 128], 0.2), feedForwardProjection: normal([128, 32], 0.2),
      feedForwardCosine: cosine(32), qkv: normal([32, 96], 0.2),
      attentionScale: MLXArray([Float(3.0)]).asType(.float16), attentionBias: normal([1, 64, 64], 0.3),
      attentionProjection: normal([32, 32], 0.2), attentionCosine: cosine(32))
  }

  private func block(
    _ input: MLXArray, _ weights: NeuralRenderingFusedWindowBlock.Weights, _ origin: NeuralRenderingWindowOrigin,
    publish: Bool
  ) -> MLXArray {
    NeuralRenderingFusedWindowBlock.apply(
      input, expansionWeight: weights.expansion, feedForwardProjectionWeight: weights.feedForwardProjection,
      feedForwardCosine: weights.feedForwardCosine, qkvWeight: weights.qkv, attentionScale: weights.attentionScale,
      attentionBias: weights.attentionBias, attentionProjectionWeight: weights.attentionProjection,
      attentionCosine: weights.attentionCosine, windowOrigin: origin, publish: publish)
  }

  func testSingleHeadVariantsMatchTheBlockAndGlueKernels() {
    MLXRandom.seed(9)
    let weights = singleHeadWeights()
    for (height, width) in [(40, 56), (34, 50)] {
      // Stem: adapter, block 0, publication and pool.
      let network = normal([1, height, width, 16], 1.0)
      let adapter = normal([16, 32], 0.3)
      let stem = NeuralRenderingFusedWindowBlock.applyVariant(
        .adapter(input: network, weight: adapter), weights: weights, windowOrigin: .zero, epilogue: .pool, publish: true)
      let stemReference = NeuralRenderingFusedGlue.publishAndPool(
        block(matmul(network, adapter), weights, .zero, publish: false), emitFull: true)
      assertBitIdentical(stemReference.full, stem[1], "stem full \(height)x\(width)")
      assertBitIdentical(stemReference.pooled, stem[0], "stem pooled \(height)x\(width)")

      let input = Ops.e4m3RoundTrip(normal([1, height, width, 32], 1.0))
      for origin in Self.origins {
        // Downsample: block and pool without the full-resolution output.
        let pooled = NeuralRenderingFusedWindowBlock.applyVariant(
          .input(input), weights: weights, windowOrigin: origin, epilogue: .pool, publish: false)[0]
        let pooledReference = NeuralRenderingFusedGlue.publishAndPool(
          block(input, weights, origin, publish: false), emitFull: false).pooled
        assertBitIdentical(pooledReference, pooled, "pool \(height)x\(width) \(origin)")

        // Upsample: published merge and the published block.
        let low = Ops.e4m3RoundTrip(normal([1, (height + 1) / 2, (width + 1) / 2, 32], 1.0))
        let skipScale = cosine(32)
        let merged = NeuralRenderingFusedWindowBlock.applyVariant(
          .merge(low: low, skip: input, lowScale: nil, skipScale: skipScale, publish: true), weights: weights,
          windowOrigin: origin, epilogue: .store, publish: true)[0]
        let mergedReference = block(
          NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: input, lowScale: nil, skipScale: skipScale, publish: true),
          weights, origin, publish: true)
        assertBitIdentical(mergedReference, merged, "merge \(height)x\(width) \(origin)")

        // Post: merge, block and head.
        let lowScale = cosine(32)
        let headWeight = normal([32, 8], 0.3)
        let head = NeuralRenderingFusedWindowBlock.applyVariant(
          .merge(low: low, skip: input, lowScale: lowScale, skipScale: skipScale, publish: false), weights: weights,
          windowOrigin: origin, epilogue: .head, publish: false, headWeight: headWeight)[0]
        let features = block(
          NeuralRenderingFusedGlue.upsampleMerge(low: low, skip: input, lowScale: lowScale, skipScale: skipScale, publish: false),
          weights, origin, publish: false)
        let projected = matmul(features, headWeight)
        let headReference = projected[0..., 0..., 0..., 0..<4] + projected[0..., 0..., 0..., 4..<8]
        assertBitIdentical(headReference, head, "head \(height)x\(width) \(origin)")
      }
    }
  }
}
