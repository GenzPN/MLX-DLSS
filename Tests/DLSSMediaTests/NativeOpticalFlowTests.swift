import CoreVideo
import DLSSMLX
import Foundation
import MLX
import VideoToolbox
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

final class NativeOpticalFlowTests: XCTestCase, @unchecked Sendable {
  func testVideoToolboxDirectionAndUnitsAtNonNativeExtent() async throws {
    guard #available(macOS 15.4, *), VTOpticalFlowConfiguration.isSupported else {
      throw XCTSkip("VideoToolbox optical flow is unavailable")
    }
    do {
      try await checkTranslation(mode: .videoToolbox)
    } catch let error as VTFrameProcessorError where error.code == .initializationFailed {
      throw XCTSkip("VideoToolbox cannot start its hardware processing pipeline")
    }
  }

  func testAutomaticBackendComputesMotion() async throws {
    try await checkTranslation(mode: .automatic)
  }

  func testVisionDirectionAndUnits() async throws {
    try await checkTranslation(mode: .vision)
  }

  /// The live preview keeps one estimator across seeks: after a reset the next
  /// frame starts a new sequence instead of pairing with the old window.
  func testResetStartsANewSequenceOnTheSameSession() async throws {
    let width = 512, height = 384
    let estimator = try NativeOpticalFlow(width: width, height: height, mode: .automatic)
    let first = try Self.texture(width: width, height: height, dx: 0, dy: 0)
    let second = try Self.texture(width: width, height: height, dx: 8, dy: -4)
    _ = try await estimator.prepare(first, index: 0, sceneCutThreshold: 0.3)
    _ = try await estimator.prepare(second, index: 1, sceneCutThreshold: 0.3)
    await estimator.reset()
    let restarted = try await estimator.prepare(first, index: 40, sceneCutThreshold: 0.3)
    XCTAssertNil(restarted)
    let optionalMotion = try await estimator.prepare(second, index: 41, sceneCutThreshold: 0.3)
    let motion = try XCTUnwrap(optionalMotion)
    let vectors = motion.vectors.asArray(Float.self)
    var horizontal: [Float] = []
    for y in height / 4..<3 * height / 4 { for x in width / 4..<3 * width / 4 {
      horizontal.append(vectors[(y * width + x) * 2] * Float(width))
    } }
    XCTAssertEqual(horizontal.sorted()[horizontal.count / 2], -8, accuracy: 0.7)
  }

  private func checkTranslation(mode: MediaMotion) async throws {
    let width = 512, height = 384
    let estimator = try NativeOpticalFlow(width: width, height: height, mode: mode)
    XCTAssertTrue([MediaMotion.videoToolbox.rawValue, MediaMotion.vision.rawValue].contains(estimator.backend))
    let previous = try Self.texture(width: width, height: height, dx: 0, dy: 0)
    let current = try Self.texture(width: width, height: height, dx: 8, dy: -4)
    let first = try await estimator.prepare(previous, index: 0, sceneCutThreshold: 0.3)
    XCTAssertNil(first)
    let optionalMotion = try await estimator.prepare(current, index: 1, sceneCutThreshold: 0.3)
    let motion = try XCTUnwrap(optionalMotion)
    let vectors = motion.vectors.asArray(Float.self)
    var horizontal: [Float] = [], vertical: [Float] = []
    for y in height / 4..<3 * height / 4 { for x in width / 4..<3 * width / 4 {
      horizontal.append(vectors[(y * width + x) * 2] * Float(width))
      vertical.append(vectors[(y * width + x) * 2 + 1] * Float(height))
    } }
    let dx = horizontal.sorted()[horizontal.count / 2], dy = vertical.sorted()[vertical.count / 2]
    print("native flow \(estimator.backend) (\(mode.rawValue)): dx=\(dx), dy=\(dy), coverage=\(motion.reliableFraction), error=\(motion.warpedLumaError)")
    XCTAssertEqual(dx, -8, accuracy: 0.7)
    XCTAssertEqual(dy, 4, accuracy: 0.7)
    XCTAssertGreaterThan(motion.reliableFraction, 0.5)
    XCTAssertLessThan(motion.warpedLumaError, 0.04)
    XCTAssertFalse(motion.reset)
  }

  static func texture(width: Int, height: Int, dx: Int, dy: Int) throws -> MLXVideoFrame {
    func noise(_ x: Int, _ y: Int, _ c: Int) -> Float {
      var value = UInt32(truncatingIfNeeded: x &* 73856093 ^ y &* 19349663 ^ c &* 83492791)
      value = (value ^ (value >> 13)) &* 1274126177
      return Float(value & 65535) / 65535
    }
    var pixels = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height { for x in 0..<width {
      let xx = max(0, min(width - 1, x - dx)), yy = max(0, min(height - 1, y - dy))
      let x0 = xx / 8, y0 = yy / 8, a = Float(xx % 8) / 8, b = Float(yy % 8) / 8
      for c in 0..<3 {
        let top = noise(x0, y0, c) * (1 - a) + noise(x0 + 1, y0, c) * a
        let bottom = noise(x0, y0 + 1, c) * (1 - a) + noise(x0 + 1, y0 + 1, c) * a
        pixels[(y * width + x) * 3 + c] = 0.15 + 0.7 * (top * (1 - b) + bottom * b)
      }
    } }
    return try pixels.withUnsafeBytes { try MLXVideoFrame(rgb: Data($0), width: width, height: height) }
  }
}
