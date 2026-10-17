import CoreMedia
import CoreVideo
import DLSSMLX
import Foundation
import VideoToolbox
import Vision

/// One estimator owns one sequential session. Both directions are checked before
/// a history sample is accepted; a scene cut also resets the session's hints.
public actor NativeOpticalFlow {
  public nonisolated let backend: String
  private let mode: MediaMotion
  private let imageIO: NativeImageIO
  private let halfWriter: MLXPixelBufferWriter
  private let width: Int
  private let height: Int
  private let workingWidth: Int
  private let workingHeight: Int
  private var session: Any?
  private var previous: (rgb: MLXVideoFrame, pixels: MLXPixelBuffer, index: Int)?
  private var randomAccess = true
  private var inFlight = false

  public init(width: Int, height: Int, mode: MediaMotion = .automatic) throws {
    self.width = width
    self.height = height
    imageIO = try NativeImageIO()
    halfWriter = try MLXPixelBufferWriter(width: width, height: height, halfOutput: true)
    // These configuration extents produce an exact quarter-size flow
    // grid. Arbitrary extents can return a larger, padded allocation; resizing
    // explicitly keeps vector units and image coordinates unambiguous.
    (workingWidth, workingHeight) = width * height <= 640 * 480 ? (640, 480) : (960, 540)
    var selected = mode
    if mode == .automatic {
      if #available(macOS 15.4, *), VTOpticalFlowConfiguration.isSupported { selected = .videoToolbox }
      else { selected = .vision }
    }
    if selected == .videoToolbox {
      guard #available(macOS 15.4, *), VTOpticalFlowConfiguration.isSupported else {
        throw MLXMediaError("VideoToolbox optical flow is unavailable on this Mac")
      }
      do {
        session = try VideoToolboxFlowSession(width: workingWidth, height: workingHeight)
      } catch let error as VTFrameProcessorError where mode == .automatic && error.code == .initializationFailed {
        // A VM can advertise support without a working frame-processor driver.
        selected = .vision
      }
    }
    self.mode = selected
    backend = selected.rawValue
  }

  /// Starts a new sequence on the same session: the next frame has no
  /// predecessor and its successor goes in as random access, as on a new estimator.
  public func reset() {
    previous = nil
    randomAccess = true
  }

  public func prepare(_ frame: MLXVideoFrame, index: Int, sceneCutThreshold: Float) async throws -> MLXVideoMotion? {
    guard !inFlight else { throw MLXMediaError("Optical flow frames must be submitted sequentially") }
    guard frame.width == width, frame.height == height else { throw MLXMediaError("Optical flow frame size changed") }
    guard mode != .zero else { return nil }
    inFlight = true
    defer { inFlight = false }
    let fullSize = try await halfWriter.write(frame)
    let pixels = mode == .videoToolbox
      ? try await imageIO.resize(fullSize, width: workingWidth, height: workingHeight) : fullSize
    let old = previous
    previous = (frame, pixels, index)
    guard let old else { return nil }
    let backward: MLXPixelBuffer, forward: MLXPixelBuffer
    let units: MLXVideoMotion.Units
    if #available(macOS 15.4, *), let session = session as? VideoToolboxFlowSession {
      (forward, backward) = try await session.process(previous: old.pixels, current: pixels,
        index: index, randomAccess: randomAccess || old.index + 1 != index)
      units = .flowPixels
    } else {
      forward = try visionFlow(from: old.pixels, to: pixels)
      backward = try visionFlow(from: pixels, to: old.pixels)
      units = .sourcePixels
    }
    let motion = try MLXVideoMotion(current: frame, previous: old.rgb, backward: backward,
      forward: forward, units: units, sceneCutThreshold: sceneCutThreshold)
    randomAccess = motion.reset
    return motion
  }

  private func visionFlow(from: MLXPixelBuffer, to: MLXPixelBuffer) throws -> MLXPixelBuffer {
    let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: to.buffer)
    request.revision = VNGenerateOpticalFlowRequestRevision1
    request.computationAccuracy = .high
    request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
    try VNImageRequestHandler(cvPixelBuffer: from.buffer).perform([request])
    guard let result = request.results?.first else { throw MLXMediaError("Vision returned no optical flow") }
    return MLXPixelBuffer(result.pixelBuffer)
  }
}

@available(macOS 15.4, *)
private final class VideoToolboxFlowSession {
  private let processor = VTFrameProcessor()
  private let width: Int
  private let height: Int

  init(width: Int, height: Int) throws {
    guard let configuration = VTOpticalFlowConfiguration(frameWidth: width, frameHeight: height,
      qualityPrioritization: .normal, revision: .revision1),
      let flowWidth = configuration.destinationPixelBufferAttributes[kCVPixelBufferWidthKey as String] as? Int,
      let flowHeight = configuration.destinationPixelBufferAttributes[kCVPixelBufferHeightKey as String] as? Int,
      flowWidth * 4 == width, flowHeight * 4 == height else {
      throw MLXMediaError("VideoToolbox returned an unexpected optical flow grid")
    }
    self.width = flowWidth
    self.height = flowHeight
    try processor.startSession(configuration: configuration)
  }

  deinit { processor.endSession() }

  func process(previous: MLXPixelBuffer, current: MLXPixelBuffer, index: Int,
               randomAccess: Bool, isolation: isolated (any Actor)? = #isolation) async throws -> (MLXPixelBuffer, MLXPixelBuffer) {
    let forward = try NativePixelBuffers.make(width: width, height: height, format: kCVPixelFormatType_TwoComponent16Half)
    let backward = try NativePixelBuffers.make(width: width, height: height, format: kCVPixelFormatType_TwoComponent16Half)
    guard let source = VTFrameProcessorFrame(buffer: previous.buffer, presentationTimeStamp: CMTime(value: Int64(index - 1), timescale: 60)),
      let next = VTFrameProcessorFrame(buffer: current.buffer, presentationTimeStamp: CMTime(value: Int64(index), timescale: 60)),
      let flow = VTFrameProcessorOpticalFlow(forwardFlow: forward.buffer, backwardFlow: backward.buffer),
      let parameters = VTOpticalFlowParameters(sourceFrame: source, nextFrame: next,
        submissionMode: randomAccess ? .random : .sequential, destinationOpticalFlow: flow) else {
      throw MLXMediaError("Cannot create VideoToolbox optical flow parameters")
    }
    // The macOS 26 AsyncSequence overload assumes destination video frames and
    // cannot process optical flow. This callback overload supports both outputs.
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
      processor.process(parameters: parameters, completionHandler: { _, error in
        if let error { continuation.resume(throwing: error) }
        else { continuation.resume() }
      })
    }
    return (forward, backward)
  }
}
