import AVFoundation
import CoreVideo
import DLSSMLX
import Foundation
import VideoToolbox

/// Native receiver backpressure suspends the task instead of polling an encoder
/// or moving frame bytes through a subprocess pipe.
@available(macOS 26.0, *)
actor NativeVideoWriter {
  private let writer: AVAssetWriter
  private let video: AVAssetWriterInput.PixelBufferReceiver
  private let audio: AVAssetWriterInput.SampleBufferReceiver?
  private let packer: MLXPixelBufferWriter
  private var previousTime: CMTime?
  private var endTime: CMTime?
  private var audioFinished = false
  private var complete = false

  init(url: URL, width: Int, height: Int, frameRate: Double,
       options: MediaProcessingOptions, hasAudio: Bool) throws {
    guard !FileManager.default.fileExists(atPath: url.path) else {
      throw MLXMediaError("Output already exists: \(url.lastPathComponent)")
    }
    let fileType: AVFileType = url.pathExtension.lowercased() == "mov" ? .mov : .mp4
    guard fileType == .mov || options.codec != .prores else {
      throw MLXMediaError("ProRes output requires a .mov file")
    }
    guard width % 2 == 0, height % 2 == 0 else {
      throw MLXMediaError("Video encoding requires even frame dimensions")
    }
    writer = try AVAssetWriter(outputURL: url, fileType: fileType)
    writer.shouldOptimizeForNetworkUse = true
    let codec: AVVideoCodecType = options.codec == .h264 ? .h264 : options.codec == .hevc ? .hevc : .proRes422HQ
    var settings: [String: Any] = [AVVideoCodecKey: codec,
      AVVideoWidthKey: width, AVVideoHeightKey: height,
      AVVideoColorPropertiesKey: [
        AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
        AVVideoTransferFunctionKey: kCVImageBufferTransferFunction_sRGB,
        AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
      ],
      AVVideoEncoderSpecificationKey: [kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder as String: true],
    ]
    if options.codec != .prores {
      settings[AVVideoCompressionPropertiesKey] = [
        AVVideoAverageBitRateKey: options.bitrate ?? max(2_000_000, Int(Double(width * height) * frameRate * 0.16)),
        AVVideoExpectedSourceFrameRateKey: frameRate, AVVideoMaxKeyFrameIntervalDurationKey: 2,
      ]
    }
    guard writer.canApply(outputSettings: settings, forMediaType: .video) else {
      throw MLXMediaError("Selected video codec or dimensions are unsupported")
    }
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    video = writer.inputPixelBufferReceiver(for: input, pixelBufferAttributes: nil)
    if hasAudio {
      let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 48000,
        AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 256000,
      ])
      input.expectsMediaDataInRealTime = false
      audio = writer.inputReceiver(for: input)
    } else { audio = nil }
    packer = try MLXPixelBufferWriter(width: width, height: height)
    guard writer.startWriting() else { throw writer.error ?? MLXMediaError("Cannot start native encoder") }
    writer.startSession(atSourceTime: .zero)
  }

  func append(_ frame: MLXVideoFrame, at time: CMTime) async throws {
    try Task.checkCancellation()
    guard !complete, endTime == nil, time.isNumeric, time >= .zero,
      previousTime.map({ time > $0 }) ?? true else { throw MLXMediaError("Output timestamps must increase") }
    let buffer = try await packer.write(frame)
    let readOnly = CVReadOnlyPixelBuffer(unsafeBuffer: buffer.buffer)
    try await video.append(readOnly, with: time)
    previousTime = time
  }

  func appendAudio(_ sample: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>) async throws {
    try Task.checkCancellation()
    guard !complete, !audioFinished, let audio else { throw MLXMediaError("Audio encoder is unavailable") }
    if let endTime, sample.presentationTimeStamp >= endTime { return }
    try await audio.append(sample)
  }

  /// Closes the video track. The session ends in `finish(at:)`, after the audio
  /// task has appended the samples before `time`: AVAssetWriter refuses any
  /// append once the session has ended.
  func finishVideo(at time: CMTime) {
    guard !complete, endTime == nil else { return }
    endTime = time
    video.finish()
  }

  func finishAudio() {
    guard !complete, !audioFinished else { return }
    audioFinished = true
    audio?.finish()
  }

  func finish(at time: CMTime) async throws {
    guard !complete else { throw MLXMediaError("Encoder is already closed") }
    finishVideo(at: time)
    finishAudio()
    writer.endSession(atSourceTime: endTime ?? time)
    await writer.finishWriting()
    complete = true
    guard writer.status == .completed else { throw writer.error ?? MLXMediaError("Cannot finish native video output") }
  }

  func cancel() { if !complete { writer.cancelWriting(); complete = true } }
}
