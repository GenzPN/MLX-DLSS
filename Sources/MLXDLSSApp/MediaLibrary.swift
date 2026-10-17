import AVFoundation
import CoreGraphics
import ImageIO

/// Source facts shown before any rendering runs.
struct MediaInfo: Equatable, Sendable {
  var width: Int
  var height: Int
  var duration: Double?
  var frameRate: Double?
}

/// Cheap, decoder-only views of a source file: metadata, thumbnails, a filmstrip
/// and an immediate frame for a timeline position. Nothing here touches the GPU
/// model, so these return while a neural-rendering preview is still running.
enum MediaLibrary {
  static func info(_ url: URL, isVideo: Bool) async -> MediaInfo? {
    if !isVideo {
      guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? Int,
        let height = properties[kCGImagePropertyPixelHeight] as? Int
      else { return nil }
      let rotated = ((properties[kCGImagePropertyOrientation] as? UInt32) ?? 1) >= 5
      return MediaInfo(width: rotated ? height : width, height: rotated ? width : height)
    }
    let asset = AVURLAsset(url: url)
    guard let track = try? await asset.loadTracks(withMediaType: .video).first,
      let loaded = try? await track.load(.naturalSize, .preferredTransform, .nominalFrameRate, .timeRange)
    else { return nil }
    let (size, transform, rate, range) = loaded
    let oriented = CGRect(origin: .zero, size: size).applying(transform)
    return MediaInfo(width: Int(abs(oriented.width).rounded()), height: Int(abs(oriented.height).rounded()),
      duration: range.end.seconds.isFinite ? range.end.seconds : nil,
      frameRate: rate.isFinite && rate > 0 ? Double(rate) : nil)
  }

  /// A frame at `time` (ignored for images), at most `maxPixels` on its long side.
  static func frame(_ url: URL, isVideo: Bool, time: Double = 0, maxPixels: Int) async -> CGImage? {
    if !isVideo {
      guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
      return CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxPixels,
      ] as CFDictionary)
    }
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: maxPixels, height: maxPixels)
    let tolerance = CMTime(value: 1, timescale: 120)
    generator.requestedTimeToleranceBefore = tolerance
    generator.requestedTimeToleranceAfter = tolerance
    return try? await generator.image(at: CMTime(seconds: time, preferredTimescale: 60000)).image
  }

  /// Frames for timeline positions of one clip at a time; keeps the asset and its
  /// generator open while the playhead moves.
  actor Scrubber {
    private var url: URL?
    private var generator: AVAssetImageGenerator?

    func frame(_ url: URL, time: Double, maxPixels: Int) async -> CGImage? {
      if url != self.url || generator == nil {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: maxPixels, height: maxPixels)
        let tolerance = CMTime(value: 1, timescale: 120)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        self.generator = generator
        self.url = url
      }
      let generator = generator!, requested = CMTime(seconds: time, preferredTimescale: 60000)
      return await withCheckedContinuation { continuation in
        generator.generateCGImageAsynchronously(for: requested) { image, _, _ in continuation.resume(returning: image) }
      }
    }
  }

  /// Evenly spaced frames across the clip for the timeline.
  static func filmstrip(_ url: URL, duration: Double, count: Int, maxPixels: Int) async -> [CGImage] {
    let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: maxPixels, height: maxPixels)
    // Keyframe-near frames decode fastest; exactness does not matter here.
    generator.requestedTimeToleranceBefore = .positiveInfinity
    generator.requestedTimeToleranceAfter = .positiveInfinity
    let times = (0..<count).map { CMTime(seconds: duration * (Double($0) + 0.5) / Double(count), preferredTimescale: 600) }
    var frames: [CGImage] = []
    for await result in generator.images(for: times) {
      guard let image = try? result.image else { continue }
      frames.append(image)
    }
    return frames
  }
}
