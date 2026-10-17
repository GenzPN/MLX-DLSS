import AppKit
import DLSSCore
import DLSSMedia
import DLSSMLX
import Foundation
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class MediaJob: Identifiable {
  enum State: String { case queued = "Queued", running = "Processing", complete = "Done", cancelled = "Cancelled", failed = "Failed" }
  let id = UUID()
  let input: URL
  let isVideo: Bool
  var state: State = .queued
  var progress: MediaProgress?
  var result: MediaProcessingResult?
  var error: String?
  /// Loaded from the file after import; nil until then or when unreadable.
  var info: MediaInfo?
  var thumbnail: CGImage?
  var filmstrip: [CGImage] = []
  /// The timeline position last shown for this item.
  var viewedTime: Double?

  init(input: URL) {
    self.input = input
    isVideo = UTType(filenameExtension: input.pathExtension)?.conforms(to: .movie) == true
  }

  var name: String { input.lastPathComponent }
  var isFinished: Bool { state != .queued && state != .running }

  /// Remaining export time from the current rate, once a few frames have passed.
  var remainingSeconds: Double? {
    guard state == .running, let progress, progress.inputFrames >= 4, progress.fraction > 0.01 else { return nil }
    return progress.elapsedSeconds * (1 - progress.fraction) / progress.fraction
  }
}

/// A source frame decoded without the model, shown while a rendered preview is pending.
struct SourceFrame {
  let job: UUID
  let time: Double
  let image: CGImage
}

@available(macOS 26.0, *)
@MainActor @Observable
final class AppModel {
  var jobs: [MediaJob] = []
  var selection: UUID? {
    didSet {
      guard selection != oldValue else { return }
      previewTime = selectedJob?.viewedTime ?? 0
      preview = nil
      previewTimeRendered = nil
      previewError = nil
    }
  }
  var isRunning = false
  var alert: String?
  var renderingEnabled = true
  var generationEnabled = false
  var superResolutionEnabled = false
  var videoUpscaling = "dlss"
  var modelPath: String
  var generationPath: String
  var superResolutionPath: String
  var dlssSuperResolutionPath: String
  var outputDirectory: String
  var profile: NeuralRenderingControlProfile = .standard
  var temporal = true
  var motion: MediaMotion = .automatic
  var processingScale: Double = 1
  var detailStrength: Double = 1
  var colourStrength: Double = 1
  var detailRadius: Double = 4
  var intensity: Double = 1
  var cutThreshold: Double = 0.3
  var factor = 2
  var order: MediaEffectOrder = .renderingThenGeneration
  var slowMotion = false
  var includeAudio = true
  var codec: MediaVideoCodec = .h264
  var startFrame = 0
  var frameLimit = 0
  var previewTime: Double = 0 {
    didSet { selectedJob?.viewedTime = previewTime }
  }
  var preview: MediaPreviewResult?
  /// The requested time `preview` was rendered for; the decoder snaps it to a frame.
  private(set) var previewTimeRendered: Double?
  var previewError: String?
  var previewRefreshing = false
  /// The selected source at `previewTime`, decoded directly; replaced by `preview` once it is current.
  var sourceFrame: SourceFrame?
  @ObservationIgnored private var previewSession: NativeMediaPreview?
  @ObservationIgnored private var previewWorker: Task<Void, Never>?
  @ObservationIgnored private var pendingPreview: (revision: UInt64, request: MediaPreviewRequest)?
  @ObservationIgnored private var previewRevision: UInt64 = 0
  @ObservationIgnored private var scheduledRequest: MediaPreviewRequest?
  @ObservationIgnored private var renderedRequest: MediaPreviewRequest?
  @ObservationIgnored private var sourceWorker: Task<Void, Never>?
  @ObservationIgnored private var pendingSource: (job: MediaJob, time: Double)?
  @ObservationIgnored private var worker: Task<Void, Never>?
  @ObservationIgnored private let processor = NativeMediaProcessor()
  @ObservationIgnored private let scrubber = MediaLibrary.Scrubber()

  /// Set once the user picks a rendering package or network; until then the
  /// small network wins wherever its package sits beside the saved one.
  private static let modelChosenKey = "renderingModelChosen"

  init() {
    let defaults = UserDefaults.standard
    // The small network is the shipping default of nvngx_dlssnr.dll 310.9.16.
    modelPath = defaults.string(forKey: "renderingModel")
      ?? ["NeuralRendering-Small.dlssmodel", "NeuralRendering.dlssmodel"].lazy.map(Self.locateWeight).first { !$0.isEmpty }
      ?? ""
    generationPath = defaults.string(forKey: "generationWeights") ?? Self.locateWeight("framegen.safetensors")
    superResolutionPath = defaults.string(forKey: "superResolutionWeights") ?? Self.locateWeight("vsr.safetensors")
    dlssSuperResolutionPath = defaults.string(forKey: "dlssSuperResolutionModel") ?? Self.locateWeight("dlss-sr.srmodel")
    outputDirectory = defaults.string(forKey: "outputDirectory") ?? FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent("MLX-DLSS/outputs").path
    if !defaults.bool(forKey: Self.modelChosenKey), renderingNetwork == .ht,
      let small = networkPath(.small), FileManager.default.fileExists(atPath: small)
    {
      modelPath = small
    }
    observePreviewInputs()
  }

  var selectedJob: MediaJob? { jobs.first { $0.id == selection } }
  var hasQueuedJobs: Bool { jobs.contains { $0.state == .queued } }
  var queuedCount: Int { jobs.count { $0.state == .queued } }
  var hasFinishedJobs: Bool { jobs.contains(where: \.isFinished) }
  var usesDLSS: Bool { selectedJob?.isVideo != false && videoUpscaling == "dlss" }
  var upscalingPath: String { usesDLSS ? dlssSuperResolutionPath : superResolutionPath }

  var previewRequest: MediaPreviewRequest? {
    guard !isRunning, let job = selectedJob else { return nil }
    return MediaPreviewRequest(input: job.input, isVideo: job.isVideo, time: previewTime,
      options: options(forPreview: true, isVideo: job.isVideo))
  }

  /// True while the shown preview belongs to another frame than the one selected.
  var previewIsStale: Bool {
    guard preview != nil, let job = selectedJob else { return true }
    return job.isVideo && previewTimeRendered != previewTime
  }

  // MARK: Preview

  /// Re-arms itself on every change of a property the preview request reads, so
  /// slider drags never re-evaluate the views that only display the result.
  private func observePreviewInputs() {
    let request = withObservationTracking { previewRequest } onChange: { [weak self] in
      Task { @MainActor in self?.observePreviewInputs() }
    }
    if request != scheduledRequest { schedulePreview(request) }
    requestSourceFrame()
  }

  private func schedulePreview(_ request: MediaPreviewRequest?) {
    previewRevision &+= 1
    scheduledRequest = request
    guard let request else {
      pendingPreview = nil
      previewRefreshing = false
      return
    }
    pendingPreview = (previewRevision, request)
    previewRefreshing = true
    previewError = nil
    guard previewWorker == nil else { return }
    previewWorker = Task {
      defer { previewWorker = nil; previewRefreshing = false }
      while let peek = pendingPreview {
        // Live controls re-compose the retained frame in milliseconds, so they
        // follow a slider closely; anything that re-runs the network waits for
        // the input to settle. Changes made during an inference replace the
        // pending request; they never queue extra renders.
        let live = renderedRequest.map { Self.liveKey($0) == Self.liveKey(peek.request) } ?? false
        try? await Task.sleep(for: .milliseconds(live ? 16 : 120))
        guard let next = pendingPreview else { break }
        pendingPreview = nil
        do {
          if renderingEnabled && next.request.options.renderingModel == nil {
            throw MLXMediaError("Choose a neural-rendering model to enable live rendering")
          }
          if superResolutionEnabled && next.request.options.superResolutionWeights == nil && next.request.options.dlssSuperResolutionModel == nil {
            throw MLXMediaError("Choose \(next.request.isVideo && videoUpscaling == "dlss" ? "a DLSS SR model" : "VSR weights") to enable live upscaling")
          }
          if previewSession == nil { previewSession = try NativeMediaPreview() }
          let result = try await previewSession!.render(next.request)
          renderedRequest = next.request
          if next.revision == previewRevision {
            preview = result
            previewTimeRendered = next.request.time
            previewError = nil
          }
        } catch {
          renderedRequest = nil
          if next.revision == previewRevision { previewError = error.localizedDescription }
        }
      }
    }
  }

  /// The request with the controls that `NativeMediaPreview` re-applies without the network.
  private static func liveKey(_ request: MediaPreviewRequest) -> MediaPreviewRequest {
    var options = request.options
    options.intensity = 1
    options.detailStrength = 1
    options.colourStrength = 1
    options.detailRadius = 4
    return MediaPreviewRequest(input: request.input, isVideo: request.isVideo, time: request.time, options: options)
  }

  private func requestSourceFrame() {
    guard let job = selectedJob else { sourceFrame = nil; pendingSource = nil; return }
    let time = job.isVideo ? previewTime : 0
    if let sourceFrame, sourceFrame.job == job.id, sourceFrame.time == time { return }
    pendingSource = (job, time)
    guard sourceWorker == nil else { return }
    sourceWorker = Task {
      defer { sourceWorker = nil }
      while let next = pendingSource {
        pendingSource = nil
        let image = next.job.isVideo
          ? await scrubber.frame(next.job.input, time: next.time, maxPixels: 2048)
          : await MediaLibrary.frame(next.job.input, isVideo: false, maxPixels: 2048)
        if let image { sourceFrame = SourceFrame(job: next.job.id, time: next.time, image: image) }
      }
    }
  }

  func step(frames: Int) {
    guard let job = selectedJob, job.isVideo, let preview else { return }
    let last = max(0, preview.duration - preview.frameInterval)
    let frame = (previewTime / preview.frameInterval).rounded() + Double(frames)
    previewTime = min(last, max(0, frame * preview.frameInterval))
  }

  // MARK: Queue

  func addFiles(_ urls: [URL]) {
    for url in urls {
      guard let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .image) || type.conforms(to: .movie) else {
        alert = "Unsupported media file: \(url.lastPathComponent)"
        continue
      }
      if let queued = jobs.first(where: { $0.input == url && $0.state == .queued }) {
        selection = queued.id
        continue
      }
      let job = MediaJob(input: url)
      jobs.append(job)
      selection = job.id
      Task { await load(job) }
    }
  }

  private func load(_ job: MediaJob) async {
    let url = job.input, isVideo = job.isVideo
    let info = await MediaLibrary.info(url, isVideo: isVideo)
    job.info = info
    var poster = 0.0
    if let duration = info?.duration, duration > 0 {
      // Open on a frame with history behind it rather than on a fade-in.
      let interval = 1 / (info?.frameRate ?? 30)
      poster = ((duration * 0.15) / interval).rounded() * interval
      if (job.viewedTime ?? 0) == 0 {
        job.viewedTime = poster
        if selection == job.id, previewTime == 0 { previewTime = poster }
      }
    }
    job.thumbnail = await MediaLibrary.frame(url, isVideo: isVideo, time: poster, maxPixels: 192)
    if isVideo, let duration = info?.duration, duration > 0 {
      job.filmstrip = await MediaLibrary.filmstrip(url, duration: duration, count: 14, maxPixels: 192)
    }
  }

  func chooseFiles() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image, .movie]
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    if panel.runModal() == .OK { addFiles(panel.urls) }
  }

  func remove(_ job: MediaJob) {
    guard job.state != .running else { return }
    jobs.removeAll { $0.id == job.id }
    if selection == job.id { selection = jobs.last?.id }
  }

  func runQueue() {
    guard !isRunning, hasQueuedJobs else { return }
    let queued = jobs.filter { $0.state == .queued }
    var optionsByKind: [Bool: MediaProcessingOptions] = [:]
    do {
      for isVideo in Set(queued.map(\.isVideo)) {
        let options = try processingOptions(isVideo: isVideo)
        if !isVideo, options.renderingModel == nil, options.superResolutionWeights == nil {
          throw MLXMediaError("Enable neural rendering or super resolution to process images")
        }
        optionsByKind[isVideo] = options
      }
    } catch { alert = error.localizedDescription; return }
    savePaths()
    let directory = URL(fileURLWithPath: outputDirectory)
    isRunning = true
    worker = Task {
      defer { isRunning = false; worker = nil }
      // Let an already submitted preview finish before export claims the GPU.
      await previewWorker?.value
      for job in queued {
        if Task.isCancelled { break }
        job.state = .running
        job.error = nil
        job.progress = nil
        selection = job.id
        let options = optionsByKind[job.isVideo]!
        do {
          let folder = directory.appendingPathComponent("native-\(job.id.uuidString.lowercased())")
          try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
          let result: MediaProcessingResult
          if job.isVideo {
            let output = folder.appendingPathComponent(options.codec == .prores ? "result.mov" : "result.mp4")
            result = try await processor.processVideo(input: job.input, output: output, options: options) { [weak job] progress in
              await MainActor.run { job?.progress = progress }
            }
          } else {
            var imageOptions = options
            imageOptions.frameGenerationWeights = nil
            imageOptions.slowMotion = false
            result = try await processor.processImage(input: job.input,
              output: folder.appendingPathComponent("result.png"), options: imageOptions)
          }
          job.result = result
          job.state = .complete
        } catch is CancellationError {
          job.state = .cancelled
          break
        } catch {
          job.error = error.localizedDescription
          job.state = .failed
        }
      }
    }
  }

  func cancel() { worker?.cancel() }
  func retry(_ job: MediaJob) { job.state = .queued; job.error = nil; job.progress = nil; job.result = nil }
  func removeFinished() {
    jobs.removeAll(where: \.isFinished)
    if selectedJob == nil { selection = jobs.last?.id }
  }

  // MARK: Controls

  var adjustmentsAreNeutral: Bool {
    intensity == 1 && detailStrength == 1 && colourStrength == 1 && detailRadius == 4 && processingScale == 1
  }

  func resetAdjustments() {
    intensity = 1
    detailStrength = 1
    colourStrength = 1
    detailRadius = 4
    processingScale = 1
  }

  private func processingOptions(isVideo: Bool) throws -> MediaProcessingOptions {
    let options = options(forPreview: false, isVideo: isVideo)
    if renderingEnabled && options.renderingModel == nil { throw MLXMediaError("Choose a neural-rendering model") }
    if generationEnabled && options.frameGenerationWeights == nil { throw MLXMediaError("Choose frame-generation weights") }
    if superResolutionEnabled && options.superResolutionWeights == nil && options.dlssSuperResolutionModel == nil {
      throw MLXMediaError("Choose \(isVideo && videoUpscaling == "dlss" ? "a DLSS SR model" : "VSR weights")")
    }
    try options.validate()
    return options
  }

  private func options(forPreview: Bool, isVideo: Bool) -> MediaProcessingOptions {
    let dlss = isVideo && videoUpscaling == "dlss"
    var options = MediaProcessingOptions(
      renderingModel: renderingEnabled && !modelPath.isEmpty ? URL(fileURLWithPath: modelPath) : nil,
      frameGenerationWeights: !forPreview && generationEnabled && !generationPath.isEmpty ? URL(fileURLWithPath: generationPath) : nil,
      superResolutionWeights: superResolutionEnabled && !dlss && !superResolutionPath.isEmpty ? URL(fileURLWithPath: superResolutionPath) : nil,
      dlssSuperResolutionModel: superResolutionEnabled && dlss && !dlssSuperResolutionPath.isEmpty ? URL(fileURLWithPath: dlssSuperResolutionPath) : nil)
    options.profile = profile
    options.temporal = temporal
    options.motion = motion
    options.processingScale = Float(processingScale)
    options.detailStrength = Float(detailStrength)
    options.colourStrength = Float(colourStrength)
    options.detailRadius = Float(detailRadius)
    options.intensity = Float(intensity)
    options.sceneCutThreshold = Float(cutThreshold)
    if !forPreview {
      options.order = order
      options.frameGenerationFactor = factor
      options.slowMotion = slowMotion && generationEnabled
      options.includeAudio = includeAudio
      options.codec = codec
      options.startFrame = startFrame
      options.frameLimit = frameLimit > 0 ? frameLimit : nil
    }
    return options
  }

  // MARK: Models

  enum ModelKind: CaseIterable, Identifiable {
    case rendering, generation, dlssSuperResolution, superResolution
    var id: Self { self }

    var title: String {
      switch self {
      case .rendering: "Neural rendering"
      case .generation: "Frame generation"
      case .dlssSuperResolution: "DLSS Super Resolution"
      case .superResolution: "RTX Video Super Resolution"
      }
    }

    var fileKind: String {
      switch self {
      case .rendering: ".dlssmodel package"
      case .generation, .superResolution: ".safetensors weights"
      case .dlssSuperResolution: ".srmodel package"
      }
    }

    var isPackage: Bool { self == .rendering || self == .dlssSuperResolution }
  }

  func path(_ kind: ModelKind) -> String {
    switch kind {
    case .rendering: modelPath
    case .generation: generationPath
    case .dlssSuperResolution: dlssSuperResolutionPath
    case .superResolution: superResolutionPath
    }
  }

  func isAvailable(_ kind: ModelKind) -> Bool {
    let path = path(kind)
    return !path.isEmpty && FileManager.default.fileExists(atPath: path)
  }

  func choose(_ kind: ModelKind) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = kind.isPackage
    panel.canChooseFiles = true
    panel.treatsFilePackagesAsDirectories = false
    panel.message = "Choose the \(kind.title.lowercased()) \(kind.fileKind)"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    switch kind {
    case .rendering:
      modelPath = url.path
      UserDefaults.standard.set(true, forKey: Self.modelChosenKey)
    case .generation: generationPath = url.path
    case .dlssSuperResolution: dlssSuperResolutionPath = url.path
    case .superResolution: superResolutionPath = url.path
    }
    savePaths()
  }

  /// The upscaler the selected media uses: DLSS SR for video unless RTX VSR is chosen.
  var upscalingKind: ModelKind { usesDLSS ? .dlssSuperResolution : .superResolution }

  /// The two networks of `nvngx_dlssnr.dll` 310.9.16, which `mlxdlss-weights all`
  /// writes side by side as `NAME.dlssmodel` and `NAME-Small.dlssmodel`.
  enum RenderingNetwork: String, CaseIterable, Identifiable {
    case small = "Small", ht = "HT"
    var id: Self { self }
  }

  /// Networks whose package sits beside the chosen one.
  var renderingNetworks: [RenderingNetwork] {
    RenderingNetwork.allCases.filter { networkPath($0).map(FileManager.default.fileExists(atPath:)) ?? false }
  }

  var renderingNetwork: RenderingNetwork {
    get { URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent.hasSuffix("-Small") ? .small : .ht }
    set {
      guard let path = networkPath(newValue), path != modelPath else { return }
      modelPath = path
      UserDefaults.standard.set(true, forKey: Self.modelChosenKey)
      savePaths()
    }
  }

  private func networkPath(_ network: RenderingNetwork) -> String? {
    guard !modelPath.isEmpty else { return nil }
    let url = URL(fileURLWithPath: modelPath)
    var name = url.deletingPathExtension().lastPathComponent
    if name.hasSuffix("-Small") { name.removeLast("-Small".count) }
    let file = name + (network == .small ? "-Small" : "") + ".dlssmodel"
    return url.deletingLastPathComponent().appendingPathComponent(file).path
  }

  func chooseOutputDirectory() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.canCreateDirectories = true
    panel.directoryURL = URL(fileURLWithPath: outputDirectory)
    if panel.runModal() == .OK, let url = panel.url { outputDirectory = url.path; savePaths() }
  }

  private func savePaths() {
    UserDefaults.standard.set(modelPath, forKey: "renderingModel")
    UserDefaults.standard.set(generationPath, forKey: "generationWeights")
    UserDefaults.standard.set(superResolutionPath, forKey: "superResolutionWeights")
    UserDefaults.standard.set(dlssSuperResolutionPath, forKey: "dlssSuperResolutionModel")
    UserDefaults.standard.set(outputDirectory, forKey: "outputDirectory")
  }

  private static func locateWeight(_ name: String) -> String {
    var roots = [URL(fileURLWithPath: FileManager.default.currentDirectoryPath)]
    var ancestor = Bundle.main.bundleURL
    for _ in 0..<5 { ancestor.deleteLastPathComponent(); roots.append(ancestor) }
    return roots.lazy.map { $0.appendingPathComponent("weights/\(name)").path }
      .first { FileManager.default.fileExists(atPath: $0) } ?? ""
  }
}
