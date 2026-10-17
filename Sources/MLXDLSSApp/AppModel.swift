import AppKit
import DLSSCore
import DLSSMedia
import DLSSMLX
import Foundation
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class MediaJob: Identifiable {
  enum State: String { case queued = "Queued", running = "Processing", complete = "Complete", cancelled = "Cancelled", failed = "Failed" }
  let id = UUID()
  let input: URL
  let isVideo: Bool
  var state: State = .queued
  var progress: MediaProgress?
  var result: MediaProcessingResult?
  var error: String?

  init(input: URL) {
    self.input = input
    isVideo = UTType(filenameExtension: input.pathExtension)?.conforms(to: .movie) == true
  }
}

@available(macOS 26.0, *)
@MainActor @Observable
final class AppModel {
  var jobs: [MediaJob] = []
  var selection: UUID? {
    didSet { if selection != oldValue { previewTime = 0; preview = nil; previewError = nil } }
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
  var previewTime: Double = 0
  var preview: MediaPreviewResult?
  var previewError: String?
  var previewRefreshing = false
  private var previewSession: NativeMediaPreview?
  private var previewWorker: Task<Void, Never>?
  private var pendingPreview: (revision: UInt64, request: MediaPreviewRequest)?
  private var previewRevision: UInt64 = 0
  private var worker: Task<Void, Never>?
  private let processor = NativeMediaProcessor()

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
  }

  var selectedJob: MediaJob? { jobs.first { $0.id == selection } }
  var hasQueuedJobs: Bool { jobs.contains { $0.state == .queued } }
  var usesDLSS: Bool { selectedJob?.isVideo != false && videoUpscaling == "dlss" }
  var upscalingPath: String { usesDLSS ? dlssSuperResolutionPath : superResolutionPath }

  var previewRequest: MediaPreviewRequest? {
    guard !isRunning, let job = selectedJob else { return nil }
    return MediaPreviewRequest(input: job.input, isVideo: job.isVideo, time: previewTime,
      options: options(forPreview: true, isVideo: job.isVideo))
  }

  func schedulePreview(_ request: MediaPreviewRequest?) {
    previewRevision &+= 1
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
      while pendingPreview != nil {
        // Read the latest snapshot after the debounce. Changes made during an
        // inference replace the pending request; they never queue extra renders.
        try? await Task.sleep(for: .milliseconds(180))
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
          if next.revision == previewRevision { preview = result; previewError = nil }
        } catch {
          if next.revision == previewRevision { previewError = error.localizedDescription }
        }
      }
    }
  }

  func addFiles(_ urls: [URL]) {
    for url in urls {
      guard let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .image) || type.conforms(to: .movie) else {
        alert = "Unsupported media file: \(url.lastPathComponent)"
        continue
      }
      let job = MediaJob(input: url)
      jobs.append(job)
      selection = job.id
    }
  }

  func chooseFiles() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image, .movie]
    panel.allowsMultipleSelection = true
    panel.canChooseDirectories = false
    if panel.runModal() == .OK { addFiles(panel.urls) }
  }

  func chooseModel(generation: Bool) {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = !generation
    panel.canChooseFiles = true
    panel.message = generation ? "Choose frame-generation .safetensors weights" : "Choose a neural-rendering .dlssmodel package"
    if panel.runModal() == .OK, let url = panel.url {
      if generation { generationPath = url.path }
      else { modelPath = url.path }
      savePaths()
    }
  }

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
    if panel.runModal() == .OK, let url = panel.url { outputDirectory = url.path; savePaths() }
  }

  func chooseSuperResolutionWeights() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = usesDLSS
    panel.canChooseFiles = !usesDLSS
    panel.message = usesDLSS ? "Choose a DLSS SR .srmodel package" : "Choose VSR 2× .safetensors weights"
    if panel.runModal() == .OK, let url = panel.url {
      if usesDLSS { dlssSuperResolutionPath = url.path }
      else { superResolutionPath = url.path }
      savePaths()
    }
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
    schedulePreview(nil)
    worker = Task {
      defer { isRunning = false; worker = nil }
      // Let an already submitted preview finish before export claims the GPU.
      await previewWorker?.value
      for job in queued {
        if Task.isCancelled { break }
        job.state = .running
        job.error = nil
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
  func retry(_ job: MediaJob) { job.state = .queued; job.error = nil; job.progress = nil }
  func removeFinished() { jobs.removeAll { $0.state != .running && $0.state != .queued } }

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
