import AVKit
import DLSSCore
import DLSSMedia
import SwiftUI

@available(macOS 26.0, *)
struct MainView: View {
  @Bindable var model: AppModel
  private enum PreviewMode { case original, live, export }
  @State private var previewMode: PreviewMode = .live

  var body: some View {
    NavigationSplitView {
      List(selection: $model.selection) {
        ForEach(model.jobs) { job in
          VStack(alignment: .leading, spacing: 6) {
            Label(job.input.lastPathComponent, systemImage: job.isVideo ? "film" : "photo")
              .lineLimit(1).font(.body.weight(.medium))
            HStack {
              Text(job.state.rawValue)
              Spacer()
              if let result = job.result { Text(result.elapsedSeconds, format: .number.precision(.fractionLength(1))) + Text(" s") }
            }.font(.caption).foregroundStyle(job.state == .failed ? .red : .secondary)
            if job.state == .running {
              ProgressView(value: job.progress?.fraction ?? 0)
            }
          }.padding(.vertical, 4).tag(job.id)
            .contextMenu {
              Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([job.input]) }
              if job.state == .failed || job.state == .cancelled {
                Button("Queue Again") { model.retry(job) }
              }
            }
        }
      }
      .overlay {
        if model.jobs.isEmpty {
          ContentUnavailableView("Your media queue", systemImage: "photo.on.rectangle.angled",
            description: Text("Import images or videos to get started."))
        }
      }
      .navigationTitle("Queue")
      .navigationSplitViewColumnWidth(min: 240, ideal: 280)
      .safeAreaInset(edge: .bottom) {
        Button("Clear Finished") { model.removeFinished() }.buttonStyle(.borderless).padding(12)
      }
    } detail: {
      VStack(spacing: 0) {
        if let job = model.selectedJob {
          HStack {
            Text(job.input.lastPathComponent).font(.headline).lineLimit(1)
            Spacer()
            Picker("Preview", selection: $previewMode) {
              Text("Original").tag(PreviewMode.original)
              Text("Live").tag(PreviewMode.live)
              if job.result != nil { Text("Export").tag(PreviewMode.export) }
            }.pickerStyle(.segmented).labelsHidden().frame(width: job.result == nil ? 170 : 240)
          }.padding()
          if previewMode == .export, let result = job.result {
            MediaPreview(url: result.output, video: job.isVideo)
              .frame(maxWidth: .infinity, maxHeight: .infinity)
          } else {
            LiveFramePreview(model: model, showOriginal: previewMode == .original, video: job.isVideo)
          }
          HStack {
            if let error = job.error {
              Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red).textSelection(.enabled)
            } else if let progress = job.progress, job.state == .running {
              Text("\(progress.inputFrames) input → \(progress.outputFrames) output frames")
              Spacer()
              Text("\(progress.elapsedSeconds, specifier: "%.1f") s · \(progress.sceneResets) scene resets")
                .foregroundStyle(.secondary)
            } else if let result = job.result {
              Text("\(result.outputFrames) \(job.isVideo ? "frames" : "image") · \(result.elapsedSeconds, specifier: "%.2f") s")
              Spacer()
              Button("Show Result in Finder") { NSWorkspace.shared.activateFileViewerSelecting([result.output]) }
            } else { Text(job.state.rawValue).foregroundStyle(.secondary); Spacer() }
          }.font(.callout).padding().frame(minHeight: 52)
        } else {
          ContentUnavailableView {
            Label("Neural rendering on your Mac", systemImage: "sparkles.rectangle.stack")
          } description: {
            Text("Process images and video with DLSS neural rendering and frame generation.")
          } actions: {
            Button("Import Media…") { model.chooseFiles() }.buttonStyle(.borderedProminent)
          }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
      }
      .background(.background)
    }
    .inspector(isPresented: .constant(true)) {
      ProcessingControls(model: model).inspectorColumnWidth(min: 285, ideal: 310, max: 370)
    }
    .toolbar {
      ToolbarItem(placement: .primaryAction) {
        Button("Import", systemImage: "plus") { model.chooseFiles() }.help("Import images or videos")
      }
      ToolbarItem(placement: .primaryAction) {
        if model.isRunning {
          Button("Stop", systemImage: "stop.fill", role: .destructive) { model.cancel() }
        } else {
          Button("Start Queue", systemImage: "play.fill") { model.runQueue() }
            .disabled(!model.hasQueuedJobs).keyboardShortcut(.return, modifiers: .command)
        }
      }
    }
    .dropDestination(for: URL.self) { urls, _ in model.addFiles(urls); return true }
    .onChange(of: model.selection) { _, _ in previewMode = .live }
    .onChange(of: model.previewRequest, initial: true) { _, request in model.schedulePreview(request) }
    .alert("Cannot Process Media", isPresented: Binding(get: { model.alert != nil }, set: { if !$0 { model.alert = nil } })) {
      Button("OK") { model.alert = nil }
    } message: { Text(model.alert ?? "") }
  }
}

@available(macOS 26.0, *)
private struct LiveFramePreview: View {
  @Bindable var model: AppModel
  let showOriginal: Bool
  let video: Bool

  var body: some View {
    VStack(spacing: 0) {
      Group {
        if let preview = model.preview {
          Image(decorative: showOriginal ? preview.original : preview.processed, scale: 1)
            .resizable().scaledToFit().padding(12)
            .accessibilityLabel(showOriginal ? "Original frame" : "Live processed frame")
        } else if let error = model.previewError {
          ContentUnavailableView("Preview unavailable", systemImage: "photo", description: Text(error))
        } else {
          ProgressView("Preparing preview…")
        }
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .background(.black.opacity(0.08))
      .overlay(alignment: .topTrailing) {
        if model.previewRefreshing, model.preview != nil {
          HStack(spacing: 8) { ProgressView().controlSize(.small); Text("Updating…") }
            .font(.caption).padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8)).padding(12)
        }
      }
      if let preview = model.preview {
        VStack(alignment: .leading, spacing: 10) {
          if video {
            HStack {
              Button { model.previewTime = max(0, preview.time - preview.frameInterval) } label: {
                Image(systemName: "backward.frame")
              }.help("Previous frame").accessibilityLabel("Previous frame")
              Slider(value: $model.previewTime, in: 0...max(0.001, preview.duration - preview.frameInterval))
                .accessibilityLabel("Preview time")
              Button { model.previewTime = min(max(0, preview.duration - preview.frameInterval), preview.time + preview.frameInterval) } label: {
                Image(systemName: "forward.frame")
              }.help("Next frame").accessibilityLabel("Next frame")
              Text("\(model.previewRefreshing ? model.previewTime : preview.time, specifier: "%.3f") / \(preview.duration, specifier: "%.2f") s")
                .monospacedDigit().frame(minWidth: 125, alignment: .trailing)
            }.disabled(model.isRunning)
          }
          HStack {
            Text("\(showOriginal ? preview.original.width : preview.processed.width) × \(showOriginal ? preview.original.height : preview.processed.height)")
            if video {
              Text("·")
              Text(preview.historyFrames > 0 ? "Temporal preview · \(preview.historyFrames) preceding frames" : "Single-frame preview")
                .help("Preview starts a fresh history from up to three preceding frames. Full export uses the complete temporal sequence. Frame generation is evaluated during export.")
            }
            Spacer()
            Text("\(preview.elapsedSeconds, specifier: "%.2f") s")
          }.foregroundStyle(.secondary)
          if let error = model.previewError {
            Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red).textSelection(.enabled)
          }
        }.font(.caption).padding(.horizontal).padding(.vertical, 10)
      }
    }
  }
}

private struct MediaPreview: View {
  let url: URL
  let video: Bool
  @State private var player = AVPlayer()

  var body: some View {
    Group {
      if video { VideoPlayer(player: player) }
      else if let image = NSImage(contentsOf: url) { Image(nsImage: image).resizable().scaledToFit().padding(12) }
      else { ContentUnavailableView("Preview unavailable", systemImage: "photo") }
    }
    .background(.black.opacity(0.08))
    .task(id: url) {
      player.pause()
      player.replaceCurrentItem(with: video ? AVPlayerItem(url: url) : nil)
    }
    .onDisappear { player.pause() }
  }
}

@available(macOS 26.0, *)
private struct ProcessingControls: View {
  @Bindable var model: AppModel

  var body: some View {
    Form {
      Section("Neural Rendering") {
        Toggle("Enable rendering", isOn: $model.renderingEnabled)
        Button(model.modelPath.isEmpty ? "Choose Model…" : URL(fileURLWithPath: model.modelPath).lastPathComponent) {
          model.chooseModel(generation: false)
        }.help(model.modelPath).lineLimit(1)
        if model.renderingNetworks.count > 1 {
          Picker("Network", selection: $model.renderingNetwork) {
            ForEach(model.renderingNetworks) { Text($0.rawValue).tag($0) }
          }
          .pickerStyle(.segmented)
          .help("Small: 31 blocks, NVIDIA's 310.9.16 default and the faster one. HT: 71 blocks.")
        }
        Group {
          Picker("Profile", selection: $model.profile) {
            ForEach(NeuralRenderingControlProfile.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
          }
          valueSlider("Processing scale", value: $model.processingScale, range: 1...4)
          valueSlider("Intensity", value: $model.intensity, range: 0...2)
          valueSlider("Detail", value: $model.detailStrength, range: 0...8)
          valueSlider("Colour", value: $model.colourStrength, range: 0...4)
          DisclosureGroup("Detail radius") { valueSlider("Radius", value: $model.detailRadius, range: 0.5...64) }
        }.disabled(!model.renderingEnabled)
      }
      Section("Video History") {
        Toggle("Temporal rendering", isOn: $model.temporal)
        Picker("Motion", selection: $model.motion) {
          Text("Automatic").tag(MediaMotion.automatic)
          Text("VideoToolbox").tag(MediaMotion.videoToolbox)
          Text("Vision").tag(MediaMotion.vision)
          Text("Zero (diagnostic)").tag(MediaMotion.zero)
        }
        valueSlider("Scene cut threshold", value: $model.cutThreshold, range: 0...1)
      }.disabled((!model.renderingEnabled && !(model.superResolutionEnabled && model.usesDLSS)) || model.selectedJob?.isVideo == false)
      Section("Frame Generation") {
        Toggle("Generate frames", isOn: $model.generationEnabled)
        Button(model.generationPath.isEmpty ? "Choose Weights…" : URL(fileURLWithPath: model.generationPath).lastPathComponent) {
          model.chooseModel(generation: true)
        }.help(model.generationPath).lineLimit(1)
        Picker("Multiplier", selection: $model.factor) {
          ForEach([2, 3, 4, 8, 16], id: \.self) { Text("\($0)×").tag($0) }
        }.disabled(!model.generationEnabled)
        Toggle("Slow motion", isOn: $model.slowMotion).disabled(!model.generationEnabled)
        Picker("Order", selection: $model.order) {
          Text("Render → Generate").tag(MediaEffectOrder.renderingThenGeneration)
          Text("Generate → Render").tag(MediaEffectOrder.generationThenRendering)
        }.disabled(!model.generationEnabled || !model.renderingEnabled)
      }.disabled(model.selectedJob?.isVideo == false)
      Section("Super Resolution · Experimental") {
        Toggle("Upscale 2×", isOn: $model.superResolutionEnabled)
        if model.selectedJob?.isVideo != false {
          Picker("Video upscaler", selection: $model.videoUpscaling) {
            Text("DLSS SR · Temporal").tag("dlss")
            Text("RTX VSR").tag("vsr")
          }
        }
        Button(model.upscalingPath.isEmpty ? "Choose \(model.usesDLSS ? "DLSS SR Model" : "VSR Weights")…" : URL(fileURLWithPath: model.upscalingPath).lastPathComponent) {
          model.chooseSuperResolutionWeights()
        }.help(model.upscalingPath).lineLimit(1)
        Text(model.usesDLSS ? "DLSS SR uses motion and preceding frames." : "RTX VSR · High Bitrate Low.")
          .font(.caption).foregroundStyle(.secondary)
      }
      Section("Output") {
        Picker("Video codec", selection: $model.codec) {
          Text("H.264").tag(MediaVideoCodec.h264)
          Text("HEVC").tag(MediaVideoCodec.hevc)
          Text("ProRes 422 HQ").tag(MediaVideoCodec.prores)
        }
        Toggle("Include audio", isOn: $model.includeAudio)
        DisclosureGroup("Video range") {
          TextField("Start frame", value: $model.startFrame, format: .number)
          TextField("Frame limit (0 = all)", value: $model.frameLimit, format: .number)
        }
        Button("Output Folder…") { model.chooseOutputDirectory() }.help(model.outputDirectory)
        Text(model.outputDirectory).font(.caption).foregroundStyle(.secondary).lineLimit(2).textSelection(.enabled)
      }
    }
    .formStyle(.grouped)
    .disabled(model.isRunning)
  }

  private func valueSlider(_ title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack { Text(title); Spacer(); Text(value.wrappedValue, format: .number.precision(.fractionLength(2))).monospacedDigit().foregroundStyle(.secondary) }
      Slider(value: value, in: range)
    }
  }
}
