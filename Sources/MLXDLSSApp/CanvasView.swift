import AVKit
import AppKit
import DLSSMedia
import SwiftUI

/// The frame fills the window; every control floats over it as Liquid Glass.
@available(macOS 26.0, *)
struct CanvasView: View {
  @Bindable var model: AppModel
  @Bindable var canvas: CanvasState

  var body: some View {
    ZStack {
      // Runs under the toolbar, so the glass items float over the canvas.
      Color(nsColor: .underPageBackgroundColor).ignoresSafeArea()
      if let job = model.selectedJob {
        JobCanvas(model: model, canvas: canvas, job: job)
      } else {
        EmptyCanvas(model: model)
      }
    }
  }
}

// MARK: - Selected item

@available(macOS 26.0, *)
private struct JobCanvas: View {
  @Bindable var model: AppModel
  @Bindable var canvas: CanvasState
  let job: MediaJob
  @FocusState private var focused: Bool
  @State private var hovering = false
  @State private var scrollMonitor: Any?

  /// Room kept free under the frame for the floating bar.
  private static let barSpace: CGFloat = 84

  var body: some View {
    stage
      .overlay(alignment: .top) {
        StatusCapsule(model: model, canvas: canvas, job: job).padding(.top, 14).padding(.horizontal, 16)
      }
      .overlay(alignment: .bottom) {
        if !canvas.showsResult {
          BottomBar(model: model, canvas: canvas, job: job).padding(16)
        }
      }
      .focusable()
      .focusEffectDisabled()
      .focused($focused)
      .simultaneousGesture(TapGesture().onEnded { focused = true })
      .onKeyPress(.leftArrow) { model.step(frames: -1); return .handled }
      .onKeyPress(.rightArrow) { model.step(frames: 1); return .handled }
      .onKeyPress(KeyEquivalent("m"), phases: [.down, .repeat, .up]) { press in
        canvas.peeking = press.phase != .up
        return .handled
      }
      .onContinuousHover { phase in
        if case .active = phase { hovering = true } else { hovering = false }
      }
      .onAppear { installScrollMonitor() }
      .onDisappear {
        if let scrollMonitor { NSEvent.removeMonitor(scrollMonitor) }
        scrollMonitor = nil
      }
  }

  @ViewBuilder private var stage: some View {
    if canvas.showsResult, let result = job.result {
      ResultView(url: result.output, video: job.isVideo,
        aspect: job.info.map { CGSize(width: $0.width, height: $0.height) }).padding(16)
    } else if let frame = displayedFrame {
      CompareStage(canvas: canvas, frame: frame, bottomInset: Self.barSpace)
    } else if let error = model.previewError {
      ContentUnavailableView("Preview Unavailable", systemImage: "exclamationmark.triangle", description: Text(error))
    } else {
      ProgressView().controlSize(.large)
    }
  }

  private var displayedFrame: StageFrame? {
    let sourceSize = job.info.map { CGSize(width: $0.width, height: $0.height) }
    if let preview = model.preview, !model.previewIsStale {
      return StageFrame(original: preview.original, processed: preview.processed,
        pixelSize: CGSize(width: preview.processed.width, height: preview.processed.height))
    }
    if let source = model.sourceFrame, source.job == job.id {
      return StageFrame(original: source.image, processed: nil,
        pixelSize: sourceSize ?? CGSize(width: source.image.width, height: source.image.height))
    }
    if let preview = model.preview {
      return StageFrame(original: preview.original, processed: preview.processed,
        pixelSize: CGSize(width: preview.processed.width, height: preview.processed.height))
    }
    return nil
  }

  /// Two-finger scrolling pans a zoomed frame; ⌘-scrolling zooms.
  private func installScrollMonitor() {
    guard scrollMonitor == nil else { return }
    scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
      let precise = event.hasPreciseScrollingDeltas, zooms = event.modifierFlags.contains(.command)
      let delta = CGSize(width: event.scrollingDeltaX * (precise ? 1 : 10), height: event.scrollingDeltaY * (precise ? 1 : 10))
      let consumed = MainActor.assumeIsolated { () -> Bool in
        guard hovering, !canvas.showsResult else { return false }
        if zooms {
          canvas.setZoom(canvas.zoom * exp(delta.height * 0.01))
          return true
        }
        guard canvas.canPan else { return false }
        canvas.move(by: delta)
        return true
      }
      return consumed ? nil : event
    }
  }
}

struct StageFrame {
  let original: CGImage
  /// Nil while only the decoded source is available.
  let processed: CGImage?
  /// Output pixels; zoom percentages refer to these.
  let pixelSize: CGSize
}

// MARK: - Compare stage

@available(macOS 26.0, *)
private struct CompareStage: View {
  @Bindable var canvas: CanvasState
  let frame: StageFrame
  let bottomInset: CGFloat

  var body: some View {
    let mode: CanvasState.Compare = frame.processed == nil ? .original : canvas.mode
    Group {
      if mode == .sideBySide, let processed = frame.processed {
        HStack(spacing: 12) {
          FrameViewport(canvas: canvas, pixelSize: frame.pixelSize, reportsLayout: true,
            content: .single(frame.original, label: "Original"))
          FrameViewport(canvas: canvas, pixelSize: frame.pixelSize, reportsLayout: false,
            content: .single(processed, label: "Rendered"))
        }
      } else if mode == .split, let processed = frame.processed {
        FrameViewport(canvas: canvas, pixelSize: frame.pixelSize, reportsLayout: true,
          content: .split(frame.original, processed))
      } else {
        let showsProcessed = mode == .rendered && frame.processed != nil
        FrameViewport(canvas: canvas, pixelSize: frame.pixelSize, reportsLayout: true,
          content: .single(showsProcessed ? frame.processed! : frame.original,
            label: frame.processed == nil ? nil : showsProcessed ? "Rendered" : "Original"))
      }
    }
    .padding(EdgeInsets(top: 16, leading: 16, bottom: bottomInset, trailing: 16))
  }
}

@available(macOS 26.0, *)
private struct FrameViewport: View {
  enum Content {
    case single(CGImage, label: String?)
    case split(CGImage, CGImage)
  }

  @Bindable var canvas: CanvasState
  let pixelSize: CGSize
  /// One viewport per stage reports the fitted size the zoom refers to.
  let reportsLayout: Bool
  let content: Content
  @Environment(\.displayScale) private var displayScale
  @State private var dragStart: CGSize?
  @State private var magnifyStart: (zoom: Double, pan: CGSize)?

  private struct Layout: Equatable {
    var viewport: CGSize
    var fitted: CGSize
    var fitScale: Double
    var displayScale: Double
  }

  var body: some View {
    GeometryReader { geometry in
      let viewport = geometry.size
      let fit = max(1e-6, min(viewport.width / pixelSize.width, viewport.height / pixelSize.height))
      let fitted = CGSize(width: pixelSize.width * fit, height: pixelSize.height * fit)
      let size = CGSize(width: fitted.width * canvas.zoom, height: fitted.height * canvas.zoom)
      let pan = clamp(canvas.pan, size: size, viewport: viewport)
      let rect = CGRect(x: (viewport.width - size.width) / 2 + pan.width, y: (viewport.height - size.height) / 2 + pan.height,
        width: size.width, height: size.height)
      let visible = rect.intersection(CGRect(origin: .zero, size: viewport))
      // Show discrete pixels once one covers two display pixels.
      let pixelated = fit * canvas.zoom * displayScale >= 2
      ZStack(alignment: .topLeading) {
        switch content {
        case .single(let image, let label):
          picture(image, rect: rect, pixelated: pixelated)
          labels(leading: label, trailing: nil, in: visible)
        case .split(let original, let processed):
          let line = visible.minX + canvas.divider * visible.width
          picture(original, rect: rect, pixelated: pixelated)
          picture(processed, rect: rect, pixelated: pixelated)
            .mask(alignment: .topLeading) {
              Rectangle().frame(width: max(0, viewport.width - line)).offset(x: line)
            }
          labels(leading: line - visible.minX > 120 ? "Original" : nil,
            trailing: visible.maxX - line > 120 ? "Rendered" : nil, in: visible)
          SplitHandle(height: visible.height)
            .position(x: line, y: visible.midY)
            .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("viewport")).onChanged { value in
              moveDivider(to: value.location.x, visible: visible)
            })
        }
      }
      .frame(width: viewport.width, height: viewport.height)
      .clipped()
      .contentShape(Rectangle())
      .coordinateSpace(.named("viewport"))
      .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .named("viewport"))
        .onChanged { value in
          if case .split = content, !canvas.canPan {
            moveDivider(to: value.location.x, visible: visible)
            return
          }
          if dragStart == nil { dragStart = canvas.pan }
          canvas.pan = canvas.clamped(CGSize(width: dragStart!.width + value.translation.width,
            height: dragStart!.height + value.translation.height))
        }
        .onEnded { _ in dragStart = nil })
      .simultaneousGesture(MagnifyGesture()
        .onChanged { value in
          if magnifyStart == nil { magnifyStart = (canvas.zoom, canvas.pan) }
          canvas.zoom = magnifyStart!.zoom
          canvas.pan = magnifyStart!.pan
          canvas.setZoom(magnifyStart!.zoom * value.magnification, around: anchor(value.startLocation, viewport))
        }
        .onEnded { _ in magnifyStart = nil })
      .onTapGesture(count: 2, coordinateSpace: .named("viewport")) { location in
        withAnimation(.smooth(duration: 0.3)) { canvas.toggleZoom(around: anchor(location, viewport)) }
      }
      .onChange(of: Layout(viewport: viewport, fitted: fitted, fitScale: fit, displayScale: displayScale), initial: true) { _, layout in
        guard reportsLayout else { return }
        canvas.viewport = layout.viewport
        canvas.fitted = layout.fitted
        canvas.fitScale = layout.fitScale
        canvas.displayScale = layout.displayScale
        canvas.pan = canvas.clamped(canvas.pan)
      }
    }
  }

  private func picture(_ image: CGImage, rect: CGRect, pixelated: Bool) -> some View {
    Image(decorative: image, scale: 1)
      .resizable()
      .interpolation(pixelated ? .none : .high)
      .frame(width: rect.width, height: rect.height)
      .position(x: rect.midX, y: rect.midY)
      .accessibilityHidden(true)
  }

  /// Names the halves in the top corners of the visible frame.
  @ViewBuilder private func labels(leading: String?, trailing: String?, in visible: CGRect) -> some View {
    if !visible.isEmpty {
      HStack {
        if let leading { FrameLabel(text: leading) }
        Spacer(minLength: 0)
        if let trailing { FrameLabel(text: trailing) }
      }
      .padding(12)
      .frame(width: visible.width, height: visible.height, alignment: .top)
      .position(x: visible.midX, y: visible.midY)
      .allowsHitTesting(false)
    }
  }

  private func clamp(_ offset: CGSize, size: CGSize, viewport: CGSize) -> CGSize {
    let x = max(0, (size.width - viewport.width) / 2), y = max(0, (size.height - viewport.height) / 2)
    return CGSize(width: min(x, max(-x, offset.width)), height: min(y, max(-y, offset.height)))
  }

  private func anchor(_ location: CGPoint, _ viewport: CGSize) -> CGPoint {
    CGPoint(x: location.x - viewport.width / 2, y: location.y - viewport.height / 2)
  }

  private func moveDivider(to x: CGFloat, visible: CGRect) {
    guard visible.width > 0 else { return }
    canvas.divider = min(1, max(0, (x - visible.minX) / visible.width))
  }
}

@available(macOS 26.0, *)
private struct FrameLabel: View {
  let text: String
  var body: some View {
    Text(text)
      .font(.subheadline.weight(.medium))
      .padding(.horizontal, 10)
      .padding(.vertical, 5)
      .glassEffect(.regular, in: .capsule)
      .fixedSize()
      .allowsHitTesting(false)
  }
}

/// The boundary between the recorded and the rendered frame, with a glass grip.
@available(macOS 26.0, *)
private struct SplitHandle: View {
  let height: CGFloat
  var body: some View {
    ZStack {
      Rectangle()
        .fill(.white.opacity(0.9))
        .frame(width: 2, height: height)
        .shadow(color: .black.opacity(0.35), radius: 1.5)
      Image(systemName: "arrow.left.and.right")
        .font(.system(size: 12, weight: .bold))
        .frame(width: 34, height: 34)
        .glassEffect(.regular.interactive(), in: .circle)
    }
    .frame(width: 34, height: height)
    .contentShape(Rectangle())
    .pointerStyle(.columnResize)
    .accessibilityElement()
    .accessibilityLabel("Split position")
  }
}

// MARK: - Floating bars

@available(macOS 26.0, *)
private struct BottomBar: View {
  @Bindable var model: AppModel
  @Bindable var canvas: CanvasState
  let job: MediaJob
  @Namespace private var glass

  var body: some View {
    GlassEffectContainer(spacing: 10) {
      HStack(spacing: 10) {
        if job.state == .running {
          ProgressStrip(model: model, job: job)
            .frame(maxWidth: 640)
            .glassEffect(.regular, in: .capsule)
            .glassEffectID("centre", in: glass)
        } else if job.isVideo {
          TransportStrip(model: model, job: job)
            .frame(maxWidth: 720)
            .glassEffect(.regular, in: .capsule)
            .glassEffectID("centre", in: glass)
        }
        ZoomCluster(canvas: canvas).glassEffectID("zoom", in: glass)
      }
    }
    .frame(maxWidth: .infinity)
    .frame(height: 52)
    .animation(.smooth(duration: 0.35), value: job.state)
  }
}

@available(macOS 26.0, *)
private struct TransportStrip: View {
  @Bindable var model: AppModel
  let job: MediaJob

  private var interval: Double {
    model.preview?.frameInterval ?? job.info?.frameRate.map { 1 / $0 } ?? 1 / 30
  }
  private var duration: Double { model.preview?.duration ?? job.info?.duration ?? 0 }
  private var lastTime: Double { max(0, duration - interval) }

  var body: some View {
    HStack(spacing: 10) {
      Button { model.step(frames: -1) } label: {
        Image(systemName: "backward.frame.fill").frame(width: 22, height: 22)
      }
      .help("Previous frame (←)")
      .accessibilityLabel("Previous frame")
      Filmstrip(frames: job.filmstrip, fraction: lastTime > 0 ? model.previewTime / lastTime : 0) { fraction in
        model.previewTime = (fraction * lastTime / interval).rounded() * interval
      }
      .frame(minWidth: 96)
      .frame(height: 30)
      Button { model.step(frames: 1) } label: {
        Image(systemName: "forward.frame.fill").frame(width: 22, height: 22)
      }
      .help("Next frame (→)")
      .accessibilityLabel("Next frame")
      VStack(alignment: .trailing, spacing: 0) {
        Text(Self.timecode(model.previewTime)).font(.callout.weight(.medium))
        Text("Frame \(Int((model.previewTime / interval).rounded()) + 1) of \(max(1, Int((duration / interval).rounded())))")
          .font(.caption).foregroundStyle(.secondary)
      }
      .monospacedDigit()
      .fixedSize()
    }
    .buttonStyle(.borderless)
    .padding(.horizontal, 14)
    .frame(height: 52)
    .disabled(model.isRunning)
  }

  static func timecode(_ seconds: Double) -> String {
    let minutes = Int(seconds) / 60
    return String(format: "%d:%05.2f", minutes, seconds - Double(minutes * 60))
  }
}

@available(macOS 26.0, *)
private struct Filmstrip: View {
  let frames: [CGImage]
  let fraction: Double
  let seek: (Double) -> Void

  var body: some View {
    GeometryReader { geometry in
      let width = geometry.size.width
      ZStack(alignment: .leading) {
        HStack(spacing: 0) {
          if frames.isEmpty {
            Rectangle().fill(.quaternary)
          } else {
            ForEach(frames.indices, id: \.self) { index in
              Image(decorative: frames[index], scale: 1)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: width / CGFloat(frames.count), height: geometry.size.height)
                .clipped()
            }
          }
        }
        .clipShape(.rect(cornerRadius: 7))
        RoundedRectangle(cornerRadius: 2)
          .fill(.white)
          .frame(width: 4, height: geometry.size.height + 10)
          .shadow(color: .black.opacity(0.4), radius: 1.5)
          .offset(x: min(width - 4, max(0, fraction * width - 2)))
      }
      .frame(maxHeight: .infinity)
      .contentShape(Rectangle())
      .gesture(DragGesture(minimumDistance: 0).onChanged { value in
        seek(min(1, max(0, value.location.x / max(1, width))))
      })
    }
    .accessibilityElement()
    .accessibilityLabel("Timeline")
    .accessibilityValue("\(Int(fraction * 100)) percent")
    .accessibilityAdjustableAction { direction in
      seek(min(1, max(0, fraction + (direction == .increment ? 0.05 : -0.05))))
    }
  }
}

@available(macOS 26.0, *)
private struct ProgressStrip: View {
  @Bindable var model: AppModel
  let job: MediaJob

  var body: some View {
    HStack(spacing: 12) {
      if let progress = job.progress, job.isVideo {
        ProgressView(value: progress.fraction).frame(minWidth: 120)
        VStack(alignment: .leading, spacing: 0) {
          Text("Frame \(progress.inputFrames) of \(progress.estimatedInputFrames)").font(.callout.weight(.medium))
          Text(detail(progress)).font(.caption).foregroundStyle(.secondary)
        }
        .monospacedDigit()
        .fixedSize()
      } else {
        ProgressView().controlSize(.small)
        Text("Processing…").font(.callout.weight(.medium))
        Spacer(minLength: 0)
      }
      Button("Stop", systemImage: "stop.fill", role: .destructive) { model.cancel() }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .help("Stop the queue (⌘.)")
    }
    .padding(.horizontal, 16)
    .frame(height: 52)
  }

  private func detail(_ progress: MediaProgress) -> String {
    var parts: [String] = []
    if progress.elapsedSeconds > 0, progress.inputFrames > 0 {
      parts.append(String(format: "%.1f fps", Double(progress.inputFrames) / progress.elapsedSeconds))
    }
    if let remaining = job.remainingSeconds {
      parts.append(Duration.seconds(remaining).formatted(.time(pattern: .minuteSecond)) + " left")
    }
    if progress.sceneResets > 0 { parts.append("\(progress.sceneResets) scene cuts") }
    return parts.isEmpty ? "Starting…" : parts.joined(separator: " · ")
  }
}

@available(macOS 26.0, *)
private struct ZoomCluster: View {
  @Bindable var canvas: CanvasState

  var body: some View {
    HStack(spacing: 2) {
      Button { withAnimation(.smooth(duration: 0.25)) { canvas.zoomOut() } } label: {
        Image(systemName: "minus.magnifyingglass").frame(width: 26, height: 26)
      }
      .disabled(canvas.zoom <= canvas.minZoom + 1e-4)
      .help("Zoom out (⌘−)")
      Button { withAnimation(.smooth(duration: 0.3)) { canvas.isFit ? canvas.actualSize() : canvas.zoomToFit() } } label: {
        Text(canvas.isFit ? "Fit" : "\(canvas.percent)%").monospacedDigit().frame(minWidth: 44)
      }
      .help(canvas.isFit ? "Actual pixels (⌘0)" : "Zoom to fit (⌘9)")
      Button { withAnimation(.smooth(duration: 0.25)) { canvas.zoomIn() } } label: {
        Image(systemName: "plus.magnifyingglass").frame(width: 26, height: 26)
      }
      .disabled(canvas.zoom >= canvas.maxZoom - 1e-4)
      .help("Zoom in (⌘+)")
    }
    .buttonStyle(.borderless)
    .padding(.horizontal, 10)
    .frame(height: 52)
    .glassEffect(.regular, in: .capsule)
  }
}

/// One line that says what the canvas shows and what happened last.
@available(macOS 26.0, *)
private struct StatusCapsule: View {
  @Bindable var model: AppModel
  @Bindable var canvas: CanvasState
  let job: MediaJob

  var body: some View {
    Group {
      if job.state == .running {
        EmptyView()
      } else if job.state == .failed, let error = job.error {
        capsule {
          Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
          Text(error).lineLimit(1).truncationMode(.tail).help(error)
          Button("Try Again") { model.retry(job) }
        }
      } else if let error = model.previewError, !canvas.showsResult {
        capsule {
          Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
          Text(error).lineLimit(1).truncationMode(.tail).help(error)
          if error.contains("Choose") { SettingsLink { Text("Models…") } }
        }
      } else if job.state == .complete, let result = job.result {
        capsule {
          Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
          Text(exportSummary(result))
          Button(canvas.showsResult ? "Show Preview" : "Show Result") {
            withAnimation(.smooth) { canvas.showsResult.toggle() }
          }
          Button("Reveal", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([result.output]) }
            .labelStyle(.iconOnly)
            .help("Show the result in Finder")
          if model.previewRefreshing && !canvas.showsResult { ProgressView().controlSize(.mini) }
        }
      } else if model.previewRefreshing {
        capsule {
          ProgressView().controlSize(.small)
          Text(model.preview == nil || model.previewIsStale ? "Rendering frame…" : "Updating…")
        }
      } else if let preview = model.preview {
        capsule {
          Text(previewSummary(preview)).foregroundStyle(.secondary)
        }
        .help(preview.historyFrames > 0
          ? "Temporal preview from \(preview.historyFrames) preceding frames. Export uses the complete sequence and adds frame generation."
          : "Single-frame preview")
      }
    }
    .animation(.smooth(duration: 0.3), value: model.previewRefreshing)
  }

  private func capsule(@ViewBuilder content: () -> some View) -> some View {
    HStack(spacing: 8) { content() }
      .font(.callout)
      .buttonStyle(.borderless)
      .padding(.horizontal, 14)
      .frame(height: 34)
      .glassEffect(.regular, in: .capsule)
      .fixedSize(horizontal: false, vertical: true)
  }

  private func previewSummary(_ preview: MediaPreviewResult) -> String {
    let source = "\(preview.original.width) × \(preview.original.height)"
    let output = "\(preview.processed.width) × \(preview.processed.height)"
    let size = source == output ? source : "\(source) → \(output)"
    return "\(size) · \(String(format: "%.2f s", preview.elapsedSeconds))"
  }

  private func exportSummary(_ result: MediaProcessingResult) -> String {
    let seconds = String(format: "%.1f s", result.elapsedSeconds)
    return job.isVideo ? "Exported \(result.outputFrames) frames in \(seconds)" : "Exported in \(seconds)"
  }
}

// MARK: - Result and empty states

@available(macOS 26.0, *)
private struct ResultView: View {
  let url: URL
  let video: Bool
  /// The source's shape; the effects keep it, so the player needs no letterbox.
  let aspect: CGSize?
  @State private var player = AVPlayer()

  var body: some View {
    Group {
      if video {
        VideoPlayer(player: player)
          .aspectRatio(aspect ?? CGSize(width: 16, height: 9), contentMode: .fit)
          .clipShape(.rect(cornerRadius: 12))
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      } else if let image = NSImage(contentsOf: url) {
        Image(nsImage: image).resizable().scaledToFit()
      } else {
        ContentUnavailableView("Result Unavailable", systemImage: "photo", description: Text(url.path))
      }
    }
    .padding(.top, 48)
    .task(id: url) {
      player.pause()
      player.replaceCurrentItem(with: video ? AVPlayerItem(url: url) : nil)
    }
    .onDisappear { player.pause() }
  }
}

@available(macOS 26.0, *)
private struct EmptyCanvas: View {
  @Bindable var model: AppModel

  var body: some View {
    VStack(spacing: 18) {
      Image(systemName: "sparkles.rectangle.stack")
        .font(.system(size: 60, weight: .light))
        .foregroundStyle(.secondary)
        .symbolRenderingMode(.hierarchical)
      VStack(spacing: 6) {
        Text("Drop Images or Video").font(.title.weight(.semibold))
        Text("Neural rendering, frame generation and 2× upscaling run on this Mac's GPU. Nothing leaves the computer.")
          .font(.title3)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
          .frame(maxWidth: 440)
      }
      Button("Import…") { model.chooseFiles() }
        .buttonStyle(.glassProminent)
        .controlSize(.large)
      ModelReadiness(model: model).padding(.top, 10)
    }
    .padding(40)
  }
}

/// Which model packages the app found, with a way to fix a missing one.
@available(macOS 26.0, *)
private struct ModelReadiness: View {
  @Bindable var model: AppModel

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      ForEach(AppModel.ModelKind.allCases) { kind in
        let available = model.isAvailable(kind)
        Label {
          Text(kind.title).foregroundStyle(available ? .primary : .secondary)
        } icon: {
          Image(systemName: available ? "checkmark.circle.fill" : "circle.dashed")
            .foregroundStyle(available ? .green : .secondary)
        }
      }
      SettingsLink { Text("Manage Models…") }
        .buttonStyle(.link)
        .padding(.top, 4)
    }
    .font(.callout)
  }
}
