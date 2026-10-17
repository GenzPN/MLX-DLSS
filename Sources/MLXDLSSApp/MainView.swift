import AppKit
import SwiftUI

@available(macOS 26.0, *)
struct MainView: View {
  @Bindable var model: AppModel
  @Bindable var canvas: CanvasState
  @SceneStorage("inspectorShown") private var inspectorShown = true
  @State private var dropTargeted = false

  var body: some View {
    NavigationSplitView {
      QueueSidebar(model: model)
    } detail: {
      CanvasView(model: model, canvas: canvas)
        .overlay { if dropTargeted { DropHighlight() } }
        .animation(.smooth(duration: 0.2), value: dropTargeted)
    }
    .inspector(isPresented: $inspectorShown) {
      InspectorView(model: model)
    }
    .navigationTitle(model.selectedJob?.name ?? "MLX DLSS")
    .navigationSubtitle(subtitle)
    .toolbar { toolbar }
    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
    .dropDestination(for: URL.self) { urls, _ in
      model.addFiles(urls)
      return !urls.isEmpty
    } isTargeted: { dropTargeted = $0 }
    .onChange(of: model.selection) { canvas.resetView() }
    .alert("Cannot Process Media", isPresented: Binding(get: { model.alert != nil }, set: { if !$0 { model.alert = nil } })) {
      Button("OK") { model.alert = nil }
    } message: { Text(model.alert ?? "") }
  }

  private var subtitle: String {
    guard let job = model.selectedJob else {
      return model.jobs.isEmpty ? "Neural rendering on this Mac" : "\(model.jobs.count) items"
    }
    guard let info = job.info else { return job.isVideo ? "Video" : "Image" }
    var parts = ["\(info.width) × \(info.height)"]
    if let rate = info.frameRate { parts.append(String(format: "%g fps", (rate * 100).rounded() / 100)) }
    if let duration = info.duration { parts.append(Duration.seconds(duration).formatted(.time(pattern: .minuteSecond))) }
    return parts.joined(separator: " · ")
  }

  @ToolbarContentBuilder private var toolbar: some ToolbarContent {
    ToolbarItem(placement: .principal) {
      Picker("Compare", selection: $canvas.compare) {
        ForEach(CanvasState.Compare.allCases) { mode in
          Label(mode.title, systemImage: mode.symbol).tag(mode)
        }
      }
      .pickerStyle(.segmented)
      .labelStyle(.iconOnly)
      .help("Original ⌘1 · Split ⌘2 · Rendered ⌘3 · Side by Side ⌘4. Hold M to see the original.")
      .disabled(model.selectedJob == nil || canvas.showsResult)
    }
    ToolbarItem(placement: .primaryAction) {
      Button("Import", systemImage: "plus") { model.chooseFiles() }
        .help("Import images or video (⌘O)")
    }
    ToolbarSpacer(.fixed, placement: .primaryAction)
    ToolbarItem(placement: .primaryAction) {
      if model.isRunning {
        Button("Stop", systemImage: "stop.fill") { model.cancel() }
          .labelStyle(.titleAndIcon)
          .help("Stop the queue (⌘.)")
      } else {
        Button("Process", systemImage: "play.fill") { model.runQueue() }
          .labelStyle(.titleAndIcon)
          .buttonStyle(.glassProminent)
          .disabled(!model.hasQueuedJobs)
          .help(model.hasQueuedJobs ? "Process \(model.queuedCount) queued items (⌘↩)" : "Nothing is queued")
      }
    }
    // The shared glass would hide the prominent style.
    .sharedBackgroundVisibility(.hidden)
    ToolbarSpacer(.fixed, placement: .primaryAction)
    ToolbarItem(placement: .primaryAction) {
      Button("Inspector", systemImage: "sidebar.trailing") {
        withAnimation { inspectorShown.toggle() }
      }
      .help(inspectorShown ? "Hide the inspector" : "Show the inspector")
    }
  }
}

@available(macOS 26.0, *)
private struct DropHighlight: View {
  var body: some View {
    RoundedRectangle(cornerRadius: 20)
      .strokeBorder(Color.accentColor, lineWidth: 3)
      .padding(10)
      .overlay {
        Label("Add to Queue", systemImage: "plus")
          .font(.title3.weight(.semibold))
          .padding(.horizontal, 20)
          .padding(.vertical, 12)
          .glassEffect(.regular.tint(.accentColor.opacity(0.3)), in: .capsule)
      }
      .allowsHitTesting(false)
  }
}

@available(macOS 26.0, *)
struct AppCommands: Commands {
  let model: AppModel
  let canvas: CanvasState

  var body: some Commands {
    CommandGroup(replacing: .newItem) {
      Button("Import Images or Video…") { model.chooseFiles() }.keyboardShortcut("o")
    }
    CommandMenu("Queue") {
      Button("Process Queue") { model.runQueue() }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(model.isRunning || !model.hasQueuedJobs)
      Button("Stop") { model.cancel() }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!model.isRunning)
      Divider()
      Button("Clear Done") { model.removeFinished() }
        .disabled(!model.hasFinishedJobs)
    }
    CommandGroup(before: .toolbar) {
      ForEach(Array(CanvasState.Compare.allCases.enumerated()), id: \.element) { index, mode in
        Toggle(mode.title, isOn: Binding(get: { canvas.compare == mode }, set: { if $0 { canvas.compare = mode } }))
          .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
      }
      Divider()
      Button("Previous Frame") { model.step(frames: -1) }
        .keyboardShortcut(.leftArrow, modifiers: .option)
        .disabled(model.selectedJob?.isVideo != true)
      Button("Next Frame") { model.step(frames: 1) }
        .keyboardShortcut(.rightArrow, modifiers: .option)
        .disabled(model.selectedJob?.isVideo != true)
      Divider()
      Button("Actual Size") { withAnimation(.smooth(duration: 0.3)) { canvas.actualSize() } }
        .keyboardShortcut("0")
      Button("Zoom to Fit") { withAnimation(.smooth(duration: 0.3)) { canvas.zoomToFit() } }
        .keyboardShortcut("9")
      Button("Zoom In") { withAnimation(.smooth(duration: 0.25)) { canvas.zoomIn() } }
        .keyboardShortcut("+")
      Button("Zoom Out") { withAnimation(.smooth(duration: 0.25)) { canvas.zoomOut() } }
        .keyboardShortcut("-")
      Divider()
    }
  }
}
