import AppKit
import SwiftUI

@available(macOS 26.0, *)
struct QueueSidebar: View {
  @Bindable var model: AppModel

  var body: some View {
    List(selection: $model.selection) {
      if !model.jobs.isEmpty {
        Section("Queue") {
          ForEach(model.jobs) { job in
            JobRow(job: job)
              .tag(job.id)
              .contextMenu { menu(for: job) }
          }
        }
      }
    }
    .onDeleteCommand {
      if let job = model.selectedJob { model.remove(job) }
    }
    .safeAreaInset(edge: .bottom, spacing: 0) {
      if !model.jobs.isEmpty {
        HStack {
          Text(summary).font(.callout).foregroundStyle(.secondary).monospacedDigit()
          Spacer()
          Button("Clear Done") { withAnimation { model.removeFinished() } }
            .buttonStyle(.borderless)
            .disabled(!model.hasFinishedJobs)
            .help("Remove finished, failed and cancelled items")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
      }
    }
    .navigationSplitViewColumnWidth(min: 220, ideal: 250, max: 340)
  }

  private var summary: String {
    let queued = model.queuedCount
    let done = model.jobs.count { $0.state == .complete }
    switch (queued, done) {
    case (0, _): return "\(done) done"
    case (_, 0): return "\(queued) queued"
    default: return "\(queued) queued · \(done) done"
    }
  }

  @ViewBuilder private func menu(for job: MediaJob) -> some View {
    if let result = job.result {
      Button("Show Result in Finder") { NSWorkspace.shared.activateFileViewerSelecting([result.output]) }
      Button("Open Result") { NSWorkspace.shared.open(result.output) }
      Divider()
    }
    Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([job.input]) }
    if job.state == .failed || job.state == .cancelled || job.state == .complete {
      Button(job.state == .complete ? "Process Again" : "Try Again") { model.retry(job) }
    }
    Divider()
    Button("Remove", role: .destructive) { model.remove(job) }.disabled(job.state == .running)
  }
}

@available(macOS 26.0, *)
private struct JobRow: View {
  let job: MediaJob

  var body: some View {
    HStack(spacing: 10) {
      Thumbnail(image: job.thumbnail, isVideo: job.isVideo)
      VStack(alignment: .leading, spacing: 3) {
        Text(job.name)
          .lineLimit(1)
          .truncationMode(.middle)
        status
      }
      Spacer(minLength: 0)
      badge
    }
    .padding(.vertical, 3)
    .help(job.error ?? job.input.path)
  }

  @ViewBuilder private var status: some View {
    switch job.state {
    case .running:
      ProgressView(value: job.progress?.fraction ?? 0)
        .controlSize(.small)
    case .complete:
      Text(job.result.map { String(format: "Done in %.1f s", $0.elapsedSeconds) } ?? "Done")
        .font(.subheadline).foregroundStyle(.secondary)
    case .failed:
      Text("Failed").font(.subheadline).foregroundStyle(.red)
    case .cancelled:
      Text("Cancelled").font(.subheadline).foregroundStyle(.secondary)
    case .queued:
      Text(details).font(.subheadline).foregroundStyle(.secondary).monospacedDigit()
    }
  }

  @ViewBuilder private var badge: some View {
    switch job.state {
    case .complete:
      Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).accessibilityLabel("Done")
    case .failed:
      Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red).accessibilityLabel("Failed")
    case .running:
      Text("\(Int((job.progress?.fraction ?? 0) * 100))%").font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
    default:
      EmptyView()
    }
  }

  private var details: String {
    guard let info = job.info else { return job.isVideo ? "Video" : "Image" }
    var parts = ["\(info.width) × \(info.height)"]
    if let duration = info.duration { parts.append(Duration.seconds(duration).formatted(.time(pattern: .minuteSecond))) }
    if let rate = info.frameRate { parts.append("\(Int(rate.rounded())) fps") }
    return parts.joined(separator: " · ")
  }
}

@available(macOS 26.0, *)
private struct Thumbnail: View {
  let image: CGImage?
  let isVideo: Bool

  var body: some View {
    ZStack {
      if let image {
        Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
      } else {
        Rectangle().fill(.quaternary)
        Image(systemName: isVideo ? "film" : "photo").foregroundStyle(.secondary)
      }
    }
    .frame(width: 54, height: 36)
    .clipShape(.rect(cornerRadius: 6))
    .overlay(alignment: .bottomTrailing) {
      if isVideo, image != nil {
        Image(systemName: "play.fill")
          .font(.system(size: 7, weight: .bold))
          .foregroundStyle(.white)
          .padding(3)
          .shadow(color: .black.opacity(0.6), radius: 1)
      }
    }
    .accessibilityHidden(true)
  }
}
