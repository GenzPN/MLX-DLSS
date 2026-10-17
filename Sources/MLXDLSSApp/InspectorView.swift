import DLSSCore
import DLSSMedia
import SwiftUI

/// The pipeline in processing order. Each stage switches on in its header and
/// shows its controls only while it is on.
@available(macOS 26.0, *)
struct InspectorView: View {
  @Bindable var model: AppModel

  private var imageSelected: Bool { model.selectedJob?.isVideo == false }

  var body: some View {
    Form {
      rendering
      temporal
      generation
      upscaling
      export
    }
    .formStyle(.grouped)
    .disabled(model.isRunning)
    .inspectorColumnWidth(min: 290, ideal: 320, max: 400)
  }

  // MARK: Stages

  private var rendering: some View {
    Section {
      StageToggle(title: "Neural Rendering", systemImage: "wand.and.sparkles", tint: .blue,
        detail: model.renderingEnabled ? "\(model.renderingNetwork.rawValue) network · \(Self.title(model.profile))" : "Off",
        isOn: $model.renderingEnabled)
      if model.renderingEnabled {
        if !model.isAvailable(.rendering) {
          MissingModelRow(model: model, kind: .rendering)
        } else if model.renderingNetworks.count > 1 {
          Picker("Network", selection: $model.renderingNetwork) {
            Text("Small").tag(AppModel.RenderingNetwork.small)
            Text("HT").tag(AppModel.RenderingNetwork.ht)
          }
          .pickerStyle(.segmented)
          .help("Small: 31 blocks, NVIDIA's 310.9.16 default and about twice as fast. HT: 71 blocks.")
        }
        Picker("Look", selection: $model.profile) {
          ForEach(NeuralRenderingControlProfile.allCases, id: \.self) { profile in
            Text(Self.title(profile)).tag(profile)
          }
        }
        .help("Standard is the strongest look, Natural is gentler, Cinematic is film-oriented. Neutral turns tone and structure off for diagnosis.")
        AdjustSlider(title: "Intensity", value: $model.intensity, range: 0...2, neutral: 1,
          help: "Overall effect strength. 0 bypasses the network.")
        AdjustSlider(title: "Detail", value: $model.detailStrength, range: 0...8, neutral: 1,
          help: "Strength of fine changes.")
        AdjustSlider(title: "Colour", value: $model.colourStrength, range: 0...4, neutral: 1,
          help: "Strength of broad colour changes.")
        DisclosureGroup("More") {
          AdjustSlider(title: "Detail radius", value: $model.detailRadius, range: 0.5...64, neutral: 4,
            help: "Size of the detail the Detail control acts on, in pixels.")
          AdjustSlider(title: "Processing scale", value: $model.processingScale, range: 1...4, neutral: 1,
            help: "Runs the network at up to 4× the source size: more detail at higher cost.")
        }
      }
    } footer: {
      if model.renderingEnabled && !model.adjustmentsAreNeutral {
        HStack {
          Spacer()
          Button("Reset Adjustments") { withAnimation { model.resetAdjustments() } }.buttonStyle(.link)
        }
      }
    }
  }

  private var temporal: some View {
    Section {
      StageToggle(title: "Temporal History", systemImage: "clock.arrow.circlepath", tint: .teal,
        detail: imageSelected ? "Video only" : model.temporal ? "Carries detail from preceding frames" : "Each frame on its own",
        isOn: $model.temporal)
      if model.temporal && !imageSelected {
        Picker("Motion", selection: $model.motion) {
          Text("Automatic").tag(MediaMotion.automatic)
          Text("VideoToolbox").tag(MediaMotion.videoToolbox)
          Text("Vision").tag(MediaMotion.vision)
          Text("None (diagnostic)").tag(MediaMotion.zero)
        }
        .help("Optical flow that carries the history between frames.")
        AdjustSlider(title: "Scene cut threshold", value: $model.cutThreshold, range: 0...1, neutral: 0.3,
          help: "Lower values reset the history at smaller changes between frames.")
      }
    }
    .disabled(imageSelected || !(model.renderingEnabled || (model.superResolutionEnabled && model.usesDLSS)))
  }

  private var generation: some View {
    Section {
      StageToggle(title: "Frame Generation", systemImage: "square.stack.3d.forward.dottedline", tint: .orange,
        detail: imageSelected ? "Video only" : model.generationEnabled ? "\(model.factor)× frames\(model.slowMotion ? " · slow motion" : "")" : "Off",
        isOn: $model.generationEnabled)
      if model.generationEnabled && !imageSelected {
        if !model.isAvailable(.generation) { MissingModelRow(model: model, kind: .generation) }
        Picker("Frames", selection: $model.factor) {
          ForEach([2, 3, 4, 8, 16], id: \.self) { Text("\($0)×").tag($0) }
        }
        .pickerStyle(.segmented)
        .help("Output frames per source frame.")
        Toggle("Slow motion", isOn: $model.slowMotion)
          .help("Keep the source frame rate, so the clip plays \(model.factor)× longer.")
        if model.renderingEnabled {
          Picker("Order", selection: $model.order) {
            Text("Render, then generate").tag(MediaEffectOrder.renderingThenGeneration)
            Text("Generate, then render").tag(MediaEffectOrder.generationThenRendering)
          }
        }
      }
    } footer: {
      if model.generationEnabled && !imageSelected {
        Text("Runs during export; the preview shows single frames.").foregroundStyle(.secondary)
      }
    }
    .disabled(imageSelected)
  }

  private var upscaling: some View {
    Section {
      StageToggle(title: "Super Resolution", systemImage: "arrow.up.left.and.arrow.down.right", tint: .green,
        detail: model.superResolutionEnabled ? "2× · \(model.usesDLSS ? "DLSS SR" : "RTX VSR")" : "Off",
        isOn: $model.superResolutionEnabled)
      if model.superResolutionEnabled {
        if model.selectedJob?.isVideo != false {
          Picker("Upscaler", selection: $model.videoUpscaling) {
            Text("DLSS SR").tag("dlss")
            Text("RTX VSR").tag("vsr")
          }
          .pickerStyle(.segmented)
          .help("DLSS SR uses motion and preceding frames. RTX VSR works on each frame alone.")
        }
        if !model.isAvailable(model.upscalingKind) { MissingModelRow(model: model, kind: model.upscalingKind) }
      }
    } footer: {
      if model.superResolutionEnabled {
        Text(imageSelected ? "Experimental. Images use RTX VSR." : "Experimental.").foregroundStyle(.secondary)
      }
    }
  }

  private var export: some View {
    Section {
      Picker("Video format", selection: $model.codec) {
        Text("H.264").tag(MediaVideoCodec.h264)
        Text("HEVC").tag(MediaVideoCodec.hevc)
        Text("ProRes 422 HQ").tag(MediaVideoCodec.prores)
      }
      Toggle("Include audio", isOn: $model.includeAudio)
      DisclosureGroup("Frame Range") {
        TextField("First frame", value: $model.startFrame, format: .number)
        TextField("Frame count", value: $model.frameLimit, format: .number, prompt: Text("All"))
          .help("0 exports every frame from the first one.")
      }
      LabeledContent("Save to") {
        Button {
          model.chooseOutputDirectory()
        } label: {
          Label(URL(fileURLWithPath: model.outputDirectory).lastPathComponent, systemImage: "folder")
            .lineLimit(1)
            .truncationMode(.middle)
        }
      }
      .help(model.outputDirectory)
    } header: {
      Text("Export")
    }
  }

  private static func title(_ profile: NeuralRenderingControlProfile) -> String {
    switch profile {
    case .standard: "Standard"
    case .natural: "Natural"
    case .cinematic: "Cinematic"
    case .neutral: "Neutral (diagnostic)"
    }
  }
}

// MARK: - Rows

/// A stage's switch, in the style of System Settings rows.
@available(macOS 26.0, *)
private struct StageToggle: View {
  let title: String
  let systemImage: String
  let tint: Color
  let detail: String
  @Binding var isOn: Bool

  var body: some View {
    Toggle(isOn: $isOn.animation(.smooth(duration: 0.25))) {
      HStack(spacing: 10) {
        Image(systemName: systemImage)
          .font(.system(size: 13, weight: .semibold))
          .foregroundStyle(.white)
          .frame(width: 26, height: 26)
          .background(tint.gradient, in: .rect(cornerRadius: 7))
          .opacity(isOn ? 1 : 0.55)
        VStack(alignment: .leading, spacing: 1) {
          Text(title).font(.body.weight(.semibold))
          Text(detail).font(.caption).foregroundStyle(.secondary).contentTransition(.opacity)
        }
      }
    }
    .toggleStyle(.switch)
  }
}

/// A slider that fills from its default and resets on a double-click of its name.
@available(macOS 26.0, *)
struct AdjustSlider: View {
  let title: String
  @Binding var value: Double
  let range: ClosedRange<Double>
  let neutral: Double
  let help: String

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack(alignment: .firstTextBaseline) {
        Text(title)
        Spacer()
        Text(value, format: .number.precision(.fractionLength(2)))
          .monospacedDigit()
          .foregroundStyle(value == neutral ? .secondary : .primary)
      }
      .contentShape(Rectangle())
      .onTapGesture(count: 2) { withAnimation(.smooth(duration: 0.2)) { value = neutral } }
      Slider(value: $value, in: range, neutralValue: neutral) { Text(title) }
        .labelsHidden()
    }
    .help("\(help) Double-click the name to reset.")
    .contextMenu {
      Button("Reset to \(neutral.formatted(.number.precision(.fractionLength(0...2))))") { value = neutral }
        .disabled(value == neutral)
    }
  }
}

@available(macOS 26.0, *)
private struct MissingModelRow: View {
  @Bindable var model: AppModel
  let kind: AppModel.ModelKind

  var body: some View {
    HStack {
      Label {
        Text("Weights not found")
      } icon: {
        Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
      }
      Spacer()
      Button("Choose…") { model.choose(kind) }
    }
    .help("Choose the \(kind.title.lowercased()) \(kind.fileKind), or set it in Settings › Models.")
  }
}
