import AppKit
import SwiftUI

/// Model packages are chosen once, so they live here instead of in the inspector.
@available(macOS 26.0, *)
struct SettingsView: View {
  @Bindable var model: AppModel

  var body: some View {
    TabView {
      Tab("Models", systemImage: "cube.transparent") {
        Form {
          Section {
            ModelRow(model: model, kind: .rendering)
            if model.renderingNetworks.count > 1 {
              Picker("Network", selection: $model.renderingNetwork) {
                Text("Small · 31 blocks").tag(AppModel.RenderingNetwork.small)
                Text("HT · 71 blocks").tag(AppModel.RenderingNetwork.ht)
              }
            }
            ModelRow(model: model, kind: .generation)
            ModelRow(model: model, kind: .dlssSuperResolution)
            ModelRow(model: model, kind: .superResolution)
          } footer: {
            Text("Extract the packages from NVIDIA's DLLs with mlxdlss-weights; the README lists the supported builds. The Small and HT networks sit side by side, and the app finds one next to the other.")
              .foregroundStyle(.secondary)
              .multilineTextAlignment(.leading)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }
        .formStyle(.grouped)
      }
      Tab("Export", systemImage: "square.and.arrow.up") {
        Form {
          Section {
            LabeledContent("Save to") {
              HStack {
                Text(model.outputDirectory).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                Button("Choose…") { model.chooseOutputDirectory() }
              }
            }
            Button("Show in Finder") {
              NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: model.outputDirectory)
            }
          } footer: {
            Text("Each export goes to its own folder inside this one.")
              .foregroundStyle(.secondary)
              .multilineTextAlignment(.leading)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
        }
        .formStyle(.grouped)
      }
    }
    .frame(width: 600, height: 400)
  }
}

@available(macOS 26.0, *)
private struct ModelRow: View {
  @Bindable var model: AppModel
  let kind: AppModel.ModelKind

  var body: some View {
    let path = model.path(kind)
    let available = model.isAvailable(kind)
    LabeledContent {
      HStack(spacing: 8) {
        if available {
          Button("Reveal", systemImage: "magnifyingglass") {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
          }
          .labelStyle(.iconOnly)
          .buttonStyle(.borderless)
          .help("Show in Finder")
        }
        Button("Choose…") { model.choose(kind) }
      }
    } label: {
      Label {
        VStack(alignment: .leading, spacing: 2) {
          Text(kind.title)
          Text(available ? URL(fileURLWithPath: path).lastPathComponent : path.isEmpty ? "Not set" : "Missing: \(path)")
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
        }
      } icon: {
        Image(systemName: available ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
          .foregroundStyle(available ? .green : .orange)
      }
    }
    .help(path.isEmpty ? kind.fileKind : path)
  }
}
