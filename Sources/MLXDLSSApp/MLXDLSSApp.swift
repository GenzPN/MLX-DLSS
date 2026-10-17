import AppKit
import SwiftUI

@main
enum AppLauncher {
  static func main() {
    guard #available(macOS 26.0, *) else {
      fputs("MLX DLSS requires macOS 26 or newer.\n", stderr)
      return
    }
    NSApplication.shared.setActivationPolicy(.regular)
    MLXDLSSApplication.main()
  }
}

@available(macOS 26.0, *)
struct MLXDLSSApplication: App {
  @State private var model = AppModel()
  @State private var canvas = CanvasState()

  var body: some Scene {
    Window("MLX DLSS", id: "main") {
      MainView(model: model, canvas: canvas)
        .frame(minWidth: 1140, minHeight: 640)
        .onOpenURL { model.addFiles([$0]) }
        .onAppear { NSApplication.shared.activate() }
    }
    .defaultSize(width: 1440, height: 900)
    .commands { AppCommands(model: model, canvas: canvas) }
    Settings {
      SettingsView(model: model)
    }
  }
}
