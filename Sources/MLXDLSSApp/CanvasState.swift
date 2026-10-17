import Foundation
import Observation

/// View state of the canvas: how the source and the rendered frame are compared,
/// and where the view is zoomed. The menu commands and the toolbar share it.
@MainActor @Observable
final class CanvasState {
  enum Compare: String, CaseIterable, Identifiable {
    case original, split, rendered, sideBySide
    var id: Self { self }

    var title: String {
      switch self {
      case .original: "Original"
      case .split: "Split"
      case .rendered: "Rendered"
      case .sideBySide: "Side by Side"
      }
    }

    var symbol: String {
      switch self {
      case .original: "photo"
      case .split: "square.lefthalf.filled"
      case .rendered: "wand.and.sparkles"
      case .sideBySide: "rectangle.split.2x1"
      }
    }
  }

  var compare = Compare(rawValue: UserDefaults.standard.string(forKey: "compareMode") ?? "") ?? .split {
    didSet { UserDefaults.standard.set(compare.rawValue, forKey: "compareMode") }
  }
  /// Shows the exported file instead of the live preview.
  var showsResult = false
  /// Shows the original while the M key is held, as in Photos.
  var peeking = false
  /// Position of the split line across the visible part of the frame, 0...1.
  var divider = 0.5
  /// Magnification relative to the frame fitted into the viewport.
  var zoom = 1.0
  /// Offset of the frame centre from the viewport centre, in points.
  var pan = CGSize.zero

  /// Layout facts reported by the canvas: the viewport, the fitted frame and
  /// points per output pixel at zoom 1.
  var viewport = CGSize(width: 1, height: 1)
  var fitted = CGSize(width: 1, height: 1)
  var fitScale = 1.0
  var displayScale = 2.0

  var mode: Compare { peeking ? .original : compare }
  /// The zoom that maps one output pixel onto one display pixel.
  var actualZoom: Double { 1 / (displayScale * fitScale) }
  var minZoom: Double { min(1, actualZoom) }
  var maxZoom: Double { max(1, actualZoom * 8) }
  var isFit: Bool { abs(zoom - 1) < 1e-4 }
  var canPan: Bool { fitted.width * zoom > viewport.width + 0.5 || fitted.height * zoom > viewport.height + 0.5 }
  /// The zoom as a percentage of actual pixels.
  var percent: Int { Int((zoom / actualZoom * 100).rounded()) }

  func zoomToFit() { zoom = 1; pan = .zero }
  func actualSize() { setZoom(actualZoom) }
  func zoomIn() { setZoom(zoom * 1.5) }
  func zoomOut() { setZoom(zoom / 1.5) }

  func toggleZoom(around anchor: CGPoint = .zero) {
    if isFit { setZoom(max(actualZoom, 2), around: anchor) } else { zoomToFit() }
  }

  /// Keeps the frame point under `anchor` (an offset from the viewport centre) in place.
  func setZoom(_ value: Double, around anchor: CGPoint = .zero) {
    let next = min(maxZoom, max(minZoom, value))
    let ratio = next / zoom
    zoom = next
    pan = clamped(CGSize(width: (pan.width - anchor.x) * ratio + anchor.x, height: (pan.height - anchor.y) * ratio + anchor.y))
  }

  func move(by delta: CGSize) {
    pan = clamped(CGSize(width: pan.width + delta.width, height: pan.height + delta.height))
  }

  /// The pan limited so the frame never leaves an empty gap inside the viewport.
  func clamped(_ offset: CGSize) -> CGSize {
    let x = max(0, (fitted.width * zoom - viewport.width) / 2)
    let y = max(0, (fitted.height * zoom - viewport.height) / 2)
    return CGSize(width: min(x, max(-x, offset.width)), height: min(y, max(-y, offset.height)))
  }

  func resetView() {
    zoomToFit()
    showsResult = false
    peeking = false
  }
}
