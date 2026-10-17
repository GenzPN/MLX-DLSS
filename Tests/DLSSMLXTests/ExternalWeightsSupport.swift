import Foundation
import XCTest

/// `MLXDLSS_LOGICAL_WEIGHTS` when the file holds the 71-block HT network.
/// Probes that address HT block indices skip other networks: the 31-block
/// small network has neither the global block 31 nor the post block 70.
func htLogicalWeightsPath(_ message: String) throws -> String {
  guard let path = ProcessInfo.processInfo.environment["MLXDLSS_LOGICAL_WEIGHTS"] else { throw XCTSkip(message) }
  guard try safetensorsTensorNames(path).contains(where: { $0.hasPrefix("block70.") }) else {
    throw XCTSkip("MLXDLSS_LOGICAL_WEIGHTS holds another network; this probe addresses the 71-block HT graph")
  }
  return path
}

/// Tensor names from a safetensors header, without reading the tensors.
private func safetensorsTensorNames(_ path: String) throws -> [String] {
  let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
  defer { try? handle.close() }
  guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else { return [] }
  let length = UInt64(littleEndian: prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
  guard length < 100_000_000, let header = try handle.read(upToCount: Int(length)),
    let object = try JSONSerialization.jsonObject(with: header) as? [String: Any]
  else { return [] }
  return Array(object.keys)
}
