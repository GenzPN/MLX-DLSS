"""``mlxdlss-weights``: bring-your-own-DLL tooling.

    mlxdlss-weights extract  nvngx_dlssnr.dll packed.safetensors [--resource WEIGHTS_HT]  # packed intermediate of one network (default: the newest)
    mlxdlss-weights decode   packed.safetensors logical.safetensors  # logical tensors for PyTorch (and every converter)
    mlxdlss-weights mlx      logical.safetensors Model.dlssmodel      # MLX-DLSS MLX package (Swift mlxdlss)
    mlxdlss-weights coreml   logical.safetensors Model.mlpackage --width W --height H   # fixed-shape Core ML (needs coremltools)
    mlxdlss-weights all      nvngx_dlssnr.dll OUTPUT_DIR [--coreml WxH ...]             # every network of the DLL in one go
    mlxdlss-weights inspect  packed.safetensors                     # list what an (unknown) DLL version contains

The DLL is never redistributed: users take it from their own NVIDIA driver,
game install or Streamline SDK package (bin/x64/nvngx_dlssnr.dll). Supported
builds are identified by their SHA-256: FileVersion 310.8.0.0 carries the
71-block HT network (``WEIGHTS_HT``); 310.9.16.0 adds the 31-block small FP8
network (``WEIGHTS_CC_SMALL_FP8``, its shipping default preset) and keeps HT.
The newest network is the default: ``extract`` takes the small network when the
DLL carries it, and ``all`` writes it first.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

KNOWN_DLL_SHA256 = {
    # the supported nvngx_dlssnr.dll: FileVersion 310.8.0.0 (WEIGHTS_HT resource, 153 packed tensors).
    # There is no separate "checkpoint version": the vNN numbers elsewhere in this project
    # (dlssnr-logical-v18, package names) are revisions of this decoder, not NVIDIA releases.
    "ceb6432f6fbdf44d886014bcd47241932bf8b67439feef9bbdd0961436662650": "supported nvngx_dlssnr.dll (FileVersion 310.8.0.0)",
    # its packed intermediate as written by `mlxdlss-weights extract`
    "08a39bcd6c032c5fec20821c44abfd99a8ad85bbff460e5cc945f35ba67d67a3": "dlssnr-weights-packed.safetensors (WEIGHTS_HT of the supported DLL)",
    # the same intermediate as produced by the earlier standalone extractor
    "1febdf3e3fc868b48e330cea21903927303fb14cd87766169e24bbdf288a9473": "dlssnr-weights-packed.safetensors (supported DLL, earlier extractor metadata)",
    # clean NVIDIA-signed 310.8.0.0 (CL 38718415); same WEIGHTS_HT as above
    "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e": "supported nvngx_dlssnr.dll (FileVersion 310.8.0.0, signed)",
    # NVIDIA-signed 310.9.16.0 (CL 39025366): WEIGHTS_HT (unchanged) + WEIGHTS_CC_SMALL_FP8
    "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f": "supported nvngx_dlssnr.dll (FileVersion 310.9.16.0: HT + small FP8)",
}

# Output names per network resource, newest first; the small network is the 310.9.16 shipping default.
NETWORK_OUTPUTS = {
    "WEIGHTS_CC_SMALL_FP8": ("dlssnr-small-weights", "-Small"),
    "WEIGHTS_HT": ("dlssnr-weights", ""),
}


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(source: pathlib.Path, destination: pathlib.Path, resource_blob: bool = False, resource: str | None = None) -> int:
    from . import extract_dlssnr_weights

    argv = [str(source), str(destination)] + (["--resource-blob"] if resource_blob else []) + (["--resource", resource] if resource else [])
    return _run_with_argv(extract_dlssnr_weights.main, argv)


def _decode(source: pathlib.Path, destination: pathlib.Path) -> int:
    from . import unpack_dlssnr_weights

    return _run_with_argv(unpack_dlssnr_weights.main, [str(source), str(destination)])


def _mlx(source: pathlib.Path, destination: pathlib.Path) -> int:
    from . import package_neural_rendering_transformer

    return _run_with_argv(package_neural_rendering_transformer.main, [str(source), str(destination)])


def _coreml(source: pathlib.Path, destination: pathlib.Path, width: int, height: int, precision: str) -> int:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("coremltools is not installed (pip install 'mlxdlss[coreml]'; macOS or Linux)", file=sys.stderr)
        return 2
    from . import convert_neural_rendering_coreml

    return _run_with_argv(
        convert_neural_rendering_coreml.main,
        [str(source), str(destination), "--precision", precision, "--width", str(width), "--height", str(height)],
    )


def _accepts_argv(function) -> bool:
    import inspect

    return len(inspect.signature(function).parameters) >= 1


def _run_with_argv(function, argv: list[str]) -> int:
    if _accepts_argv(function):
        return int(function(argv) or 0)
    saved = sys.argv
    try:
        sys.argv = [saved[0], *argv]
        return int(function() or 0)
    finally:
        sys.argv = saved


def _inspect(packed: pathlib.Path) -> int:
    from safetensors.numpy import safe_open

    with safe_open(str(packed), framework="numpy") as source:
        metadata = source.metadata() or {}
        names = list(source.keys())
        print(f"format {metadata.get('format', '?')}  tensors {len(names)}  source sha256 {metadata.get('source_sha256', '?')}")
        for name in sorted(names):
            shape = source.get_slice(name).get_shape()
            print(f"  {name}  {shape[0] if len(shape) == 1 else shape} bytes")
    print("decode support is decided per tensor family by `mlxdlss-weights decode`; unknown families are reported there")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlxdlss-weights", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="DLL -> packed intermediate safetensors")
    extract.add_argument("dll", type=pathlib.Path); extract.add_argument("destination", type=pathlib.Path)
    extract.add_argument("--resource", default=None, help="network resource: WEIGHTS_CC_SMALL_FP8 or WEIGHTS_HT; default: the newest the DLL carries (small from 310.9.16)")
    extract.add_argument("--resource-blob", action="store_true", help="source is an already extracted resource")
    decode = commands.add_parser("decode", help="packed intermediate -> logical safetensors")
    decode.add_argument("packed", type=pathlib.Path); decode.add_argument("destination", type=pathlib.Path)
    mlx = commands.add_parser("mlx", help="logical safetensors -> .dlssmodel package for the Swift/MLX runtime")
    mlx.add_argument("logical", type=pathlib.Path); mlx.add_argument("destination", type=pathlib.Path)
    coreml = commands.add_parser("coreml", help="logical safetensors -> fixed-shape Core ML package")
    coreml.add_argument("logical", type=pathlib.Path); coreml.add_argument("destination", type=pathlib.Path)
    coreml.add_argument("--width", type=int, default=320); coreml.add_argument("--height", type=int, default=320)
    coreml.add_argument("--precision", default="float16", choices=("float16", "float32"))
    everything = commands.add_parser("all", help="DLL -> packed, logical and .dlssmodel of every network it carries (plus optional Core ML sizes)")
    everything.add_argument("dll", type=pathlib.Path); everything.add_argument("output_dir", type=pathlib.Path)
    everything.add_argument("--coreml", action="append", default=[], metavar="WxH", help="also build a Core ML package at this size (repeatable)")
    everything.add_argument("--coreml-precision", default="float16", choices=("float16", "float32"))
    everything.add_argument("--name", default="NeuralRendering", help="base name for the produced packages")
    inspect = commands.add_parser("inspect", help="list the tensors of a packed intermediate (for unknown DLL versions)")
    inspect.add_argument("packed", type=pathlib.Path)
    digest = commands.add_parser("sha256", help="print the SHA-256 and file version of a DLL and whether it is a known checkpoint")
    digest.add_argument("path", type=pathlib.Path)
    framegen = commands.add_parser("extract-fg", help="libnvidia-ngx-dlssg.so or nvngx_dlssg.dll -> dense frame generation safetensors for mlxdlss-video / FrameGenerator")
    framegen.add_argument("library", type=pathlib.Path); framegen.add_argument("destination", type=pathlib.Path)
    vsr = commands.add_parser("extract-vsr", help="VSR 1.8.2 library -> High Bitrate Low 2x weights for native Metal")
    vsr.add_argument("library", type=pathlib.Path); vsr.add_argument("destination", type=pathlib.Path)
    sr = commands.add_parser("package-sr", help="local DLSS SR library and CUDA capture -> native .srmodel")
    sr.add_argument("library", type=pathlib.Path); sr.add_argument("capture", type=pathlib.Path)
    sr.add_argument("destination", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sha256":
        from .extract_dlssnr_weights import pe_file_version

        value = sha256_of(args.path)
        version = pe_file_version(args.path.read_bytes()) or "no PE version info"
        print(f"{value}  {args.path}  version {version}  {KNOWN_DLL_SHA256.get(value, 'unknown checkpoint')}")
        return 0
    if args.command == "inspect":
        return _inspect(args.packed)
    if args.command == "extract-fg":
        from .extract_dlssg_weights import main as extract_fg

        return extract_fg([str(args.library), str(args.destination)])
    if args.command == "extract-vsr":
        from .extract_vsr_weights import main as extract_vsr

        return extract_vsr([str(args.library), str(args.destination)])
    if args.command == "package-sr":
        from .package_dlss_sr import main as package_sr

        return package_sr([str(args.library), str(args.capture), str(args.destination)])
    if args.command == "extract":
        return _extract(args.dll, args.destination, args.resource_blob, args.resource)
    if args.command == "decode":
        return _decode(args.packed, args.destination)
    if args.command == "mlx":
        return _mlx(args.logical, args.destination)
    if args.command == "coreml":
        return _coreml(args.logical, args.destination, args.width, args.height, args.precision)
    if args.command == "all":
        from .extract_dlssnr_weights import list_rcdata_resources

        out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
        print(f"dll sha256 {sha256_of(args.dll)}")
        present = set(list_rcdata_resources(args.dll.read_bytes()))
        resources = [name for name in NETWORK_OUTPUTS if name in present]
        if not resources:
            print(f"no known network resource in {args.dll} (found {sorted(present)})", file=sys.stderr); return 2
        for resource in resources:
            stem, suffix = NETWORK_OUTPUTS[resource]
            packed = out / f"{stem}-packed.safetensors"; logical = out / f"{stem}-logical.safetensors"; mlx = out / f"{args.name}{suffix}.dlssmodel"
            steps = (
                ("extract", lambda: _extract(args.dll, packed, resource=resource)),
                ("decode", lambda: _decode(packed, logical)),
                ("mlx", lambda: _mlx(logical, mlx)),
            )
            for step, function in steps:
                code = function()
                if code:
                    print(f"{resource} {step} failed with code {code}", file=sys.stderr); return code
                print(f"{resource} {step}: ok")
            for size in args.coreml:
                width, height = (int(v) for v in size.lower().split("x"))
                code = _coreml(logical, out / f"{args.name}{suffix}-{width}x{height}-{args.coreml_precision}.mlpackage", width, height, args.coreml_precision)
                if code:
                    return code
                print(f"{resource} coreml {size}: ok")
        print(f"artifacts in {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
