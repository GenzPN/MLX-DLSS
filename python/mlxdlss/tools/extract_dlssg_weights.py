"""Extract the frame generation weights from the user's own DLSS-G library.

Two builds families are supported: ``libnvidia-ngx-dlssg.so.310.7.0`` (DLSS SDK
310.7.0, Linux) and the Windows ``nvngx_dlssg.dll`` of 310.8.0 and 310.9.16
(Streamline SDK 2.14.16). The 42 fp16 weight blobs live uncompressed in the
library's read-only data in the GPU kernels' packed layouts:

* ``custom_*`` convolutions (block0/block1): ``[co/8][tap][ci/8][co%8][ci%8]``;
* ``block1.stem0``: plain ``[co][tap][ci]`` (Cin = 18);
* U-Net ``k_conv_fp16_nhwc``: ``mma.m16n8k16`` A-fragments, bias in the tail.

They are re-laid out as dense ``[Cout, Cin, kh, kw]`` tensors with the names
``mlxdlss.framegen`` expects. The ``.so`` uses absolute offsets. The Windows
builds keep one contiguous weight image whose position moves between builds
(310.8.0 -> 310.9.16: +9328 bytes) while its contents stay byte-identical to the
310.7.0 tensors; the extractor finds it by the SHA-256 of its first 256 bytes and
reads relative offsets. Other builds need a new table. The source digest is
recorded in the output metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import numpy as np

FRAG = np.array([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15])

# name -> (kind, Cout, Cin, k, weight_offset, bias_offset)   offsets in bytes into the .so
SPEC_310_7_0: dict[str, tuple[str, int, int, int, int, int | None]] = {
    "block0.stem0": ("custom", 32, 16, 3, 2056480, 2021664),
    "block0.stem1": ("custom", 32, 32, 3, 2174944, 2021600),
    "block0.stem2": ("custom", 64, 32, 3, 1908704, 1908576),
    "block0.res0": ("custom", 64, 64, 3, 1825216, 1825088),
    "block0.res1": ("custom", 64, 64, 3, 1677632, 1677440),
    "block0.res2": ("custom", 64, 64, 3, 2097952, 1677312),
    "block0.res3": ("custom", 64, 64, 3, 1945568, 1649536),
    "block0.res4": ("custom", 64, 64, 3, 1575808, 1575680),
    "block0.res5": ("custom", 64, 64, 3, 1483520, 1483392),
    "block0.res6": ("custom", 64, 64, 3, 1751360, 1464832),
    "block0.res7": ("custom", 64, 64, 3, 1391104, 1372544),
    "block0.bot0": ("custom3", 32, 64, 3, 1216032, 1215840),
    "block0.bot1": ("custom", 8, 32, 3, 1178016, 1177888),
    "block1.stem0": ("plain", 16, 18, 3, 1162240, 1162208),
    "block1.stem1": ("custom", 32, 16, 3, 1120960, 1120896),
    "block1.res0": ("custom", 32, 32, 3, 2038048, 1113440),
    "block1.res1": ("custom", 32, 32, 3, 1326624, 1677568),
    "block1.res2": ("custom", 32, 32, 3, 1095008, 1094880),
    "block1.res3": ("custom", 32, 32, 3, 1130176, 1372480),
    "block1.res4": ("custom", 32, 32, 3, 1464960, 1094816),
    "block1.res5": ("custom", 32, 32, 3, 1557248, 1094944),
    "block1.res6": ("custom", 32, 32, 3, 1345056, 1113504),
    "block1.res7": ("custom", 32, 32, 3, 1372672, 1154432),
    "block1.bot0": ("custom3", 16, 32, 3, 1649664, 1177920),
    "block1.bot1": ("custom", 8, 16, 3, 2019296, 1094800),
    "unet.conv0": ("unet", 32, 48, 3, 6774976, None),
    "unet.conv1": ("unet", 64, 32, 1, 6697408, None),
    "unet.conv2": ("unet", 128, 64, 3, 6398400, None),
    "unet.conv3": ("unet", 128, 128, 3, 5527104, None),
    "unet.conv4": ("unet", 256, 128, 3, 3760256, None),
    "unet.conv5": ("unet", 512, 256, 1, 3129888, None),
    "unet.conv6": ("unet", 256, 512, 1, 4983456, None),
    "unet.conv7": ("unet", 256, 256, 1, 6121024, None),
    "unet.conv8": ("unet", 128, 256, 1, 2923008, None),
    "unet.conv9": ("unet", 128, 128, 1, 2797216, None),
    "unet.conv10": ("unet", 128, 128, 1, 3670560, None),
    "unet.conv11": ("unet", 128, 128, 1, 2673472, None),
    "unet.conv12": ("unet", 64, 128, 1, 2638656, None),
    "unet.conv13": ("unet", 64, 64, 1, 2590848, None),
    "unet.conv14": ("unet", 32, 64, 1, 2572320, None),
    "unet.conv15": ("unet", 32, 32, 1, 2626496, None),
    "unet.conv16": ("unet", 16, 32, 1, 2541664, None),
}
# name -> (kind, Cout, Cin, k, weight_offset, bias_offset), relative to the weight
# image of nvngx_dlssg.dll 310.8.0 / 310.9.16 (verified bit-exact on both against
# the 310.7.0 tensors).
SPEC_WINDOWS_RELATIVE: dict[str, tuple[str, int, int, int, int, int | None]] = {
    "block0.stem0": ("custom", 32, 16, 3, 127664, 171632),
    "block0.stem1": ("custom", 32, 32, 3, 0, 171696),
    "block0.stem2": ("custom", 64, 32, 3, 247792, 284656),
    "block0.res0": ("custom", 64, 64, 3, 294416, 368144),
    "block0.res1": ("custom", 64, 64, 3, 442000, 515792),
    "block0.res2": ("custom", 64, 64, 3, 21680, 515920),
    "block0.res3": ("custom", 64, 64, 3, 174064, 543696),
    "block0.res4": ("custom", 64, 64, 3, 543824, 617552),
    "block0.res5": ("custom", 64, 64, 3, 636112, 709840),
    "block0.res6": ("custom", 64, 64, 3, 368272, 728400),
    "block0.res7": ("custom", 64, 64, 3, 728528, 820688),
    "block0.bot0": ("custom3", 32, 64, 3, 866736, 977328),
    "block0.bot1": ("custom", 8, 32, 3, 1010720, 1015424),
    "block1.stem0": ("plain", 16, 18, 3, 1025904, 1031088),
    "block1.stem1": ("custom", 32, 16, 3, 1063136, 1072352),
    "block1.res0": ("custom", 32, 32, 3, 136880, 1079808),
    "block1.res1": ("custom", 32, 32, 3, 848304, 515728),
    "block1.res2": ("custom", 32, 32, 3, 1079872, 1098368),
    "block1.res3": ("custom", 32, 32, 3, 1044704, 820816),
    "block1.res4": ("custom", 32, 32, 3, 709968, 1098432),
    "block1.res5": ("custom", 32, 32, 3, 617680, 1098304),
    "block1.res6": ("custom", 32, 32, 3, 829872, 1079744),
    "block1.res7": ("custom", 32, 32, 3, 802256, 1038832),
    "block1.bot0": ("custom3", 16, 32, 3, 516048, 1015328),
    "block1.bot1": ("custom", 8, 16, 3, 171760, 1098496),
    "unet.conv0": ("unet", 32, 48, 3, 1314064, None),
    "unet.conv1": ("unet", 64, 32, 1, 1437648, None),
    "unet.conv2": ("unet", 128, 64, 3, 1447888, None),
    "unet.conv3": ("unet", 128, 128, 3, 2024256, None),
    "unet.conv4": ("unet", 256, 128, 3, 3197168, None),
    "unet.conv5": ("unet", 512, 256, 1, 4474704, None),
    "unet.conv6": ("unet", 256, 512, 1, 2629344, None),
    "unet.conv7": ("unet", 256, 256, 1, 1753920, None),
    "unet.conv8": ("unet", 128, 256, 1, 5087056, None),
    "unet.conv9": ("unet", 128, 128, 1, 5278336, None),
    "unet.conv10": ("unet", 128, 128, 1, 4405072, None),
    "unet.conv11": ("unet", 128, 128, 1, 5402064, None),
    "unet.conv12": ("unet", 64, 128, 1, 5471696, None),
    "unet.conv13": ("unet", 64, 64, 1, 5535872, None),
    "unet.conv14": ("unet", 32, 64, 1, 5563616, None),
    "unet.conv15": ("unet", 32, 32, 1, 5513536, None),
    "unet.conv16": ("unet", 16, 32, 1, 5600896, None),
}
# SHA-256 of the first 256 bytes of that image (block0.stem1's packed weights).
WINDOWS_IMAGE_SHA256 = "4ee053756689bbcf925e7bbc36fe40e7c572b5e2188bba7951e6d57e44b500d6"
WINDOWS_IMAGE_ALIGNMENT = 16


def locate_windows_image(blob: bytes) -> int | None:
    """Offset of the weight image in a Windows build, or ``None``."""
    view = memoryview(blob)
    for offset in range(0, len(blob) - 256 + 1, WINDOWS_IMAGE_ALIGNMENT):
        if hashlib.sha256(view[offset : offset + 256]).hexdigest() == WINDOWS_IMAGE_SHA256:
            return offset
    return None


def spec_for(blob: bytes) -> tuple[dict[str, tuple[str, int, int, int, int, int | None]], str]:
    """The offset table for ``blob`` and a short description of the build family."""
    if blob[:4] == b"\x7fELF":
        return SPEC_310_7_0, "libnvidia-ngx-dlssg.so 310.7.0 table"
    if blob[:2] == b"MZ":
        base = locate_windows_image(blob)
        if base is None:
            raise ValueError("no known frame generation weight image in this nvngx_dlssg.dll (supported: 310.8.0, 310.9.16)")
        spec = {
            name: (kind, cout, cin, k, base + woff, None if boff is None else base + boff)
            for name, (kind, cout, cin, k, woff, boff) in SPEC_WINDOWS_RELATIVE.items()
        }
        return spec, f"nvngx_dlssg.dll weight image at {base}"
    raise ValueError("not an ELF or PE library")


def _halfs(blob: bytes, offset: int, count: int) -> np.ndarray:
    end = offset + count * 2
    if end > len(blob):
        raise ValueError(f"blob at {offset} + {count} halfs exceeds the file ({len(blob)} bytes)")
    return np.frombuffer(blob, np.float16, count, offset)


def custom_dense(halfs: np.ndarray, cout: int, cin: int) -> np.ndarray:
    a = halfs.reshape(cout // 8, 9, cin // 8, 8, 8)
    return np.ascontiguousarray(a.transpose(0, 3, 2, 4, 1).reshape(cout, cin, 3, 3))


def plain_dense(halfs: np.ndarray, cout: int, cin: int) -> np.ndarray:
    return np.ascontiguousarray(halfs.reshape(cout, 9, cin).transpose(0, 2, 1).reshape(cout, cin, 3, 3))


def unet_dense(halfs: np.ndarray, cout: int, cin: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    K = k * k * cin
    KB = K // 16
    dy, dx, ci, n = np.meshgrid(np.arange(k), np.arange(k), np.arange(cin), np.arange(cout), indexing="ij")
    kk = (dy * k + dx) * cin + ci
    idx = (n // 8) * KB * 128 + (kk // 16) * 128 + (n % 8) * 16 + FRAG[kk % 16]
    w = halfs[idx]  # [k, k, cin, cout]
    bias = halfs[2 * K * cout :][(np.arange(cout) // 8) * 128 + np.arange(cout) % 8]
    return np.ascontiguousarray(w.transpose(3, 2, 0, 1)), np.ascontiguousarray(bias)


def extract(library: pathlib.Path, spec: dict | None = None) -> tuple[dict[str, np.ndarray], str]:
    blob = library.read_bytes()
    spec = spec or spec_for(blob)[0]
    digest = hashlib.sha256(blob).hexdigest()
    out: dict[str, np.ndarray] = {}
    for name, (kind, cout, cin, k, woff, boff) in spec.items():
        if kind == "custom":
            out[f"{name}.weight"] = custom_dense(_halfs(blob, woff, 9 * cin * cout), cout, cin)
            out[f"{name}.bias"] = np.ascontiguousarray(_halfs(blob, boff, cout))
        elif kind == "custom3":  # three heads concatenated (weights then biases)
            w = _halfs(blob, woff, 3 * 9 * cin * cout)
            b = _halfs(blob, boff, 3 * cout)
            for i in range(3):
                out[f"{name}.head{i}.weight"] = custom_dense(w[i * 9 * cin * cout : (i + 1) * 9 * cin * cout], cout, cin)
                out[f"{name}.head{i}.bias"] = np.ascontiguousarray(b[i * cout : (i + 1) * cout])
        elif kind == "plain":
            out[f"{name}.weight"] = plain_dense(_halfs(blob, woff, 9 * cin * cout), cout, cin)
            out[f"{name}.bias"] = np.ascontiguousarray(_halfs(blob, boff, cout))
        elif kind == "unet":
            K = k * k * cin
            w, b = unet_dense(_halfs(blob, woff, 2 * K * cout + (cout // 8) * 128), cout, cin, k)
            out[f"{name}.weight"] = w
            out[f"{name}.bias"] = b
        else:
            raise ValueError(kind)
    return out, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "library", type=pathlib.Path,
        help="libnvidia-ngx-dlssg.so.310.7.0 (DLSS SDK) or nvngx_dlssg.dll 310.8.0 / 310.9.16 (Streamline SDK)")
    parser.add_argument("destination", type=pathlib.Path, help="output .safetensors")
    args = parser.parse_args(argv)
    from safetensors.numpy import save_file

    tensors, digest = extract(args.library)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.destination), metadata={"format": "dlssg-framegen-dense-v1", "layout": "dense [Cout,Cin,kh,kw] fp16", "source": args.library.name, "source_sha256": digest})
    params = sum(int(v.size) for v in tensors.values())
    print(f"wrote {len(tensors)} tensors, {params:,} parameters -> {args.destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
