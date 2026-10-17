import hashlib
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from mlxdlss.tools import extract_dlssg_weights as fg


class WindowsImageTests(unittest.TestCase):
    """The Windows builds are read through a hashed anchor and relative offsets."""

    def _library(self, base: int) -> tuple[bytes, np.ndarray]:
        rng = np.random.default_rng(1)
        end = max(w + 3 * 9 * cin * cout * 2 if kind == "custom3" else w + 9 * cin * cout * 2 + 4096
                  for kind, cout, cin, _, w, _ in fg.SPEC_WINDOWS_RELATIVE.values())
        image = bytearray(rng.integers(0, 256, end + 65536, dtype=np.uint8).tobytes())
        # block0.stem1 is a plain custom convolution at relative offset 0.
        _, cout, cin, _, woff, boff = fg.SPEC_WINDOWS_RELATIVE["block0.stem1"]
        dense = rng.standard_normal((cout, cin, 3, 3)).astype(np.float16)
        packed = dense.reshape(cout // 8, 8, cin // 8, 8, 9).transpose(0, 4, 2, 1, 3).tobytes()
        image[woff : woff + len(packed)] = packed
        bias = rng.standard_normal(cout).astype(np.float16)
        image[boff : boff + bias.nbytes] = bias.tobytes()
        blob = b"MZ" + bytes(base - 2) + bytes(image)
        return blob, np.concatenate([dense.reshape(-1), bias])

    def test_image_is_found_at_any_aligned_position(self):
        for base in (4096, 4096 + 16 * 37):
            blob, expected = self._library(base)
            anchor = hashlib.sha256(blob[base : base + 256]).hexdigest()
            with mock.patch.object(fg, "WINDOWS_IMAGE_SHA256", anchor):
                self.assertEqual(fg.locate_windows_image(blob), base)
                spec, _ = fg.spec_for(blob)
                self.assertEqual(spec["block0.stem1"][4], base)
                with tempfile.TemporaryDirectory() as directory:
                    path = pathlib.Path(directory) / "nvngx_dlssg.dll"
                    path.write_bytes(blob)
                    tensors, digest = fg.extract(path)
            self.assertEqual(digest, hashlib.sha256(blob).hexdigest())
            actual = np.concatenate([tensors["block0.stem1.weight"].reshape(-1), tensors["block0.stem1.bias"]])
            np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

    def test_unknown_windows_build_is_rejected(self):
        blob = b"MZ" + bytes(8190)
        with self.assertRaisesRegex(ValueError, "no known frame generation weight image"):
            fg.spec_for(blob)

    def test_elf_uses_the_linux_table(self):
        spec, _ = fg.spec_for(b"\x7fELF" + bytes(64))
        self.assertIs(spec, fg.SPEC_310_7_0)


if __name__ == "__main__":
    unittest.main()
