import pathlib
import unittest

from mlxdlss import topology

WEIGHTS = pathlib.Path(__file__).resolve().parents[2] / "weights"


def families(preset: topology.Topology) -> list[tuple[str, int] | None]:
    return [(block.family, block.channels) for block in preset.blocks]


class TopologyInferenceTests(unittest.TestCase):
    def test_complete_maps_recover_their_stage_depths(self):
        for name, preset in topology.KNOWN.items():
            with self.subTest(name):
                self.assertEqual(topology.infer_from_families(families(preset)).spec, preset.spec)

    def test_partial_maps_resolve_to_the_first_agreeing_preset(self):
        partial: list[tuple[str, int] | None] = [None] * 71
        partial[70] = ("post", 32)
        self.assertIs(topology.infer_from_families(partial), topology.HT)
        partial = [None] * 31
        partial[30] = ("post", 32)
        self.assertIs(topology.infer_from_families(partial), topology.SMALL)

    def test_complete_map_that_breaks_the_mirror_is_rejected(self):
        broken = families(topology.SMALL)
        broken[2] = ("window", 32)
        with self.assertRaisesRegex(ValueError, "mirrored topology|downsample"):
            topology.infer_from_families(broken)

    def test_partial_map_that_matches_no_preset_is_rejected(self):
        partial: list[tuple[str, int] | None] = [None] * 71
        partial[70] = ("window", 32)
        with self.assertRaises(ValueError):
            topology.infer_from_families(partial)

    def test_recovered_logical_weights(self):
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors is optional")
        for file, preset in (
            ("dlssnr-weights-logical.safetensors", topology.HT),
            ("dlssnr-small-weights-logical.safetensors", topology.SMALL),
        ):
            path = WEIGHTS / file
            if not path.exists():
                continue
            with self.subTest(file), safe_open(path, framework="numpy") as handle:
                shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}
                self.assertEqual(topology.infer_logical(shapes).spec, preset.spec)


if __name__ == "__main__":
    unittest.main()
