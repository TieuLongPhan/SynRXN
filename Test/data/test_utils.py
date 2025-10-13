import unittest
from synrxn.data import utils
import tempfile
from pathlib import Path


class TestUtils(unittest.TestCase):
    def test_normalize_version(self):
        self.assertEqual(utils.normalize_version("v0.0.5"), "0.0.5")
        self.assertEqual(utils.normalize_version("V1.2.3"), "1.2.3")
        self.assertEqual(utils.normalize_version("  2.0 "), "2.0")

    def test_parse_checksum_field(self):
        algo, hex_ = utils.parse_checksum_field("sha256:abcdef1234")
        self.assertEqual(algo, "sha256")
        self.assertEqual(hex_, "abcdef1234")

        algo2, hex2 = utils.parse_checksum_field("not-a-checksum")
        self.assertIsNone(algo2)
        self.assertIsNone(hex2)

    def test_json_save_load_silent(self):
        td = tempfile.TemporaryDirectory()
        p = Path(td.name) / "x.json"
        utils.save_json_silent(p, {"a": 1})
        data = utils.load_json_silent(p)
        self.assertEqual(data.get("a"), 1)
        td.cleanup()

    def test_sha256_hex(self):
        h = utils.sha256_hex("hello")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)


if __name__ == "__main__":
    unittest.main()
