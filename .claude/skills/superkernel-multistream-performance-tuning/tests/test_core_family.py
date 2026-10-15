import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import multistream_core_family  # noqa: E402


class CoreFamilyTest(unittest.TestCase):
    def classify(self, accelerator_core, block_num, mix_block_num):
        return multistream_core_family.classify_profile_identity(
            accelerator_core, block_num, mix_block_num
        )["core_family"]

    def test_pure_accelerator_cores(self):
        self.assertEqual(self.classify("AI_VECTOR_CORE", 32, 0), "VECTOR")
        self.assertEqual(self.classify("AI_CORE", 22, 0), "CUBE")

    def test_zero_mix_blocks_make_mix_aiv_or_aic_pure(self):
        self.assertEqual(self.classify("MIX_AIV", 48, 0), "VECTOR")
        self.assertEqual(self.classify("MIX_AIC", 24, 0), "CUBE")

    def test_nonzero_mix_blocks_make_mix_aiv_or_aic_mix(self):
        self.assertEqual(self.classify("MIX_AIV", 32, 16), "MIX")
        self.assertEqual(self.classify("MIX_AIC", 24, 48), "MIX")

    def test_all_three_profile_fields_are_required_for_compute_rows(self):
        with self.assertRaisesRegex(ValueError, "block_num"):
            self.classify("MIX_AIV", None, 0)
        with self.assertRaisesRegex(ValueError, "mix_block_num"):
            self.classify("MIX_AIC", 24, None)


if __name__ == "__main__":
    unittest.main()
