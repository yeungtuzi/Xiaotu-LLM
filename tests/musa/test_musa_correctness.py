"""Unit tests for the opt-in MUSA correctness mode (stdlib only)."""
from __future__ import annotations
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

MOD = Path(__file__).resolve().parents[2] / "vllm_xiaotu_moe" / "musa_correctness.py"


def load():
    spec = importlib.util.spec_from_file_location("xtu_musa_correctness", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMusaCorrectness(unittest.TestCase):
    def test_disabled_noop(self):
        mod = load()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod.apply(), [])

    def test_defaults_set(self):
        mod = load()
        with patch.dict(os.environ, {"XTU_MUSA_CORRECTNESS": "1"}, clear=True):
            applied = mod.apply()
            self.assertEqual(os.environ["VLLM_EXPERTS_LOAD_DEVICE"], "cpu")
            self.assertEqual(os.environ["XIAOTU_MOE_GPU_RESIDENT_LAYERS"], "")
            self.assertEqual(os.environ["XIAOTU_MOE_NSLICE_SMALL"], "0")
            self.assertEqual(os.environ["VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS"], "0")
            self.assertEqual(os.environ["XIAOTU_PLE_CPU"], "1")
            self.assertEqual(os.environ["XIAOTU_SPEC_DECODE"], "0")
            self.assertTrue(any("VLLM_EXPERTS_LOAD_DEVICE" in x for x in applied))

    def test_explicit_experts_device_wins(self):
        mod = load()
        with patch.dict(os.environ, {
            "XTU_MUSA_CORRECTNESS": "1",
            "VLLM_EXPERTS_LOAD_DEVICE": "gpu",
        }, clear=True):
            mod.apply()
            self.assertEqual(os.environ["VLLM_EXPERTS_LOAD_DEVICE"], "gpu")


if __name__ == "__main__":
    unittest.main()
