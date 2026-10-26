"""Unit tests for the opt-in MUSA correctness mode (stdlib only)."""
from __future__ import annotations
import importlib.util
import os
import sys
import types
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
            self.assertEqual(os.environ["VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS"], "0")
            self.assertEqual(os.environ["XIAOTU_PLE_CPU"], "0")
            self.assertEqual(os.environ["XIAOTU_SPEC_DECODE"], "0")
            self.assertTrue(any("VLLM_EXPERTS_LOAD_DEVICE" in x for x in applied))

    def test_gpu_device_musa_enables(self):
        mod = load()
        with patch.dict(os.environ, {"GPU_DEVICE": "MUSA"}, clear=True):
            self.assertTrue(mod.enabled())
            mod.apply()
            self.assertEqual(os.environ["XTU_MUSA_CORRECTNESS"], "1")
            self.assertEqual(os.environ["VLLM_EXPERTS_LOAD_DEVICE"], "cpu")
            self.assertEqual(mod.requested_device(), "MUSA")

    def test_gpu_device_cuda_is_default(self):
        mod = load()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod.requested_device(), "CUDA")
            self.assertFalse(mod.enabled())
            self.assertEqual(mod.apply(), [])

    def test_musa_path_active(self):
        mod = load()
        with patch.dict(os.environ, {"GPU_DEVICE": "MUSA"}, clear=True):
            self.assertTrue(mod.musa_path_active())
        with patch.dict(os.environ, {"GPU_DEVICE": "CUDA"}, clear=True):
            self.assertFalse(mod.musa_path_active())
        with patch.dict(os.environ, {"XTU_MUSA_CORRECTNESS": "1",
                                      "GPU_DEVICE": "CUDA"}, clear=True):
            self.assertTrue(mod.musa_path_active())

    @staticmethod
    def _fake_platform(is_musa: bool):
        """Inject a stand-in ``vllm.platforms`` without importing vLLM."""
        vllm = types.ModuleType("vllm")
        platforms = types.ModuleType("vllm.platforms")

        class _CP:
            @staticmethod
            def is_musa():
                return is_musa

        platforms.current_platform = _CP()
        vllm.platforms = platforms
        return patch.dict(sys.modules, {"vllm": vllm, "vllm.platforms": platforms})

    def test_unset_is_cuda_even_on_musa_host(self):
        """GPU_DEVICE unset must take the CUDA path regardless of hardware."""
        mod = load()
        with self._fake_platform(True), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod.requested_device(), "CUDA")
            self.assertFalse(mod.musa_path_active())
        mod = load()
        with self._fake_platform(True), patch.dict(
                os.environ, {"GPU_DEVICE": "CUDA"}, clear=True):
            self.assertFalse(mod.musa_path_active())

    def test_unknown_value_falls_back_to_autodetect(self):
        mod = load()
        with self._fake_platform(True), patch.dict(
                os.environ, {"GPU_DEVICE": "FOO"}, clear=True):
            self.assertTrue(mod.musa_path_active())
        with self._fake_platform(False), patch.dict(
                os.environ, {"GPU_DEVICE": "FOO"}, clear=True):
            self.assertFalse(mod.musa_path_active())

    def test_alias_values_are_musa(self):
        mod = load()
        for value in ("MUSA", "musa", "MTGPU", "mtgpu", "S4000", "s4000"):
            with patch.dict(os.environ, {"GPU_DEVICE": value}, clear=True):
                self.assertTrue(mod.musa_path_active(), value)

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
