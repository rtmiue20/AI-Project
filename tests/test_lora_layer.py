"""Unit tests for pure-PyTorch LoRALinear and parameter calculation utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calculate_parameters import analyze_linear_layer
from src.lora_layer import LoRALinear


class TestLoRALinear(unittest.TestCase):
    """Test initialization, gradient flow, weight merging, and unmerging."""

    def setUp(self) -> None:
        torch.manual_seed(123)
        self.layer = LoRALinear(
            in_features=64, out_features=128, r=8, alpha=16.0, bias=True
        )
        self.layer.eval()
        self.x = torch.randn(4, 64)

    def test_initial_output_matches_base_linear(self) -> None:
        """Because B is initialized to zero, initial DeltaW must be zero."""
        with torch.no_grad():
            y_lora = self.layer(self.x)
            y_base = self.layer.base_layer(self.x)
        self.assertTrue(torch.allclose(y_lora, y_base, atol=1e-6))

    def test_gradients_flow_only_to_adapters(self) -> None:
        """Base weights remain frozen while lora_A and lora_B receive gradients."""
        self.layer.train()
        # Use non-zero lora_B or two-step check: at step 0, B=0 gives dL/dB != 0
        # and dL/dA = 0 on step 1, then dL/dA != 0 once B != 0. Verify both!
        out = self.layer(self.x)
        loss = out.pow(2).mean()
        loss.backward()

        self.assertIsNone(self.layer.base_layer.weight.grad)
        if self.layer.base_layer.bias is not None:
            self.assertIsNone(self.layer.base_layer.bias.grad)
        assert self.layer.lora_B is not None
        assert self.layer.lora_A is not None
        self.assertIsNotNone(self.layer.lora_B.grad)
        self.assertGreater(self.layer.lora_B.grad.abs().sum().item(), 0.0)

        # Second step with non-zero B confirms lora_A also receives gradients
        with torch.no_grad():
            self.layer.lora_B.normal_(mean=0.0, std=0.05)
        self.layer.zero_grad()
        loss2 = self.layer(self.x).pow(2).mean()
        loss2.backward()

        self.assertIsNone(self.layer.base_layer.weight.grad)
        self.assertIsNotNone(self.layer.lora_A.grad)
        self.assertGreater(self.layer.lora_A.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.layer.lora_B.grad.abs().sum().item(), 0.0)

    def test_merge_weights_equivalence(self) -> None:
        """Merged forward pass must match unmerged forward pass within 1e-5."""
        with torch.no_grad():
            assert self.layer.lora_B is not None
            self.layer.lora_B.normal_(mean=0.0, std=0.1)
            y_1 = self.layer(self.x)
            self.layer.merge_weights()
            y_2 = self.layer(self.x)

        self.assertTrue(self.layer.merged)
        self.assertTrue(torch.allclose(y_1, y_2, atol=1e-5))

    def test_unmerge_weights_restores_original_base(self) -> None:
        """Unmerging must restore original base_layer.weight and unmerged outputs."""
        original_weight = self.layer.base_layer.weight.detach().clone()
        with torch.no_grad():
            assert self.layer.lora_B is not None
            self.layer.lora_B.normal_(mean=0.0, std=0.1)
            y_before = self.layer(self.x)

        self.layer.merge_weights()
        self.assertFalse(torch.allclose(self.layer.base_layer.weight, original_weight, atol=1e-6))

        self.layer.unmerge_weights()
        self.assertFalse(self.layer.merged)
        self.assertTrue(torch.allclose(self.layer.base_layer.weight, original_weight, atol=1e-6))

        with torch.no_grad():
            y_after = self.layer(self.x)
        self.assertTrue(torch.allclose(y_before, y_after, atol=1e-6))


class TestCalculateParameters(unittest.TestCase):
    """Test exact parameter counts and reduction ratios for 1024x1024 matrix."""

    def test_analyze_linear_layer_1024x1024(self) -> None:
        rows = analyze_linear_layer(d=1024, k=1024, ranks=(4, 8, 16))
        by_method = {row["method"]: row for row in rows}

        self.assertEqual(by_method["Full Fine-tuning"]["trainable_params"], 1_048_576)
        self.assertEqual(by_method["Full Fine-tuning"]["reduction_pct"], 0.0)

        self.assertEqual(by_method["LoRA (r=4)"]["trainable_params"], 8_192)
        self.assertAlmostEqual(by_method["LoRA (r=4)"]["reduction_pct"], 99.22, places=2)

        self.assertEqual(by_method["LoRA (r=8)"]["trainable_params"], 16_384)
        self.assertAlmostEqual(by_method["LoRA (r=8)"]["reduction_pct"], 98.44, places=2)

        self.assertEqual(by_method["LoRA (r=16)"]["trainable_params"], 32_768)
        self.assertAlmostEqual(by_method["LoRA (r=16)"]["reduction_pct"], 96.88, places=2)


if __name__ == "__main__":
    unittest.main()
