"""Pure PyTorch implementation of Low-Rank Adaptation (LoRA) linear layer.

Implements Hu et al. (ICLR 2022):
    h = W_0 * x + (alpha / r) * (B @ A) * x
with zero-overhead weight merging for inference:
    W_deploy = W_0 + (alpha / r) * (B @ A)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with frozen base weights and low-rank adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r < 0:
            raise ValueError(f"Rank r must be non-negative, got {r}")
        if r > 0 and alpha <= 0:
            raise ValueError(f"Scaling alpha must be positive when r > 0, got {alpha}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = (self.alpha / r) if r > 0 else 0.0
        self.merged = False

        # Base pretrained weight W_0 (frozen during LoRA training)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            # A ~ N(0, sigma^2), B = 0 ensures Delta W = B @ A = 0 at step 0
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(r, in_features)
            )
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(out_features, r)
            )
            nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(in_features))
            self.lora_dropout: nn.Module = (
                nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            )
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> "LoRALinear":
        """Wrap an existing nn.Linear layer, copying and freezing its weights."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            layer.base_layer.weight.copy_(linear.weight)
            if linear.bias is not None and layer.base_layer.bias is not None:
                layer.base_layer.bias.copy_(linear.bias)
        return layer

    def delta_weight(self) -> torch.Tensor:
        """Compute the scaled low-rank weight update (alpha / r) * (B @ A)."""
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            return torch.zeros_like(self.base_layer.weight)
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass using merged weights or parallel base + low-rank path."""
        base_out = self.base_layer(x)
        if self.merged or self.r == 0 or self.lora_A is None or self.lora_B is None:
            return base_out

        # Compute (x @ A^T) @ B^T to preserve rank-r bottleneck O(r*(d+k)) rather than O(d*k)
        dropped = self.lora_dropout(x)
        low_rank_out = F.linear(F.linear(dropped, self.lora_A), self.lora_B)
        return base_out + self.scaling * low_rank_out

    def merge_weights(self) -> None:
        """Merge low-rank adapter weights into base_layer.weight for zero-latency inference."""
        if self.r == 0 or self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight.data += self.delta_weight()
        self.merged = True

    def unmerge_weights(self) -> None:
        """Subtract merged adapter weights from base_layer.weight to resume training."""
        if self.r == 0 or not self.merged:
            return
        with torch.no_grad():
            self.base_layer.weight.data -= self.delta_weight()
        self.merged = False

    def trainable_parameters(self) -> int:
        """Return total number of trainable parameters in this layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Return total number of parameters (frozen + trainable) in this layer."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self.merged}"
        )


def verify_lora_layer() -> None:
    """Self-verification of LoRALinear behavior, initialization, and weight merging."""
    torch.manual_seed(42)
    d, k, r, alpha = 1024, 1024, 8, 16.0

    layer = LoRALinear(in_features=k, out_features=d, r=r, alpha=alpha, bias=True)
    layer.eval()
    x = torch.randn(4, k)

    # 1. At initialization (B = 0), output must match base_layer(x) identically
    with torch.no_grad():
        y_init = layer(x)
        y_base = layer.base_layer(x)
        init_match = torch.allclose(y_init, y_base, atol=1e-6)

    # 2. Simulate trained non-zero B and verify unmerged vs merged output equivalence
    with torch.no_grad():
        assert layer.lora_B is not None
        layer.lora_B.normal_(mean=0.0, std=0.05)
        y_1 = layer(x)  # Unmerged: x @ W_0^T + (alpha/r) * x @ (B @ A)^T
        layer.merge_weights()
        y_2 = layer(x)  # Merged:   x @ (W_0 + (alpha/r) * B @ A)^T

    max_diff = (y_1 - y_2).abs().max().item()
    merge_match = torch.allclose(y_1, y_2, atol=1e-5)

    # 3. Verify unmerge restores base weights
    layer.unmerge_weights()
    with torch.no_grad():
        y_unmerged = layer(x)
        unmerge_match = torch.allclose(y_1, y_unmerged, atol=1e-6)

    print("=== LoRALinear Self-Verification (1024 x 1024, r=8, alpha=16) ===")
    print(f"Trainable parameters : {layer.trainable_parameters():,}")
    print(f"Total parameters     : {layer.total_parameters():,}")
    print(f"Base weight frozen   : {not layer.base_layer.weight.requires_grad}")
    print(f"Init DeltaW == 0     : {init_match}")
    print(f"torch.allclose(y_1, y_2, atol=1e-5): {merge_match} (max diff = {max_diff:.2e})")
    print(f"Unmerge equivalence  : {unmerge_match}")

    if not (init_match and merge_match and unmerge_match):
        raise AssertionError("LoRALinear verification failed!")


if __name__ == "__main__":
    verify_lora_layer()
