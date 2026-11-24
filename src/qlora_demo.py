"""QLoRA 4-bit NormalFloat (NF4) & Double Quantization simulation + HF config helper.

Simulates the three core QLoRA innovations from Dettmers et al. (NeurIPS 2023)
on Windows/CPU without requiring Linux-specific bitsandbytes CUDA kernels, and
provides a ready-to-use BitsAndBytesConfig + LoraConfig builder for Colab/Linux GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# Exact 16 asymmetric NormalFloat4 (NF4) quantization levels from Dettmers et al. (2023)
NF4_LEVELS = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230182647705,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


def quantize_nf4_blockwise(
    weight: torch.Tensor,
    blocksize: int = 64,
    double_quant: bool = True,
    dq_blocksize: int = 256,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Simulate blockwise NF4 quantization and optional Double Quantization.

    Args:
        weight: Input floating-point weight matrix.
        blocksize: Number of weights sharing one absmax scaling constant c1.
        double_quant: Whether to quantize c1 constants into 8-bit values.
        dq_blocksize: Number of c1 constants sharing one second-level scale c2.

    Returns:
        (dequantized_weight, telemetry_dict)
    """
    orig_shape = weight.shape
    flat = weight.detach().to(torch.float32).reshape(-1)
    n_elems = flat.numel()

    pad = (blocksize - (n_elems % blocksize)) % blocksize
    if pad > 0:
        flat = F.pad(flat, (0, pad))

    blocks = flat.view(-1, blocksize)
    absmax_c1 = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

    if double_quant:
        # Simulate quantizing FP32 c1 constants to unsigned 8-bit integers per dq_blocksize
        num_c1 = absmax_c1.numel()
        dq_pad = (dq_blocksize - (num_c1 % dq_blocksize)) % dq_blocksize
        c1_flat = F.pad(absmax_c1.reshape(-1), (0, dq_pad))
        c1_blocks = c1_flat.view(-1, dq_blocksize)
        c2_scale = c1_blocks.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        c1_q = torch.round((c1_blocks / c2_scale) * 255.0).clamp(0, 255)
        c1_deq = (c1_q / 255.0) * c2_scale
        effective_c1 = c1_deq.reshape(-1)[:num_c1].view_as(absmax_c1)
    else:
        effective_c1 = absmax_c1

    normalized = blocks / effective_c1
    # Map each normalized value in [-1, 1] to nearest NF4 codebook entry (4-bit index 0..15)
    levels = NF4_LEVELS.to(flat.device)
    dists = (normalized.unsqueeze(-1) - levels).abs()
    q_indices = dists.argmin(dim=-1)  # 4-bit indices (stored as uint8 pair in practice)

    # Dequantize back to FP32 for error evaluation
    deq_blocks = levels[q_indices] * effective_c1
    deq_flat = deq_blocks.reshape(-1)[:n_elems]
    deq_weight = deq_flat.view(orig_shape)

    # Memory footprint accounting (bits per parameter)
    fp16_bits = 16.0
    if double_quant:
        # 4-bit payload + 8-bit c1 per blocksize + 32-bit c2 per (blocksize * dq_blocksize)
        bits_per_param = 4.0 + (8.0 / blocksize) + (32.0 / (blocksize * dq_blocksize))
    else:
        # 4-bit payload + 32-bit c1 per blocksize
        bits_per_param = 4.0 + (32.0 / blocksize)

    mse = F.mse_loss(deq_weight, weight.to(torch.float32)).item()
    rmse = mse ** 0.5
    cos_sim = F.cosine_similarity(
        deq_weight.reshape(1, -1), weight.to(torch.float32).reshape(1, -1)
    ).item()

    stats = {
        "shape": list(orig_shape),
        "num_params": int(n_elems),
        "blocksize": blocksize,
        "double_quant": double_quant,
        "bits_per_param": round(bits_per_param, 4),
        "compression_vs_fp16": round(fp16_bits / bits_per_param, 2),
        "compression_vs_fp32": round(32.0 / bits_per_param, 2),
        "mse": mse,
        "rmse": rmse,
        "cosine_similarity": round(cos_sim, 6),
    }
    return deq_weight, stats


def build_qlora_peft_config(r: int = 8, lora_alpha: int = 16) -> Dict[str, Any]:
    """Build Hugging Face BitsAndBytesConfig and LoraConfig for QLoRA when supported."""
    status: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "bitsandbytes_available": False,
        "bnb_config": None,
        "lora_config": None,
        "note": "",
    }

    try:
        import bitsandbytes  # noqa: F401
        from peft import LoraConfig
        from transformers import BitsAndBytesConfig

        status["bitsandbytes_available"] = True
        if torch.cuda.is_available():
            status["bnb_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            status["lora_config"] = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
            status["note"] = "QLoRA hardware config ready (CUDA + bitsandbytes)."
        else:
            status["note"] = (
                "bitsandbytes detected but CUDA GPU is unavailable; "
                "use Google Colab/Linux GPU for live 4-bit kernel execution."
            )
    except Exception as exc:
        status["note"] = (
            f"Hardware QLoRA skipped ({type(exc).__name__}: {exc}). "
            "Running pure-PyTorch NF4 simulation on CPU (recommended on Windows)."
        )

    return status


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    # Gaussian pretrained weight matrix 1024 x 1024 (typical Transformer projection)
    weight = torch.randn(1024, 1024, dtype=torch.float32) * 0.02

    _, stats_single = quantize_nf4_blockwise(weight, blocksize=64, double_quant=False)
    _, stats_double = quantize_nf4_blockwise(weight, blocksize=64, double_quant=True, dq_blocksize=256)
    hw_status = build_qlora_peft_config()

    print("=== QLORA NF4 QUANTIZATION SIMULATION (1024 x 1024 Gaussian Weight) ===")
    print(f"Environment check: {hw_status['note']}\n")
    for label, st in [("NF4 (Single Quant)", stats_single), ("NF4 + Double Quant", stats_double)]:
        print(
            f"{label:<20} | {st['bits_per_param']:.3f} bits/param | "
            f"Compression vs FP16: {st['compression_vs_fp16']:.2f}x | "
            f"RMSE: {st['rmse']:.6f} | Cosine Sim: {st['cosine_similarity']:.6f}"
        )

    saved_bits = stats_single["bits_per_param"] - stats_double["bits_per_param"]
    print(f"\nDouble Quantization savings: {saved_bits:.3f} bits/param (~0.373 bpp per Dettmers et al., 2023)")

    out_path = RESULTS_DIR / "qlora_simulation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "nf4_single_quant": stats_single,
                "nf4_double_quant": stats_double,
                "double_quant_savings_bpp": round(saved_bits, 4),
                "hardware_environment": {
                    "cuda_available": hw_status["cuda_available"],
                    "bitsandbytes_available": hw_status["bitsandbytes_available"],
                    "note": hw_status["note"],
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved QLoRA simulation JSON: {out_path}")


if __name__ == "__main__":
    main()
