"""Mathematical verification of LoRA parameter reduction and 7B VRAM breakdown.

Computes exact trainable parameter counts for a 1024x1024 linear layer
across ranks r in {4, 8, 16}, prints Markdown and LaTeX tables, estimates
theoretical 7B LLM VRAM footprints (Full FT vs LoRA vs QLoRA), and exports
publication-quality figures plus JSON summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def _format_table(
    rows: List[List[Any]], headers: List[str], tablefmt: str = "github"
) -> str:
    """Format tabular data using tabulate if available, else a clean fallback."""
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt=tablefmt)

    sep = " | " if tablefmt != "latex" else " & "
    lines = [sep.join(str(h) for h in headers)]
    if tablefmt == "github":
        lines.append(sep.join("---" for _ in headers))
    for row in rows:
        lines.append(sep.join(str(cell) for cell in row))
    return "\n".join(lines)


def analyze_linear_layer(
    d: int = 1024,
    k: int = 1024,
    ranks: Iterable[int] = (4, 8, 16),
) -> List[Dict[str, Any]]:
    """Compute exact parameter counts and reduction ratios for a d x k linear layer."""
    full_params = d * k
    results: List[Dict[str, Any]] = [
        {
            "method": "Full Fine-tuning",
            "rank": None,
            "formula": f"{d} x {k}",
            "trainable_params": full_params,
            "trainable_pct": 100.0,
            "reduction_pct": 0.0,
        }
    ]

    for r in ranks:
        lora_params = r * (d + k)
        trainable_pct = (lora_params / full_params) * 100.0
        reduction_pct = ((full_params - lora_params) / full_params) * 100.0
        results.append(
            {
                "method": f"LoRA (r={r})",
                "rank": r,
                "formula": f"{r} x ({d} + {k})",
                "trainable_params": lora_params,
                "trainable_pct": round(trainable_pct, 4),
                "reduction_pct": round(reduction_pct, 2),
            }
        )

    return results


def estimate_7b_vram_breakdown(
    num_params_billions: float = 7.0,
    lora_trainable_fraction: float = 0.005,
    activations_gb: float = 6.0,
) -> List[Dict[str, Any]]:
    """Estimate theoretical VRAM breakdown (GB, 1 GB = 1e9 bytes) for a 7B model.

    Demonstrates why a ~99% drop in trainable parameters does not equal a 99%
    drop in GPU VRAM: base model weights and forward activations remain resident.
    """
    p = num_params_billions * 1e9
    gb = 1e9
    trainable_p = p * lora_trainable_fraction

    # 1. Full Fine-tuning (FP16 weights & gradients, FP32 AdamW m_t and v_t)
    full_weights = (p * 2) / gb          # 14.0 GB
    full_grads = (p * 2) / gb            # 14.0 GB
    full_optim = (p * 8) / gb            # 56.0 GB (2 * 4 bytes * 7B)

    # 2. LoRA (FP16 frozen base + trainable adapters)
    lora_weights = (p * 2 + trainable_p * 2) / gb
    lora_grads = (trainable_p * 2) / gb
    lora_optim = (trainable_p * 8) / gb

    # 3. QLoRA (NF4 4-bit base + Double Quantization ~4.127 bits/param + BF16 adapters)
    bits_per_param_nf4_dq = 4.0 + (8.0 / 64.0) + (32.0 / (64.0 * 256.0))  # ~4.127 bits
    qlora_base_bytes = p * (bits_per_param_nf4_dq / 8.0)
    qlora_weights = (qlora_base_bytes + trainable_p * 2) / gb
    qlora_grads = (trainable_p * 2) / gb
    qlora_optim = (trainable_p * 8) / gb

    configs = [
        ("Full Fine-tuning (FP16)", full_weights, full_grads, full_optim, activations_gb),
        ("LoRA (FP16, 0.5% trainable)", lora_weights, lora_grads, lora_optim, activations_gb),
        ("QLoRA (NF4 + Double Quant)", qlora_weights, qlora_grads, qlora_optim, activations_gb),
    ]

    breakdown: List[Dict[str, Any]] = []
    full_total = full_weights + full_grads + full_optim + activations_gb
    for name, w, g, o, act in configs:
        total = w + g + o + act
        vram_saved_pct = ((full_total - total) / full_total) * 100.0
        breakdown.append(
            {
                "config": name,
                "weights_gb": round(w, 2),
                "gradients_gb": round(g, 2),
                "optimizer_states_gb": round(o, 2),
                "activations_gb": round(act, 2),
                "total_vram_gb": round(total, 2),
                "vram_saved_vs_full_pct": round(vram_saved_pct, 2),
            }
        )
    return breakdown


def plot_parameter_comparison(
    param_rows: List[Dict[str, Any]], output_path: Path
) -> None:
    """Save bar chart comparing trainable parameters and reduction percentages."""
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#CCCCCC"})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    labels = [row["method"] for row in param_rows]
    counts = [row["trainable_params"] for row in param_rows]
    reductions = [row["reduction_pct"] for row in param_rows]
    colors = ["#4C566A", "#2E86AB", "#48A9A6", "#E07A5F"]

    bars = ax.bar(labels, counts, color=colors[: len(labels)], width=0.55, edgecolor="#222222", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_ylim(1_000, 4_000_000)
    ax.set_ylabel("Số lượng tham số huấn luyện (Log Scale)", fontweight="bold")
    ax.set_title(
        "So sánh số lượng tham số huấn luyện trên lớp tuyến tính 1024 × 1024",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count, red in zip(bars, counts, reductions):
        annotation = f"{count:,}" if red == 0 else f"{count:,}\n(Giảm {red:.2f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.18,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2933",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_vram_breakdown(
    vram_rows: List[Dict[str, Any]], output_path: Path
) -> None:
    """Save stacked bar chart of 7B VRAM footprint components."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    labels = [r["config"] for r in vram_rows]
    weights = np.array([r["weights_gb"] for r in vram_rows])
    grads = np.array([r["gradients_gb"] for r in vram_rows])
    optim = np.array([r["optimizer_states_gb"] for r in vram_rows])
    acts = np.array([r["activations_gb"] for r in vram_rows])

    x = np.arange(len(labels))
    width = 0.5

    ax.bar(x, weights, width, label="Trọng số mô hình (Weights)", color="#3D5A80")
    ax.bar(x, grads, width, bottom=weights, label="Đạo hàm (Gradients)", color="#98C1D9")
    ax.bar(
        x,
        optim,
        width,
        bottom=weights + grads,
        label="Trạng thái AdamW (FP32 m_t, v_t)",
        color="#EE6C4D",
    )
    ax.bar(
        x,
        acts,
        width,
        bottom=weights + grads + optim,
        label="Kích hoạt trung gian (Activations)",
        color="#F4A261",
    )

    totals = weights + grads + optim + acts
    for idx, (tot, row) in enumerate(zip(totals, vram_rows)):
        saved = row["vram_saved_vs_full_pct"]
        text = f"{tot:.2f} GB" if saved == 0 else f"{tot:.2f} GB\n(-{saved:.1f}%)"
        ax.text(idx, tot + 1.8, text, ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.axhline(24, color="#2A9D8F", linestyle="--", linewidth=1.2, label="Ngưỡng GPU 24 GB (RTX 3090/4090)")
    ax.axhline(80, color="#B5179E", linestyle=":", linewidth=1.2, label="Ngưỡng GPU 80 GB (A100)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Ước lượng bộ nhớ VRAM (GB)", fontweight="bold")
    ax.set_title(
        "Phân rã VRAM lý thuyết cho mô hình 7B: Vì sao giảm 99% tham số ≠ giảm 99% VRAM",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="upper right", frameon=True, fontsize=8.8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    param_rows = analyze_linear_layer(d=1024, k=1024, ranks=(4, 8, 16))
    vram_rows = estimate_7b_vram_breakdown()

    param_headers = ["Phương pháp", "Công thức", "Tham số huấn luyện", "Tỷ lệ giữ lại (%)", "Tỷ lệ giảm (%)"]
    param_table_data = [
        [
            r["method"],
            r["formula"],
            f"{r['trainable_params']:,}",
            f"{r['trainable_pct']:.2f}%",
            f"{r['reduction_pct']:.2f}%",
        ]
        for r in param_rows
    ]

    vram_headers = ["Cấu hình (7B)", "Weights (GB)", "Gradients (GB)", "Optimizer (GB)", "Activations (GB)", "Tổng (GB)", "Giảm VRAM (%)"]
    vram_table_data = [
        [
            r["config"],
            f"{r['weights_gb']:.2f}",
            f"{r['gradients_gb']:.2f}",
            f"{r['optimizer_states_gb']:.2f}",
            f"{r['activations_gb']:.2f}",
            f"{r['total_vram_gb']:.2f}",
            f"{r['vram_saved_vs_full_pct']:.2f}%",
        ]
        for r in vram_rows
    ]

    print("=== BANG 1: SO SANH THAM SO LOP TUYEN TINH 1024 x 1024 (MARKDOWN) ===")
    print(_format_table(param_table_data, param_headers, tablefmt="github"))
    print("\n=== BANG 1: DINH DANG LATEX ===")
    print(_format_table(param_table_data, param_headers, tablefmt="latex"))

    print("\n=== BANG 2: PHAN RA VRAM MO HINH 7B (MARKDOWN) ===")
    print(_format_table(vram_table_data, vram_headers, tablefmt="github"))

    params_fig = FIGURES_DIR / "params_comparison.png"
    vram_fig = FIGURES_DIR / "vram_breakdown.png"
    plot_parameter_comparison(param_rows, params_fig)
    plot_vram_breakdown(vram_rows, vram_fig)

    json_path = RESULTS_DIR / "parameter_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "linear_1024x1024": param_rows,
                "vram_7b_breakdown": vram_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved figures: {params_fig}, {vram_fig}")
    print(f"Saved JSON:    {json_path}")


if __name__ == "__main__":
    main()
