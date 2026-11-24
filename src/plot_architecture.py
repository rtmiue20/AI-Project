"""Programmatic diagram generator for the LoRA architecture (W_0 + (alpha/r) * B @ A).

Produces figures/lora_architecture.png at 300 DPI for inclusion in the UEH report
and seminar presentation slides.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    textcolor: str = "#111827",
    fontsize: float = 10.5,
    linestyle: str = "-",
) -> None:
    """Draw a rounded rectangle with centered label."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=textcolor,
    )


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#374151",
) -> None:
    """Draw a clean straight arrow between two coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=1.7,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def plot_lora_architecture(output_path: Path) -> None:
    """Render the LoRA forward computation and inference weight-merge diagram."""
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Title
    ax.text(
        6.0,
        9.05,
        "Kiến trúc LoRA (Low-Rank Adaptation) & Cơ chế gộp trọng số suy luận",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#111827",
    )

    # Input node x (bottom center)
    _draw_box(
        ax,
        x=4.8,
        y=1.6,
        w=2.4,
        h=0.75,
        text="Đầu vào  $x \\in \\mathbb{R}^{k}$",
        facecolor="#F3F4F6",
        edgecolor="#4B5563",
        fontsize=11,
    )

    # Left branch: Frozen W_0
    _draw_box(
        ax,
        x=1.2,
        y=3.9,
        w=3.6,
        h=1.9,
        text=(
            "Trọng số gốc đóng băng\n"
            "$W_0 \\in \\mathbb{R}^{d \\times k}$\n"
            "(requires_grad = False)"
        ),
        facecolor="#DCEEFB",
        edgecolor="#2B6CB0",
        textcolor="#1E3A8A",
        fontsize=10.5,
    )

    # Right branch: Trainable A -> B -> scaling
    _draw_box(
        ax,
        x=7.2,
        y=3.1,
        w=3.5,
        h=1.05,
        text="Ma trận $A \\in \\mathbb{R}^{r \\times k}$\nKhởi tạo $\\mathcal{N}(0, \\sigma^2)$ (Trainable)",
        facecolor="#FFEDD5",
        edgecolor="#C2410C",
        textcolor="#7C2D12",
        fontsize=10,
    )
    _draw_box(
        ax,
        x=7.2,
        y=4.85,
        w=3.5,
        h=1.05,
        text="Ma trận $B \\in \\mathbb{R}^{d \\times r}$\nKhởi tạo $= 0$ (Trainable)",
        facecolor="#FFEDD5",
        edgecolor="#C2410C",
        textcolor="#7C2D12",
        fontsize=10,
    )
    _draw_box(
        ax,
        x=7.65,
        y=6.4,
        w=2.6,
        h=0.65,
        text="Hệ số tỷ lệ: $\\times \\frac{\\alpha}{r}$",
        facecolor="#FEF3C7",
        edgecolor="#D97706",
        textcolor="#78350F",
        fontsize=10.5,
    )

    # Sum node (+)
    sum_circle = plt.Circle((6.0, 7.55), 0.42, facecolor="#ECFDF5", edgecolor="#059669", linewidth=2.0)
    ax.add_patch(sum_circle)
    ax.text(6.0, 7.55, "+", ha="center", va="center", fontsize=18, fontweight="bold", color="#065F46")

    # Output node h
    _draw_box(
        ax,
        x=3.5,
        y=8.15,
        w=5.0,
        h=0.65,
        text="Đầu ra  $h = W_0 x + \\frac{\\alpha}{r} (B A) x$",
        facecolor="#ECFDF5",
        edgecolor="#059669",
        textcolor="#065F46",
        fontsize=11.5,
    )

    # Arrows
    _draw_arrow(ax, (4.8, 2.05), (3.0, 3.85), color="#2B6CB0")
    _draw_arrow(ax, (7.2, 2.05), (8.95, 3.05), color="#C2410C")
    _draw_arrow(ax, (8.95, 4.18), (8.95, 4.82), color="#C2410C")
    _draw_arrow(ax, (8.95, 5.93), (8.95, 6.36), color="#D97706")
    _draw_arrow(ax, (3.0, 5.85), (5.58, 7.55), color="#2B6CB0")
    _draw_arrow(ax, (8.95, 7.08), (6.42, 7.55), color="#D97706")
    _draw_arrow(ax, (6.0, 7.97), (6.0, 8.13), color="#059669")

    # Branch badges
    ax.text(1.2, 6.1, "Nhánh đóng băng (Base)", fontsize=9.5, fontweight="bold", color="#1E3A8A")
    ax.text(7.2, 2.7, "Nhánh thích ứng hạng thấp ($r \\ll \\min(d, k)$)", fontsize=9.5, fontweight="bold", color="#9A3412")

    # Bottom Callout: Zero Inference Latency Merge
    _draw_box(
        ax,
        x=1.2,
        y=0.25,
        w=9.5,
        h=0.95,
        text=(
            "Gộp trọng số khi triển khai (Zero Inference Latency):   "
            "$W_{\\text{deploy}} = W_0 + \\frac{\\alpha}{r} B A$   "
            "$\\Rightarrow$ Không tăng độ trễ và bộ nhớ suy luận!"
        ),
        facecolor="#EEF2FF",
        edgecolor="#4338CA",
        textcolor="#312E81",
        fontsize=10.5,
        linestyle="--",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "lora_architecture.png"
    plot_lora_architecture(out)
    print(f"Saved architecture diagram: {out}")


if __name__ == "__main__":
    main()
