"""Synthetic low-rank adaptation benchmark comparing Full FT vs LoRA (r=4, 8, 16).

Demonstrates empirically that when the true task shift Delta W* has low intrinsic
rank (r* = 8), LoRA with r >= r* converges cleanly to the target mapping while
training far fewer parameters, whereas r < r* exhibits capacity bottlenecking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lora_layer import LoRALinear

FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def _copy_base_weights(target: nn.Linear, source: nn.Linear) -> None:
    """Initialize target linear layer with exact weights of source."""
    with torch.no_grad():
        target.weight.copy_(source.weight)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)


def run_synthetic_experiment(
    dim: int = 256,
    true_rank: int = 8,
    steps: int = 150,
    lr: float = 1e-2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Full FT and LoRA models on a synthetic rank-8 adaptation problem."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrained base linear layer W_0 and low-rank target shift Delta W* = U @ V
    base_linear = nn.Linear(dim, dim, bias=True).to(device)
    with torch.no_grad():
        u_true = torch.randn(dim, true_rank, device=device) * 0.25
        v_true = torch.randn(true_rank, dim, device=device) * 0.25
        noise = torch.randn(dim, dim, device=device) * 1e-3
        target_weight = base_linear.weight + (u_true @ v_true) + noise
        target_bias = base_linear.bias.clone()

    # Synthetic train and validation datasets
    x_train = torch.randn(512, dim, device=device)
    y_train = x_train @ target_weight.T + target_bias
    x_val = torch.randn(256, dim, device=device)
    y_val = x_val @ target_weight.T + target_bias

    # Build candidate models
    full_model = nn.Linear(dim, dim, bias=True).to(device)
    _copy_base_weights(full_model, base_linear)

    candidates: List[tuple[str, nn.Module]] = [("Full Fine-tuning", full_model)]
    for r in (4, 8, 16):
        lora_model = LoRALinear.from_linear(
            base_linear, r=r, alpha=float(2 * r), lora_dropout=0.0
        ).to(device)
        candidates.append((f"LoRA (r={r})", lora_model))

    criterion = nn.MSELoss()
    runs_summary: List[Dict[str, Any]] = []

    for name, model in candidates:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

        train_losses: List[float] = []
        val_losses: List[float] = []

        t0 = time.perf_counter()
        for _ in range(steps):
            model.train()
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_val), y_val)

            train_losses.append(float(loss.item()))
            val_losses.append(float(val_loss.item()))
        elapsed = time.perf_counter() - t0

        merge_verified = None
        if isinstance(model, LoRALinear):
            model.eval()
            with torch.no_grad():
                y_unmerged = model(x_val)
                model.merge_weights()
                y_merged = model(x_val)
                merge_verified = bool(torch.allclose(y_unmerged, y_merged, atol=1e-5))
                model.unmerge_weights()

        runs_summary.append(
            {
                "name": name,
                "trainable_params": num_trainable,
                "total_params": num_total,
                "trainable_pct": round((num_trainable / num_total) * 100.0, 2),
                "elapsed_sec": round(elapsed, 4),
                "initial_val_mse": round(val_losses[0], 6),
                "final_train_mse": round(train_losses[-1], 6),
                "final_val_mse": round(val_losses[-1], 6),
                "merge_verified": merge_verified,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
        )

    return {
        "config": {
            "dim": dim,
            "true_rank": true_rank,
            "steps": steps,
            "lr": lr,
            "device": str(device),
        },
        "runs": runs_summary,
    }


def plot_convergence(results: Dict[str, Any], output_path: Path) -> None:
    """Plot validation MSE convergence curves for Full FT and LoRA ranks."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    palette = {
        "Full Fine-tuning": ("#374151", "--"),
        "LoRA (r=4)": ("#E07A5F", "-"),
        "LoRA (r=8)": ("#2A9D8F", "-"),
        "LoRA (r=16)": ("#2563EB", "-"),
    }

    for run in results["runs"]:
        name = run["name"]
        color, style = palette.get(name, ("#111827", "-"))
        label = (
            f"{name} ({run['trainable_params']:,} params, "
            f"Final MSE={run['final_val_mse']:.2e})"
        )
        steps = list(range(1, len(run["val_losses"]) + 1))
        ax.plot(steps, run["val_losses"], label=label, color=color, linestyle=style, linewidth=2.1)

    ax.set_yscale("log")
    ax.set_xlabel("Bước huấn luyện (Training Step)", fontweight="bold")
    ax.set_ylabel("Validation MSE Loss (Log Scale)", fontweight="bold")
    ax.set_title(
        f"Hội tụ hàm mất mát trên bài toán thích ứng hạng nội tại r* = {results['config']['true_rank']}",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.2, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_synthetic_experiment()

    print("=== SYNTHETIC LOW-RANK ADAPTATION RESULTS (256x256, r*=8) ===")
    for run in results["runs"]:
        merge_note = (
            f" | merge_ok={run['merge_verified']}"
            if run["merge_verified"] is not None
            else ""
        )
        print(
            f"{run['name']:<18} | Trainable: {run['trainable_params']:>7,} "
            f"({run['trainable_pct']:>5.2f}%) | Final Val MSE: {run['final_val_mse']:.6f} "
            f"| Time: {run['elapsed_sec']:.3f}s{merge_note}"
        )

    fig_path = FIGURES_DIR / "loss_convergence.png"
    plot_convergence(results, fig_path)

    # Save compact metrics JSON (truncate full per-step arrays for clean readability)
    compact = {
        "config": results["config"],
        "runs": [
            {k: v for k, v in run.items() if k not in ("train_losses", "val_losses")}
            for run in results["runs"]
        ],
    }
    json_path = RESULTS_DIR / "synthetic_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    print(f"\nSaved convergence plot: {fig_path}")
    print(f"Saved metrics JSON:     {json_path}")


if __name__ == "__main__":
    main()
