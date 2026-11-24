"""DistilBERT SST-2 benchmark comparing Full Fine-tuning vs PEFT LoRA (r=4, r=16).

Runs cleanly on CPU (Windows PyCharm) or CUDA GPU, supporting both Hugging Face
GLUE/SST-2 online loading and an offline balanced CSV fallback (data/sst2_sample.csv).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "sst2_sample.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def _get_peak_memory_mb(device: torch.device) -> Tuple[float, str]:
    """Return (memory_mb, label) using CUDA peak VRAM on GPU or process peak RSS on CPU."""
    if device.type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2), "Peak CUDA VRAM (MB)"

    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 ** 2), 2), "Peak Process RSS (MB)"

    return 0.0, "Peak Memory (MB)"


def _load_sst2_splits(
    offline: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seed: int,
) -> Tuple[Dataset, Dataset, str]:
    """Load train/validation splits from local CSV or Hugging Face glue/sst2."""
    if not offline:
        try:
            ds = load_dataset("glue", "sst2")
            train_ds = ds["train"].shuffle(seed=seed)
            eval_ds = ds["validation"]
            if max_train_samples is not None:
                train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
            if max_eval_samples is not None:
                eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
            return train_ds, eval_ds, "glue/sst2"
        except Exception as exc:
            print(f"[WARN] Could not load glue/sst2 online ({exc}); falling back to {DATA_PATH}")

    raw = load_dataset("csv", data_files=str(DATA_PATH))["train"]
    indices = list(range(len(raw)))
    labels = raw["label"]
    train_idx, eval_idx = train_test_split(
        indices, test_size=1 / 3, stratify=labels, random_state=seed
    )
    train_ds = raw.select(train_idx)
    eval_ds = raw.select(eval_idx)
    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds, f"offline:{DATA_PATH.name}"


def _prepare_dataloaders(
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: Any,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Tokenize datasets and wrap them in PyTorch DataLoaders."""

    def tokenize_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    cols_to_remove = [c for c in train_ds.column_names if c not in ("label",)]
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    train_tok = train_tok.rename_column("label", "labels")
    eval_tok = eval_tok.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_tok, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, eval_loader


def _build_model(
    model_name: str, lora_rank: int | None
) -> Tuple[torch.nn.Module, str, float]:
    """Create either a full fine-tuning model or a PEFT LoRA-wrapped model."""
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if lora_rank is None:
        return base_model, "Full Fine-tuning", 0.0

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    return peft_model, f"LoRA (r={lora_rank}, alpha={2 * lora_rank})", float(2 * lora_rank)


def evaluate_model(
    model: torch.nn.Module, eval_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluate classification loss, accuracy, and macro F1 on eval_loader."""
    model.eval()
    preds: List[int] = []
    targets: List[int] = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            steps += 1
            batch_preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            batch_labels = batch["labels"].cpu().tolist()
            preds.extend(batch_preds)
            targets.extend(batch_labels)

    return {
        "eval_loss": round(total_loss / max(steps, 1), 4),
        "accuracy": round(float(accuracy_score(targets, preds)), 4),
        "f1": round(float(f1_score(targets, preds, zero_division=0)), 4),
    }


def train_single_config(
    model_name: str,
    lora_rank: int | None,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    grad_accum: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    """Run training and evaluation for one fine-tuning configuration."""
    torch.manual_seed(seed)
    model, label, _ = _build_model(model_name, lora_rank)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = round((trainable_params / total_params) * 100.0, 4)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_optim_steps = max((len(train_loader) * epochs) // max(grad_accum, 1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_optim_steps // 10, 1),
        num_training_steps=total_optim_steps,
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    epoch_losses: List[float] = []
    epoch_accs: List[float] = []

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            running_loss += float(raw_loss.item())

            scaled_loss = raw_loss / grad_accum
            scaled_loss.backward()

            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = round(running_loss / max(len(train_loader), 1), 4)
        eval_metrics = evaluate_model(model, eval_loader, device)
        epoch_losses.append(avg_train_loss)
        epoch_accs.append(eval_metrics["accuracy"])
        print(
            f"[{label}] Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Eval Loss: {eval_metrics['eval_loss']:.4f} | Acc: {eval_metrics['accuracy']:.4f} | "
            f"F1: {eval_metrics['f1']:.4f}"
        )

    elapsed = round(time.perf_counter() - t0, 2)
    peak_mem_mb, mem_label = _get_peak_memory_mb(device)
    final_eval = evaluate_model(model, eval_loader, device)

    return {
        "method": label,
        "lora_rank": lora_rank,
        "learning_rate": lr,
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_pct": trainable_pct,
        "elapsed_sec": elapsed,
        "peak_memory_mb": peak_mem_mb,
        "memory_metric_label": mem_label,
        "final_eval_loss": final_eval["eval_loss"],
        "final_accuracy": final_eval["accuracy"],
        "final_f1": final_eval["f1"],
        "epoch_train_losses": epoch_losses,
        "epoch_eval_accuracies": epoch_accs,
    }


def plot_hf_results(runs: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot training loss convergence and accuracy vs trainable parameter ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    colors = ["#374151", "#2A9D8F", "#2563EB"]

    # Panel A: Epoch training loss
    ax1 = axes[0]
    for run, color in zip(runs, colors):
        epochs = list(range(1, len(run["epoch_train_losses"]) + 1))
        ax1.plot(
            epochs,
            run["epoch_train_losses"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{run['method']} ({run['trainable_pct']:.2f}%)",
        )
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Training Loss", fontweight="bold")
    ax1.set_title("Hội tụ Training Loss (DistilBERT)", fontweight="bold")
    ax1.set_xticks(list(range(1, len(runs[0]["epoch_train_losses"]) + 1)))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Eval Accuracy & Trainable Parameter %
    ax2 = axes[1]
    labels = [r["method"] for r in runs]
    accs = [r["final_accuracy"] * 100.0 for r in runs]
    bars = ax2.bar(labels, accs, color=colors[: len(runs)], width=0.5, edgecolor="#222222")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax2.set_title("Độ chính xác & Tỷ lệ tham số huấn luyện", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar, run in zip(bars, runs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{run['final_accuracy']*100:.1f}%\n({run['trainable_pct']:.2f}% params)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DistilBERT Full FT vs LoRA SST-2 Benchmark")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test preset (80 train / 40 eval / 2 epochs)")
    parser.add_argument("--offline", action="store_true", help="Use local data/sst2_sample.csv instead of downloading glue/sst2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training examples")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Cap validation examples")
    parser.add_argument("--lr-full", type=float, default=2e-5, help="Learning rate for Full Fine-tuning")
    parser.add_argument("--lr-lora", type=float, default=3e-4, help="Learning rate for LoRA")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased", help="HF model identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.epochs = 2
        args.max_train_samples = args.max_train_samples or 80
        args.max_eval_samples = args.max_eval_samples or 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DistilBERT PEFT experiment on device: {device}")

    train_ds, eval_ds, dataset_source = _load_sst2_splits(
        offline=args.offline,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )
    print(f"Dataset source: {dataset_source} | Train={len(train_ds)} | Eval={len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, eval_loader = _prepare_dataloaders(
        train_ds, eval_ds, tokenizer, batch_size=args.batch_size
    )

    configs = [
        (None, args.lr_full),
        (4, args.lr_lora),
        (16, args.lr_lora),
    ]

    runs: List[Dict[str, Any]] = []
    for rank, lr in configs:
        run_res = train_single_config(
            model_name=args.model_name,
            lora_rank=rank,
            train_loader=train_loader,
            eval_loader=eval_loader,
            epochs=args.epochs,
            grad_accum=args.grad_accum,
            lr=lr,
            device=device,
            seed=args.seed,
        )
        runs.append(run_res)

    print("\n=== TONG HOP KET QUA THUC NGHIEM DISTILBERT (SST-2) ===")
    for r in runs:
        print(
            f"{r['method']:<24} | Trainable: {r['trainable_params']:>10,} ({r['trainable_pct']:>6.2f}%) | "
            f"Acc: {r['final_accuracy']:.4f} | F1: {r['final_f1']:.4f} | "
            f"Time: {r['elapsed_sec']:.2f}s | {r['memory_metric_label']}: {r['peak_memory_mb']:.1f}"
        )

    fig_path = FIGURES_DIR / "hf_peft_convergence.png"
    plot_hf_results(runs, fig_path)

    payload = {
        "environment": {
            "device": str(device),
            "model_name": args.model_name,
            "dataset_source": dataset_source,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
        },
        "runs": runs,
    }
    json_path = RESULTS_DIR / "hf_peft_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved figure: {fig_path}")
    print(f"Saved JSON:   {json_path}")


if __name__ == "__main__":
    main()
