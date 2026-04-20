"""
scripts/train.py
================
Fine-tunes a YOLOv8 model on the helmet detection dataset.

Usage:
    python scripts/train.py                        # default settings
    python scripts/train.py --epochs 100 --batch 16
    python scripts/train.py --model yolov8m.pt    # larger model
    python scripts/train.py --resume               # resume last run
"""

import argparse
import sys
from pathlib import Path

# ── Make sure project root is on sys.path ────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 helmet detector")
    p.add_argument(
        "--model",
        default="yolov8n.pt",         # nano = fastest; options: n / s / m / l / x
        help="YOLOv8 pre-trained weights (downloaded automatically if absent)",
    )
    p.add_argument(
        "--data",
        default=str(ROOT / "config" / "dataset.yaml"),
        help="Path to dataset.yaml",
    )
    p.add_argument("--epochs",    type=int,   default=50,   help="Training epochs")
    p.add_argument("--imgsz",     type=int,   default=640,  help="Input image size")
    p.add_argument("--batch",     type=int,   default=16,   help="Batch size (-1 = auto)")
    p.add_argument("--workers",   type=int,   default=4,    help="DataLoader workers")
    p.add_argument("--device",    default="",               help="cuda device(s) or 'cpu'")
    p.add_argument("--project",   default=str(ROOT / "runs" / "train"), help="Save dir")
    p.add_argument("--name",      default="helmet_detector", help="Experiment name")
    p.add_argument("--resume",    action="store_true",      help="Resume last training")
    p.add_argument("--lr0",       type=float, default=0.01, help="Initial learning rate")
    p.add_argument("--patience",  type=int,   default=20,   help="Early-stop patience")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Device check
# ─────────────────────────────────────────────────────────────

def check_device(requested: str) -> str:
    """Choose best available device."""
    if requested:
        return requested
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu}  |  VRAM: {vram:.1f} GB")
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[MPS] Apple Silicon GPU detected.")
        return "mps"
    print("[CPU] No GPU found — training will be slow. Consider Google Colab for free GPU.")
    return "cpu"


# ─────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    device = check_device(args.device)

    print("\n" + "=" * 55)
    print("  Helmet Detection — Training")
    print("=" * 55)
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Device  : {device}")
    print("=" * 55 + "\n")

    # ── Load model ───────────────────────────────────────────
    if args.resume:
        # Find last checkpoint automatically
        last_ckpt = ROOT / "runs" / "train" / args.name / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[RESUME] Loading checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"[WARN] No checkpoint found at {last_ckpt}. Starting fresh.")
            model = YOLO(args.model)
    else:
        # Download pre-trained weights from Ultralytics (COCO-trained)
        print(f"[INFO] Loading base model: {args.model}")
        model = YOLO(args.model)

    # ── Train ────────────────────────────────────────────────
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,          # overwrite experiment folder
        pretrained=True,        # use COCO pre-trained weights
        optimizer="AdamW",      # works well for fine-tuning
        lr0=args.lr0,
        lrf=0.01,               # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=args.patience, # early stopping
        save=True,
        save_period=10,         # save checkpoint every N epochs
        val=True,
        plots=True,             # save training plots
        # Augmentation (helps with small datasets)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        verbose=True,
    )

    # ── Post-training summary ─────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("  Training Complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Results dir  : {Path(args.project) / args.name}")
    print("=" * 55)
    print("\n[NEXT] Run inference:")
    print(f"  python scripts/detect.py --weights {best_weights} --source 0")
    print(f"  python scripts/detect.py --weights {best_weights} --source video.mp4")

    return str(best_weights)


# ─────────────────────────────────────────────────────────────
# Validation helper (run after training)
# ─────────────────────────────────────────────────────────────

def validate(weights: str, data: str):
    """Quick validation run on the val split."""
    print("\n[Validation] Running validation...")
    model = YOLO(weights)
    metrics = model.val(data=data)
    print(f"\n  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    best = train()
    # Optionally run validation right after training
    # validate(best, str(ROOT / "config" / "dataset.yaml"))
