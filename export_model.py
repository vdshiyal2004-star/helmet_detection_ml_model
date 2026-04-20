"""
scripts/export_model.py
=======================
Export a trained YOLOv8 .pt model to various deployment formats.

Supported formats:
    onnx      — cross-platform (Triton, ONNX Runtime, TensorRT, etc.)
    torchscript — mobile / C++ deployment
    tflite    — Android / embedded
    coreml    — Apple devices
    engine    — TensorRT (requires NVIDIA GPU + TensorRT)
    openvino  — Intel hardware

Usage:
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format engine --half   # FP16 TensorRT
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


FORMATS = ["onnx", "torchscript", "tflite", "coreml", "engine", "openvino", "saved_model"]


def parse_args():
    p = argparse.ArgumentParser(description="Export trained YOLOv8 helmet model")
    p.add_argument(
        "--weights",
        default=str(ROOT / "models" / "helmet_yolov8n.pt"),
        help="Path to trained .pt file",
    )
    p.add_argument("--format",  choices=FORMATS, default="onnx", help="Export format")
    p.add_argument("--imgsz",   type=int, default=640,  help="Input image size")
    p.add_argument("--half",    action="store_true",    help="FP16 export (GPU only)")
    p.add_argument("--dynamic", action="store_true",    help="Dynamic batch (ONNX)")
    p.add_argument("--simplify",action="store_true",    help="Simplify ONNX graph")
    p.add_argument("--opset",   type=int, default=17,   help="ONNX opset version")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    print(f"\n[INFO] Exporting: {weights}  →  {args.format.upper()}")
    model = YOLO(str(weights))

    export_path = model.export(
        format   = args.format,
        imgsz    = args.imgsz,
        half     = args.half,
        dynamic  = args.dynamic,
        simplify = args.simplify,
        opset    = args.opset,
    )

    print(f"\n[OK] Exported model saved to: {export_path}")
    print("You can now use this model in production pipelines or other frameworks.")


if __name__ == "__main__":
    main()
