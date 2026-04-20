import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# ── Project imports ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.drawing    import draw_detection, draw_hud, flash_alert
from utils.alert      import trigger_alert
from ultralytics      import YOLO

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR        = ROOT / "output"
VIDEO_OUT_DIR     = OUTPUT_DIR / "videos"
IMAGE_OUT_DIR     = OUTPUT_DIR / "images"
DEFAULT_WEIGHTS   = ROOT / "models" / "helmet_yolov8n.pt"
FALLBACK_WEIGHTS  = "yolov8n.pt" 

CLASS_NAMES = {0: "helmet", 1: "no_helmet"}

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Helmet Detection — Inference")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--source", default="0", help="Path to image folder, video, or '0' for webcam")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--no-alert", action="store_true")
    p.add_argument("--plates", action="store_true")
    p.add_argument("--max-det", type=int, default=50)
    return p.parse_args()

def load_model(weights_path: str, device: str) -> YOLO:
    wp = Path(weights_path)
    if not wp.exists():
        weights_path = FALLBACK_WEIGHTS
    model = YOLO(weights_path)
    # Check for CUDA/GPU
    try:
        import torch
        dev = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    except:
        dev = "cpu"
    model.to(dev)
    return model

def open_source(source: str):
    """Returns (data_object, is_webcam, fps, w, h, is_directory)"""
    source_path = Path(source)
    
    if source_path.is_dir():
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [p for p in source_path.iterdir() if p.suffix.lower() in valid_extensions]
        if not images:
            print(f"[ERROR] No images found in {source}")
            sys.exit(1)
        return sorted(images), False, 1.0, 0, 0, True

    try:
        source_input = int(source)
        is_webcam = True
    except ValueError:
        source_input = source
        is_webcam = False

    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, is_webcam, fps, w, h, False

def save_result(frame, original_source):
    """Saves the detected image with a unique name."""
    if isinstance(original_source, Path):
        save_name = f"det_{original_source.name}"
    else:
        ts = datetime.now().strftime("%H%M%S_%f")
        save_name = f"webcam_{ts}.jpg"
    
    path = IMAGE_OUT_DIR / save_name
    cv2.imwrite(str(path), frame)
    print(f"[SAVED] {path}")

# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run():
    args = parse_args()
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights, args.device)
    source_data, is_webcam, src_fps, W, H, is_dir = open_source(args.source)
    
    writer = None
    if not args.no_save and not is_dir:
        out_path = VIDEO_OUT_DIR / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps if not is_webcam else 30.0, (W, H))

    fps_buffer = deque(maxlen=30)
    prev_time = time.time()
    total_dets = helmet_count = no_helmet_cnt = flash_frames = 0
    paused = False
    img_idx = 0

    while True:
        if paused:
            if cv2.waitKey(50) & 0xFF == ord("p"): paused = False
            elif cv2.waitKey(50) & 0xFF == ord("q"): break
            continue

        # ── Get Frame ──
        current_source_info = None
        if is_dir:
            if img_idx >= len(source_data): break
            current_source_info = source_data[img_idx]
            frame = cv2.imread(str(current_source_info))
            ret = True if frame is not None else False
            img_idx += 1
            if ret: H, W = frame.shape[:2]
        else:
            ret, frame = source_data.read()
            current_source_info = "webcam"

        if not ret: break

        # ── Inference ──
        results = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
        
        annotated = frame.copy()
        found_something = False

        for result in results:
            if result.boxes is None: continue
            for box in result.boxes:
                found_something = True
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, result.names.get(cls_id, f"cls{cls_id}"))

                total_dets += 1
                if label == "helmet": helmet_count += 1
                else: no_helmet_cnt += 1

                draw_detection(annotated, x1, y1, x2, y2, label, conf_score)

        # ── Save Every Detected Photo ──
        if found_something:
            save_result(annotated, current_source_info)

        # ── HUD and Display ──
        now = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        draw_hud(annotated, sum(fps_buffer)/len(fps_buffer), total_dets, helmet_count, no_helmet_cnt, args.source)

        if writer: writer.write(annotated)
        if not args.no_show:
            cv2.imshow("Helmet Detection", annotated)
            # 1000ms (1 second) delay for images so you can see them
            key = cv2.waitKey(1000 if is_dir else 1) & 0xFF
            if key == ord("q"): break
            elif key == ord("p"): paused = True

    if not is_dir: source_data.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("\n--- Processing Finished ---")

if __name__ == "__main__":
    run()