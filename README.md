# 🪖 Helmet Detection System
### Real-time AI-powered helmet detection using YOLOv8 + OpenCV

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Real-time Detection](#real-time-detection)
7. [Extra Features](#extra-features)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This system detects whether motorcycle riders are wearing helmets in real-time video streams.

| Feature | Details |
|---|---|
| Model | YOLOv8n (nano — fast) / YOLOv8s/m (more accurate) |
| Classes | `helmet` (0) · `no_helmet` (1) |
| Input | Webcam · Video file · RTSP stream |
| Alerts | Console warning + beep sound |
| Output | Annotated video · Snapshots on violation |
| Optional | Number plate OCR · Desktop notification |

---

## Project Structure

```
helmet-detection/
├── config/
│   └── dataset.yaml          ← YOLOv8 dataset config
├── dataset/
│   ├── images/
│   │   ├── train/            ← training images
│   │   ├── val/              ← validation images
│   │   └── test/             ← test images
│   └── labels/
│       ├── train/            ← YOLO .txt annotations
│       ├── val/
│       └── test/
├── models/
│   └── helmet_yolov8n.pt     ← trained weights (after training)
├── output/
│   ├── videos/               ← saved detection videos
│   └── images/               ← no-helmet snapshots
├── runs/
│   └── train/
│       └── helmet_detector/  ← training logs, plots, checkpoints
├── scripts/
│   ├── download_dataset.py   ← dataset download & preparation
│   ├── convert_voc_to_yolo.py← VOC XML → YOLO format
│   ├── train.py              ← model training
│   ├── detect.py             ← real-time inference ⭐
│   └── export_model.py       ← export to ONNX / TensorRT / etc.
├── utils/
│   ├── alert.py              ← sound + console alerts
│   ├── drawing.py            ← bounding boxes, HUD overlay
│   └── plate_reader.py       ← number plate OCR (optional)
├── requirements.txt
├── setup_env.sh
└── README.md
```

---

## Environment Setup

### Quick setup (Linux / macOS)
```bash
git clone <repo-url> helmet-detection
cd helmet-detection
bash setup_env.sh
source helmet_env/bin/activate
```

### Manual setup (Windows / all platforms)
```bash
# 1. Create virtual environment
python -m venv helmet_env
source helmet_env/bin/activate       # Linux/macOS
# OR: helmet_env\Scripts\activate    # Windows

# 2. Install PyTorch (with CUDA for GPU)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision

# 3. Install all dependencies
pip install -r requirements.txt
```

### Verify installation
```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())

from ultralytics import YOLO
model = YOLO("yolov8n.pt")           # downloads ~6 MB
print("YOLOv8 ready!")
```

### GPU Setup (NVIDIA)
1. Install [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)
2. Install [cuDNN](https://developer.nvidia.com/cudnn) matching your CUDA version
3. Re-run the PyTorch CUDA install above
4. Verify: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

---

## Dataset Preparation

### Option A — Demo dataset (no download, for testing pipeline)
```bash
python scripts/download_dataset.py --source demo
```

### Option B — Roboflow (recommended, pre-formatted for YOLO)
1. Visit https://universe.roboflow.com/search?q=helmet+detection
2. Choose a dataset → **Download Dataset** → format: **YOLOv8**
3. Unzip into `dataset/` following the folder structure above
4. Or use the Roboflow Python API:
```python
pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("helmet-detection")
dataset = project.version(1).download("yolov8")
```

### Option C — Kaggle dataset
```bash
# 1. Get API key from https://www.kaggle.com/settings → API
# 2. Save to ~/.kaggle/kaggle.json
# 3. Run:
python scripts/download_dataset.py --source kaggle
```

### Option D — Custom dataset
1. Collect images of riders with/without helmets
2. Annotate with [LabelImg](https://github.com/HumanSignal/labelImg):
   ```bash
   pip install labelImg
   labelImg
   ```
   - Select **YOLO** format in LabelImg
   - Label each object as `helmet` or `no_helmet`
3. Organize into `dataset/images/` and `dataset/labels/`

### Converting Kaggle VOC XML to YOLO
```bash
python scripts/convert_voc_to_yolo.py \
    --img-dir path/to/kaggle/images \
    --ann-dir path/to/kaggle/annotations \
    --out-dir dataset \
    --split 0.8 0.1 0.1
```

---

## Training

```bash
# Default (YOLOv8n, 50 epochs, auto-detect GPU)
python scripts/train.py

# Larger model, more epochs
python scripts/train.py --model yolov8s.pt --epochs 100 --batch 32

# Resume from last checkpoint
python scripts/train.py --resume

# Force CPU
python scripts/train.py --device cpu
```

### Model size guide

| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| `yolov8n.pt` | ⚡ Fastest | ⭐⭐ | Raspberry Pi, mobile, CPU |
| `yolov8s.pt` | ⚡⚡ Fast | ⭐⭐⭐ | Webcam with mid GPU |
| `yolov8m.pt` | ⚡⚡⚡ Medium | ⭐⭐⭐⭐ | Server / NVIDIA GPU |
| `yolov8l.pt` | Slow | ⭐⭐⭐⭐⭐ | High-accuracy production |

Training outputs are saved to `runs/train/helmet_detector/`:
- `weights/best.pt` — best checkpoint (use this for inference)
- `weights/last.pt` — most recent checkpoint
- `results.png` — loss/mAP curves
- `confusion_matrix.png`
- `val_batch0_pred.jpg` — validation predictions

---

## Real-time Detection

```bash
# Webcam (default, device 0)
python scripts/detect.py

# Specific webcam
python scripts/detect.py --source 1

# Video file
python scripts/detect.py --source path/to/video.mp4

# Custom trained weights
python scripts/detect.py --weights runs/train/helmet_detector/weights/best.pt

# Adjust confidence threshold (lower = more detections, higher = more precise)
python scripts/detect.py --conf 0.5

# Disable alert sound
python scripts/detect.py --no-alert

# Don't save output video
python scripts/detect.py --no-save

# With number plate OCR
python scripts/detect.py --plates
```

### Keyboard shortcuts (live window)
| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Force save snapshot |
| `p` | Pause / unpause |

---

## Extra Features

### Number Plate Detection
```bash
pip install easyocr
python scripts/detect.py --plates
```
Detected plate text is shown below the bounding box of each rider.

### Export for Production
```bash
# ONNX (runs on any hardware)
python scripts/export_model.py --format onnx --weights runs/train/.../best.pt

# TensorRT (NVIDIA GPU, fastest inference)
python scripts/export_model.py --format engine --half
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'ultralytics'` | `pip install ultralytics` |
| `CUDA out of memory` | Reduce `--batch` or use `--imgsz 416` |
| Webcam not opening | Try `--source 1` or `--source 2` |
| Poor accuracy | Train with more data; use `yolov8s.pt` or larger |
| Sound not working | Install pygame: `pip install pygame` |
| Slow on CPU | Use `--imgsz 320` or `--model yolov8n.pt` |

---

## Dataset Links
- Kaggle: https://www.kaggle.com/datasets/andrewmvd/helmet-detection
- Roboflow: https://universe.roboflow.com/search?q=helmet+detection
- OpenImages (subset): https://storage.googleapis.com/openimages/web/index.html

---

*Built with YOLOv8 (Ultralytics) · OpenCV · PyTorch*
