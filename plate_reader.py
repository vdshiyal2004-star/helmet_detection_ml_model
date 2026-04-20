"""
utils/plate_reader.py
=====================
Optional number plate (license plate) detection using EasyOCR.
Called from detect.py when --plates flag is set.

Requirements:
    pip install easyocr opencv-python
"""

import re
from typing import Optional
import numpy as np


# ── Lazy-load EasyOCR so it doesn't slow startup ─────────────
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        except ImportError:
            print("[WARN] easyocr not installed. Run: pip install easyocr")
            _reader = None
    return _reader


# ─────────────────────────────────────────────────────────────
# Core plate detection
# ─────────────────────────────────────────────────────────────

def read_plate(frame: np.ndarray, region: Optional[tuple] = None) -> str:
    """
    Run OCR on a frame (or a cropped region) and return any text
    that looks like a license plate.

    Args:
        frame  : BGR numpy array (full frame or cropped ROI)
        region : (x1, y1, x2, y2) to crop before OCR; None = full frame

    Returns:
        Detected plate text (empty string if nothing found / OCR unavailable)
    """
    reader = _get_reader()
    if reader is None:
        return ""

    # Crop if region supplied
    if region is not None:
        x1, y1, x2, y2 = [max(0, v) for v in region]
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame

    if roi.size == 0:
        return ""

    try:
        results = reader.readtext(roi, detail=0, paragraph=False)
    except Exception:
        return ""

    # Filter results to plate-like strings
    plate_candidates = []
    for text in results:
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        # Typical plate: 4-10 alphanumeric chars
        if 4 <= len(cleaned) <= 10:
            plate_candidates.append(cleaned)

    return " | ".join(plate_candidates)


# ─────────────────────────────────────────────────────────────
# Draw plate text on frame
# ─────────────────────────────────────────────────────────────

def draw_plate(frame: np.ndarray, plate_text: str, x1: int, y2: int) -> np.ndarray:
    """
    Overlay detected plate text below a bounding box.

    Args:
        frame      : BGR frame
        plate_text : string to render
        x1, y2     : top-left x and bottom y of the associated bounding box
    """
    import cv2
    if not plate_text:
        return frame

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness  = 1
    color      = (0, 255, 255)   # yellow

    (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    tx, ty = x1, y2 + th + 6

    # Background pill
    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
    cv2.putText(frame, plate_text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame
