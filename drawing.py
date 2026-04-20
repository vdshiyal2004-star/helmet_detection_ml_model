"""
utils/drawing.py
================
All OpenCV drawing helpers: bounding boxes, labels, HUD overlay,
confidence bars, and the "No Helmet" capture frame highlight.
"""

import cv2
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────

COLORS = {
    "helmet":    (34,  197,  94),   # green
    "no_helmet": (239,  68,  68),   # red
    "unknown":   (156, 163, 175),   # grey
    "hud_bg":    (15,  23,  42),    # dark navy
    "hud_text":  (226, 232, 240),   # light grey
    "accent":    (99,  102, 241),   # indigo
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────
# Bounding box + label
# ─────────────────────────────────────────────────────────────

def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a rounded-corner bounding box with label pill.

    Args:
        frame      : BGR image
        x1,y1,x2,y2: bounding-box corners
        label      : "helmet" or "no_helmet"
        confidence : 0.0 – 1.0
        thickness  : line thickness

    Returns:
        Annotated frame (in-place modification + return)
    """
    key   = label.lower().replace(" ", "_")
    color = COLORS.get(key, COLORS["unknown"])

    # ── Bounding box ──────────────────────────────────────────
    _draw_rounded_rect(frame, x1, y1, x2, y2, color, thickness, radius=10)

    # ── Semi-transparent fill inside box ─────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # ── Label pill ────────────────────────────────────────────
    display_label = label.upper().replace("_", " ")
    pct_text      = f"{confidence * 100:.1f}%"
    full_text     = f"  {display_label}  {pct_text}  "

    font_scale = 0.55
    (tw, th), baseline = cv2.getTextSize(full_text, FONT, font_scale, 1)

    pill_x1 = x1
    pill_y1 = max(0, y1 - th - baseline - 10)
    pill_x2 = x1 + tw
    pill_y2 = y1

    # Pill background
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)
    # Pill text
    cv2.putText(
        frame, full_text,
        (pill_x1, pill_y2 - baseline - 2),
        FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA
    )

    return frame


# ─────────────────────────────────────────────────────────────
# HUD (Heads-Up Display) overlay
# ─────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    fps: float,
    total_detections: int,
    helmet_count: int,
    no_helmet_count: int,
    source_label: str = "",
) -> np.ndarray:
    """
    Draw a semi-transparent information panel in the top-left corner.
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    pad   = 10
    lh    = 22        # line height
    lines = 5
    pw    = 240
    ph    = pad * 2 + lh * lines

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (pw, ph), COLORS["hud_bg"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border accent
    cv2.line(frame, (pw, 0), (pw, ph), COLORS["accent"], 2)

    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    info = [
        (f"FPS: {fps:.1f}",                                COLORS["hud_text"]),
        (f"Total detections: {total_detections}",          COLORS["hud_text"]),
        (f"Helmet:    {helmet_count}",                     COLORS["helmet"]),
        (f"No Helmet: {no_helmet_count}",                  COLORS["no_helmet"]),
        (ts,                                               COLORS["accent"]),
    ]

    for i, (text, color) in enumerate(info):
        y = pad + (i + 1) * lh
        cv2.putText(frame, text, (pad, y), FONT, 0.48, color, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────
# "No Helmet" capture flash
# ─────────────────────────────────────────────────────────────

def flash_alert(frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Briefly tint the frame red — called for 1-2 frames when
    a no-helmet detection triggers a snapshot.
    """
    red_overlay = np.full_like(frame, (0, 0, 200), dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, red_overlay, alpha, 0)


# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────

def _draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int,
    radius: int = 8,
):
    """Draw a rectangle with rounded corners using arc segments."""
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

    # Straight segments
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0,  0, 90, color, thickness)
