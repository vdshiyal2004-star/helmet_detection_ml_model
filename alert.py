"""
utils/alert.py
==============
Alert system for "No Helmet" detections.
Supports: console print, sound beep (pygame), system notification.
"""

import time
import threading
from pathlib import Path
from datetime import datetime


# ── Cooldown so alerts don't spam ────────────────────────────
_last_alert_time: float = 0.0
ALERT_COOLDOWN_SECONDS: float = 3.0          # minimum gap between alerts


# ─────────────────────────────────────────────────────────────
# Sound subsystem (pygame — non-blocking)
# ─────────────────────────────────────────────────────────────

_pygame_ready = False

def _init_pygame():
    """Initialise pygame mixer once."""
    global _pygame_ready
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        _pygame_ready = True
    except Exception:
        pass   # silent fail — sound is optional

_init_pygame()


def _beep_pygame(frequency: int = 880, duration_ms: int = 400):
    """Generate a synthetic beep via pygame (no sound file needed)."""
    try:
        import pygame
        import numpy as np
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
        wave = (32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        sound = pygame.sndarray.make_sound(wave)
        sound.play()
    except Exception:
        pass


def _play_sound_file(path: str):
    """Play a .wav / .mp3 file (if it exists)."""
    try:
        import pygame
        if not _pygame_ready:
            return
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception:
        pass


def play_alert_sound(sound_file: str = ""):
    """
    Play alert sound in a background thread so detection doesn't pause.
    Falls back to a synthesised beep if no file is given / found.
    """
    def _play():
        if sound_file and Path(sound_file).exists():
            _play_sound_file(sound_file)
        else:
            _beep_pygame(frequency=880, duration_ms=350)

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────
# Console / terminal alert
# ─────────────────────────────────────────────────────────────

def console_alert(message: str = "⚠️  NO HELMET DETECTED!"):
    """Print a bold console warning."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'!'*55}")
    print(f"  [{ts}]  {message}")
    print(f"{'!'*55}\n")


# ─────────────────────────────────────────────────────────────
# System desktop notification (optional)
# ─────────────────────────────────────────────────────────────

def system_notification(title: str = "Helmet Alert", body: str = "No helmet detected!"):
    """Send a desktop notification (Linux/macOS only, best-effort)."""
    try:
        import subprocess
        import platform
        system = platform.system()
        if system == "Linux":
            subprocess.Popen(
                ["notify-send", title, body],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        elif system == "Darwin":
            cmd = f'display notification "{body}" with title "{title}"'
            subprocess.Popen(
                ["osascript", "-e", cmd],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        # Windows: no built-in easy way without extra deps
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Unified trigger (call this from detect.py)
# ─────────────────────────────────────────────────────────────

def trigger_alert(
    sound: bool = True,
    console: bool = True,
    notification: bool = False,
    sound_file: str = "",
    cooldown: float = ALERT_COOLDOWN_SECONDS,
):
    """
    Trigger all enabled alert channels with an optional cooldown.

    Args:
        sound        : play beep / sound file
        console      : print terminal warning
        notification : system desktop notification
        sound_file   : path to a custom .wav/.mp3 (empty = synth beep)
        cooldown     : minimum seconds between consecutive alerts
    """
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time < cooldown:
        return   # still within cooldown window
    _last_alert_time = now

    if console:
        console_alert()
    if sound:
        play_alert_sound(sound_file)
    if notification:
        system_notification()
