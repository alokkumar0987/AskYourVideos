"""Sample frames from a video at a fixed rate."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int  # frame number in the source video
    ts: float  # seconds from the start
    rgb: np.ndarray  # HxWx3 uint8, RGB order


def video_info(path: str | Path) -> tuple[float, int]:
    """Return (native_fps, total_frames) for a video."""
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total
    finally:
        cap.release()


def sample_frames(
    path: str | Path, fps: float = 1.0, start: float = 0.0, end: float | None = None
) -> Iterator[Frame]:
    """Yield frames at roughly ``fps`` frames per second, optionally within [start, end).

    Skipped frames are ``grab()``-ed, not decoded, which is much cheaper than
    seeking on most codecs.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    cap = _open(path)
    try:
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(native / fps))
        first = max(0, round(start * native))
        last = None if end is None else round(end * native)
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        idx = first
        while last is None or idx < last:
            if (idx - first) % step == 0:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield Frame(idx, idx / native, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, ts: float) -> np.ndarray | None:
    """Decode the single frame nearest to ``ts`` seconds (RGB), or None.

    Seeks by frame number, the same way ``sample_frames`` does, so a timestamp
    taken from a sampled frame decodes to that exact frame (boxes line up).
    """
    cap = _open(path)
    try:
        native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(ts * native))
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


def _open(path: str | Path) -> cv2.VideoCapture:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {p}")
    return cap
