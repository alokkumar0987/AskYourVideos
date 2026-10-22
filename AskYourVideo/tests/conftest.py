from __future__ import annotations

import cv2
import numpy as np
import pytest

COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}  # RGB


@pytest.fixture
def rgb_video(tmp_path):
    """12 s at 10 fps: 0-4 s red, 4-8 s green, 8-12 s blue."""
    path = tmp_path / "rgb.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for color in ("red", "green", "blue"):
        bgr = np.full((48, 64, 3), COLORS[color][::-1], np.uint8)
        for _ in range(40):
            writer.write(bgr)
    writer.release()
    return path
