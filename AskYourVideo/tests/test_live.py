"""Live mode against the synthetic video instead of a camera."""

from __future__ import annotations

import time

import cv2
import numpy as np
from conftest import COLORS

from askvideo.detect import Box
from askvideo.index import VideoIndex
from askvideo.live import run_live


class ColorEmbedder:
    name = "fake"
    dim = 3

    def embed_images(self, images):
        v = np.array([im.reshape(-1, 3).mean(axis=0) for im in images], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts):
        v = np.array([COLORS[t] for t in texts], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class SlowTracker:
    """Pretends to detect a 'thing' with a stable id; slow enough for the embed thread to keep up."""

    def __init__(self):
        self.calls = 0

    def track_stream(self, rgb, labels):
        self.calls += 1
        time.sleep(0.02)
        return [Box("thing", 0.9, (1, 1, 20, 20), 7)]


def test_live_session_alerts_records_and_becomes_searchable(rgb_video, tmp_path):
    tracker = SlowTracker()
    result = run_live(
        str(rgb_video),
        tracker,
        ["thing"],
        tmp_path / "rec.mp4",
        embedder=ColorEmbedder(),
        watch=["blue"],
        watch_threshold=0.9,
        record_fps=10,
        show=False,
        snapshot=tmp_path / "last.jpg",
    )

    # A file source is detected synchronously: every frame, deterministic.
    assert tracker.calls == result.frames_detected == result.frames_shown == 120
    assert [a.query for a in result.alerts] == ["blue"]  # fires once, not every frame
    assert 7.5 <= result.alerts[0].ts <= 9.5

    cap = cv2.VideoCapture(str(result.record_path))
    recorded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 115 <= recorded <= 125  # 12 s at 10 fps: file time matches session time
    assert (tmp_path / "last.jpg").is_file()

    index = VideoIndex("fake", fps=1.0, segment_sec=4.0)
    index.add_embedded(str(result.record_path), result.frames, result.embeddings)
    top = index.search("green", ColorEmbedder(), top_k=1)[0]
    assert top.start == 4.0


def test_live_without_embedder_only_tracks(rgb_video, tmp_path):
    result = run_live(str(rgb_video), SlowTracker(), ["thing"], tmp_path / "rec.mp4", show=False, duration=2.0)
    assert result.embeddings is None and result.alerts == []
    assert 0 < result.frames_shown <= 21
