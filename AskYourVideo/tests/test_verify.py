"""Offline tests for stage 2, using the synthetic red/green/blue video and a fake detector."""

from __future__ import annotations

import numpy as np
import pytest

from askvideo.detect import Box
from askvideo.frames import sample_frames
from askvideo.index import Hit
from askvideo.verify import frame_evidence, verify_hits


class ColorDetector:
    """'Detects' a color label when the frame is mostly that color."""

    def __init__(self):
        self.frames_seen = 0

    def detect(self, frames, labels, track=False):
        self.frames_seen += len(frames)
        out = []
        for f in frames:
            mean = f.reshape(-1, 3).mean(axis=0)
            color = ("red", "green", "blue")[int(np.argmax(mean))]
            out.append([Box(color, 0.9, (0, 0, 10, 10))] * 2 if color in labels else [])
        return out


def _hit(video, start, score):
    return Hit(str(video), start, start + 4.0, score, start, int(start // 4))


def test_frames_can_be_read_from_a_time_range(rgb_video):
    frames = list(sample_frames(rgb_video, fps=2.0, start=4.0, end=8.0))
    assert len(frames) == 8
    assert 4.0 <= frames[0].ts and frames[-1].ts < 8.0
    assert all(f.rgb[..., 1].mean() > 200 for f in frames)  # all green


def test_frame_evidence_is_an_and_over_labels():
    boxes = [Box("person", 0.9, (0, 0, 1, 1)), Box("biscuit", 0.4, (0, 0, 1, 1))]
    assert frame_evidence(boxes, ["person", "biscuit"]) == pytest.approx(0.4)
    assert frame_evidence(boxes, ["person", "cake"]) == 0.0


def test_verification_reranks_and_only_touches_candidates(rgb_video):
    # Retrieval ranked red first, but only the blue segment contains a "blue" object.
    hits = [_hit(rgb_video, 0.0, 0.9), _hit(rgb_video, 8.0, 0.1)]
    det = ColorDetector()
    ranked, stats = verify_hits(hits, det, ["blue"], fps=2.0)

    assert ranked[0].hit.start == 8.0 and ranked[0].verified
    assert ranked[0].support == pytest.approx(0.9)
    assert 8.0 <= ranked[0].best_ts < 12.0
    assert ranked[0].counts == {"blue": 2}  # two boxes in the same frame
    assert not ranked[1].verified
    assert stats.frames_detected == det.frames_seen == 16
    assert stats.video_frames == 120
    assert stats.fraction == pytest.approx(16 / 120)


def test_gate_keeps_retrieval_order_among_verified(rgb_video):
    # Two verified hits with equal detector support: the gate must not reorder them
    # away from retrieval score, even when they arrive in the wrong order.
    hits = [_hit(rgb_video, 0.0, 0.2), _hit(rgb_video, 0.0, 0.8)]
    ranked, _ = verify_hits(hits, ColorDetector(), ["red"], rank="gate")
    assert [v.hit.score for v in ranked] == [0.8, 0.2]


def test_unknown_ranking_is_rejected(rgb_video):
    with pytest.raises(ValueError, match="rank"):
        verify_hits([_hit(rgb_video, 0.0, 1.0)], ColorDetector(), ["red"], rank="magic")


def test_verification_requires_labels(rgb_video):
    with pytest.raises(ValueError):
        verify_hits([_hit(rgb_video, 0.0, 1.0)], ColorDetector(), [])
