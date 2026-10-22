"""Offline tests: a synthetic red/green/blue video and a color-based fake embedder."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import COLORS

from askvideo.frames import sample_frames
from askvideo.index import VideoIndex


class ColorEmbedder:
    """Embeds an image as its mean RGB color; a text as that color's unit vector."""

    name = "fake"
    dim = 3

    def embed_images(self, images):
        v = np.array([im.reshape(-1, 3).mean(axis=0) for im in images], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts):
        v = np.array([COLORS[t] for t in texts], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_sample_frames_rate(rgb_video):
    frames = list(sample_frames(rgb_video, fps=1.0))
    assert len(frames) == 12
    assert [round(f.ts) for f in frames] == list(range(12))


def test_search_finds_the_right_segment(rgb_video):
    index = VideoIndex("fake", fps=2.0, segment_sec=4.0)
    assert index.add_video(rgb_video, ColorEmbedder()) == 3

    for color, start in (("red", 0.0), ("green", 4.0), ("blue", 8.0)):
        top = index.search(color, ColorEmbedder(), top_k=1)[0]
        assert top.start == start
        assert start <= top.best_ts < start + 4.0


def test_reindexing_same_video_is_a_noop(rgb_video):
    index = VideoIndex("fake")
    index.add_video(rgb_video, ColorEmbedder())
    assert index.add_video(rgb_video, ColorEmbedder()) == 0
    assert len(index.videos) == 1


def test_save_load_roundtrip(rgb_video, tmp_path):
    index = VideoIndex("fake", fps=2.0)
    index.add_video(rgb_video, ColorEmbedder())
    loaded = VideoIndex.load(index.save(tmp_path / "idx"))

    assert loaded.segments == index.segments
    np.testing.assert_allclose(loaded.seg_emb, index.seg_emb)
    assert loaded.search("blue", ColorEmbedder(), 1) == index.search("blue", ColorEmbedder(), 1)


def test_model_mismatch_is_rejected(rgb_video):
    index = VideoIndex("siglip2")
    with pytest.raises(ValueError, match="not comparable"):
        index.add_video(rgb_video, ColorEmbedder())
