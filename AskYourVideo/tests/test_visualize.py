import numpy as np

from askvideo.detect import Box
from askvideo.visualize import render_hit


def test_render_hit_adds_caption_and_keeps_only_query_labels():
    rgb = np.full((360, 640, 3), 90, np.uint8)
    boxes = [
        Box("person", 0.9, (100, 20, 500, 350), None),  # context box
        Box("biscuit", 0.5, (300, 200, 330, 225), None),  # small: gets a zoom inset
        Box("cup", 0.8, (520, 40, 600, 120), None),  # not asked for: dropped
    ]
    out = render_hit(
        rgb, boxes, query="a man eating a biscuit", ts=271, labels=["person", "biscuit"],
        verified=True, footnote="test.mp4",
    )
    assert out.shape[1] == 640 and out.shape[0] > 360 and out.dtype == np.uint8
    assert (out[:360] != rgb).any()  # boxes / inset were drawn


def test_render_hit_ignores_labels_the_query_did_not_ask_for():
    rgb = np.full((360, 640, 3), 90, np.uint8)
    out = render_hit(
        rgb, [Box("cup", 0.8, (520, 40, 600, 120), None)], query="q", ts=0, labels=["biscuit"],
        verified=False,
    )
    assert (out[:360] == rgb).all()  # frame untouched; only the caption is added


def test_render_hit_without_detections():
    rgb = np.zeros((100, 200, 3), np.uint8)
    out = render_hit(rgb, (), query="anything", ts=5)
    assert out.shape[:2] == (100 + round(118 * 200 / 1280), 200)
