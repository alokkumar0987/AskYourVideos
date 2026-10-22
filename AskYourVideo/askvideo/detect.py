"""Open-vocabulary object detection (YOLOE) with optional ByteTrack tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

DEFAULT_MODEL = "yoloe-26s-seg.pt"


@dataclass(frozen=True)
class Box:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


class Detector(Protocol):
    def detect(
        self, frames: Sequence[np.ndarray], labels: Sequence[str], track: bool = False
    ) -> list[list[Box]]:
        """Detect ``labels`` in RGB ``frames``. With ``track``, frames are treated as one
        continuous clip and boxes carry track ids that persist across it."""
        ...


class YoloeDetector:
    """YOLOE with text prompts: any label works, not just a fixed class list.

    Weights (the detector and its MobileCLIP text encoder) are resolved through
    Ultralytics' ``weights_dir`` setting, so relocate them with
    ``yolo settings weights_dir=<dir>``.
    """

    def __init__(self, model: str = DEFAULT_MODEL, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLOE, settings

        path = Path(model)
        if not path.exists() and (Path(settings["weights_dir"]) / model).exists():
            path = Path(settings["weights_dir"]) / model
        self._model = YOLOE(str(path))
        self.conf = conf
        self.imgsz = imgsz
        self._labels: tuple[str, ...] = ()
        self._streaming = False

    def detect(
        self, frames: Sequence[np.ndarray], labels: Sequence[str], track: bool = False
    ) -> list[list[Box]]:
        if not frames:
            return []
        self._set_labels(labels)
        kwargs = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames]
        if track:
            # persist=False on the first frame starts a fresh ByteTrack state, so
            # ids never leak from one clip into the next.
            results = [
                self._model.track(im, persist=i > 0, tracker="bytetrack.yaml", **kwargs)[0]
                for i, im in enumerate(bgr)
            ]
        else:
            results = self._model.predict(bgr, **kwargs)
        return [_to_boxes(r) for r in results]

    def track_stream(self, frame: np.ndarray, labels: Sequence[str]) -> list[Box]:
        """Detect and track in one frame of a continuous stream (RGB in).

        Unlike ``detect(track=True)``, tracker state persists across calls until
        ``reset_stream()``, so ids stay stable for as long as the stream runs.
        """
        self._set_labels(labels)
        result = self._model.track(
            np.ascontiguousarray(frame[:, :, ::-1]),
            persist=self._streaming,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        self._streaming = True
        return _to_boxes(result)

    def reset_stream(self) -> None:
        self._streaming = False

    def _set_labels(self, labels: Sequence[str]) -> None:
        labels = tuple(labels)
        if labels != self._labels:
            self._model.set_classes(list(labels))
            self._labels = labels
            self._streaming = False  # set_classes rebuilds the predictor and its trackers


def _to_boxes(result) -> list[Box]:
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
    return [
        Box(result.names[int(c)], float(s), tuple(float(v) for v in xyxy), tid)
        for c, s, xyxy, tid in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist(), ids)
    ]


def draw_boxes(rgb: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    """Return a BGR copy of ``rgb`` with labeled boxes drawn on it."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b.xyxy)
        color = _color(b.label)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        tag = f"{b.label} {b.conf:.2f}" + (f" #{b.track_id}" if b.track_id is not None else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x0, max(0, y0 - th - 8)), (x0 + tw + 6, y0), color, -1)
        cv2.putText(out, tag, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _color(label: str) -> tuple[int, int, int]:
    h = sum(map(ord, label))
    return (60 + h * 37 % 180, 60 + h * 91 % 180, 60 + h * 53 % 180)
