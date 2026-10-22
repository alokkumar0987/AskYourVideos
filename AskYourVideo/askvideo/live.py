"""Live mode: track objects on a webcam (or any stream), watch for scene queries,
and record the session so it can be searched afterwards like any other video.

Up to three loops share the CPU:

- the main loop reads frames, records, and draws at the camera's own rate;
- on a camera, a detection worker runs the detector + ByteTrack on the newest
  frame it can get and publishes its boxes; the main loop draws the latest
  boxes on every frame, so video stays smooth even when the detector manages
  only a few frames per second (boxes trail by one detection, ~0.25 s on CPU).
  On a video file every frame is detected in the main loop instead, so results
  are complete and repeatable;
- an embedding worker embeds about one frame per second with the scene model
  and scores it against the ``watch`` queries.

Both workers always take the newest frame and drop stale ones, so a slow model
lowers its own rate instead of stalling the video.

The recording is written at a fixed rate against the session clock (frames are
repeated when the loop runs slower), so a timestamp found by a later search
points at the same moment in the saved file.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from askvideo.detect import Box, draw_boxes
from askvideo.embedder import Embedder


@dataclass
class Alert:
    ts: float
    query: str
    score: float


@dataclass
class LiveResult:
    record_path: Path
    seconds: float
    frames_shown: int
    frames_detected: int
    alerts: list[Alert] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)  # {"video", "ts", "index"} per embedding
    embeddings: np.ndarray | None = None

    @property
    def fps(self) -> float:
        return self.frames_shown / self.seconds if self.seconds else 0.0

    @property
    def detect_fps(self) -> float:
        return self.frames_detected / self.seconds if self.seconds else 0.0


class _LatestWorker(threading.Thread):
    """Processes only the newest submitted item; ``handle`` runs on this thread."""

    def __init__(self):
        super().__init__(daemon=True)
        self._inbox: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    def submit(self, ts: float, rgb: np.ndarray) -> None:
        try:
            self._inbox.put_nowait((ts, rgb))
        except queue.Full:  # worker is behind: replace the stale frame with this one
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                pass
            self._inbox.put_nowait((ts, rgb))

    def stop(self) -> None:
        # Make room for the sentinel even if a frame is still waiting.
        try:
            self._inbox.get_nowait()
        except queue.Empty:
            pass
        self._inbox.put(None)
        self.join()

    def run(self) -> None:
        while (item := self._inbox.get()) is not None:
            self.handle(*item)

    def handle(self, ts: float, rgb: np.ndarray) -> None:
        raise NotImplementedError


class _EmbedWorker(_LatestWorker):
    def __init__(self, embedder: Embedder, watch: Sequence[str]):
        super().__init__()
        self._embedder = embedder
        self._watch = embedder.embed_texts(list(watch)) if watch else None
        self._latest: np.ndarray | None = None
        self.records: list[tuple[float, np.ndarray]] = []

    def latest_scores(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def handle(self, ts: float, rgb: np.ndarray) -> None:
        emb = self._embedder.embed_images([rgb])[0]
        self.records.append((ts, emb))
        if self._watch is not None:
            with self._lock:
                self._latest = self._watch @ emb


class _DetectWorker(_LatestWorker):
    def __init__(self, detector, labels: Sequence[str]):
        super().__init__()
        self._detector = detector
        self._labels = list(labels)
        self._boxes: list[Box] = []
        self.calls = 0

    def latest_boxes(self) -> list[Box]:
        with self._lock:
            return self._boxes

    def handle(self, ts: float, rgb: np.ndarray) -> None:
        boxes = self._detector.track_stream(rgb, self._labels)
        with self._lock:
            self._boxes = boxes
            self.calls += 1


def open_source(source: str | int) -> tuple[cv2.VideoCapture, bool]:
    """Open a camera index or a video path. Returns (capture, is_camera)."""
    is_camera = isinstance(source, int)
    if is_camera and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # Windows: much faster to open
    else:
        cap = cv2.VideoCapture(source if is_camera else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    return cap, is_camera


def run_live(
    source: str | int,
    detector,
    labels: Sequence[str],
    record_path: str | Path,
    embedder: Embedder | None = None,
    watch: Sequence[str] = (),
    watch_threshold: float = 0.10,
    embed_every: float = 1.0,
    record_fps: float = 10.0,
    duration: float | None = None,
    show: bool = True,
    snapshot: str | Path | None = None,
    on_frame: Callable[[np.ndarray, list[Alert]], None] | None = None,
    stop: threading.Event | None = None,
) -> LiveResult:
    """Run the live loop until the source ends, ``duration`` passes, 'q' is
    pressed, or ``stop`` is set.

    ``detector`` needs ``track_stream(rgb, labels)``; pass no labels to skip
    detection. Without an ``embedder`` there is no watching and no index data.
    ``on_frame`` receives every annotated RGB frame and the alerts so far, e.g.
    to show the stream in a UI instead of an OpenCV window.
    """
    if watch and embedder is None:
        raise ValueError("watch queries need an embedder")
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    cap, is_camera = open_source(source)
    worker = _EmbedWorker(embedder, watch) if embedder is not None else None
    if worker:
        worker.start()
    if labels and hasattr(detector, "reset_stream"):
        detector.reset_stream()
    det_worker = _DetectWorker(detector, labels) if labels and is_camera else None
    if det_worker:
        det_worker.start()

    writer = None
    alerts: list[Alert] = []
    active: set[str] = set()
    written = shown = detected = 0
    next_embed = 0.0
    t0 = time.perf_counter()
    tick, loop_fps = t0, 0.0
    last_view = None
    try:
        while stop is None or not stop.is_set():
            ok, bgr = cap.read()
            if not ok:
                break
            # Session clock: wall time for cameras, media time for files.
            ts = time.perf_counter() - t0 if is_camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if duration is not None and ts >= duration:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if writer is None:
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), record_fps, (w, h)
                )
            while written <= ts * record_fps:  # pad so file time == session time
                writer.write(bgr)
                written += 1

            if worker and ts >= next_embed:
                worker.submit(ts, rgb)
                next_embed = ts + embed_every

            boxes: list[Box] = []
            if det_worker:
                det_worker.submit(ts, rgb)
                boxes = det_worker.latest_boxes()
            elif labels:
                boxes = detector.track_stream(rgb, labels)
                detected += 1
            view = draw_boxes(rgb, boxes)
            shown += 1

            scores = worker.latest_scores() if worker else None
            if scores is not None:
                for q, s in zip(watch, scores):
                    if s >= watch_threshold and q not in active:
                        active.add(q)
                        alerts.append(Alert(round(ts, 2), q, float(s)))
                    elif s < watch_threshold:
                        active.discard(q)

            now = time.perf_counter()
            dt, tick = max(now - tick, 1e-6), now
            loop_fps = 1 / dt if not loop_fps else 0.9 * loop_fps + 0.1 / dt  # smoothed
            det_fps = (det_worker.calls if det_worker else detected) / max(ts, 1e-6)
            _overlay(view, ts, loop_fps, det_fps if labels else None, watch, scores, watch_threshold)
            last_view = view
            if on_frame:
                on_frame(view[:, :, ::-1], alerts)  # draw_boxes returns BGR
            if show:
                cv2.imshow("askvideo live (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if det_worker:
            det_worker.stop()
            detected = det_worker.calls
        if worker:
            worker.stop()
        if show:
            cv2.destroyAllWindows()

    if snapshot is not None and last_view is not None:
        Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), last_view)

    result = LiveResult(record_path, time.perf_counter() - t0, shown, detected, alerts)
    if worker and worker.records:
        result.frames = [
            {"video": str(record_path), "ts": round(ts, 3), "index": round(ts * record_fps)}
            for ts, _ in worker.records
        ]
        result.embeddings = np.stack([e for _, e in worker.records])
    return result


def _overlay(view, ts, fps, det_fps, watch, scores, threshold) -> None:
    head = f"{ts:6.1f}s  video {fps:4.1f} FPS"
    if det_fps is not None:
        head += f"  detect {det_fps:4.1f}/s"
    lines = [(head, (0, 255, 255))]
    if scores is not None:
        for q, s in zip(watch, scores):
            hit = s >= threshold
            lines.append((f"{'MATCH' if hit else 'watch'} {s:+.2f}  {q}", (0, 0, 255) if hit else (230, 230, 230)))
    # Dark band behind the text so it stays readable on bright scenes.
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0] for t, _ in lines) + 20
    band = view[0 : 12 + 26 * len(lines), 0:width]
    band[:] = (band * 0.35).astype(view.dtype)
    for i, (text, color) in enumerate(lines):
        cv2.putText(view, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
