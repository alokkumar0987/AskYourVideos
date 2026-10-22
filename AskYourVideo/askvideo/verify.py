"""Stage 2: confirm retrieved segments with a pixel-level detector and re-rank them.

Retrieval is high-recall but coarse: it knows a segment *looks like* the query,
not that the required objects are actually in frame. Here every candidate
segment is sampled more densely, the detector looks for each required label,
and segments are re-ranked by how consistently all labels show up together.
Only the candidate segments are ever decoded, which is where the savings over
per-frame detection come from.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from askvideo.detect import Box, Detector
from askvideo.frames import sample_frames, video_info
from askvideo.index import Hit


@dataclass(frozen=True)
class Verified:
    hit: Hit
    support: float  # mean per-frame evidence over the segment (0..1); ranking key
    evidence: float  # evidence on the single best frame
    best_ts: float  # timestamp of that best frame
    boxes: tuple[Box, ...]  # detections on the best frame
    counts: dict[str, int] = field(default_factory=dict)  # most instances seen in one frame
    frames_checked: int = 0

    @property
    def verified(self) -> bool:
        return self.evidence > 0


@dataclass(frozen=True)
class VerifyStats:
    candidates: int
    frames_detected: int  # frames the detector actually ran on
    video_frames: int  # native frames in the videos involved
    detect_seconds: float

    @property
    def fraction(self) -> float:
        return self.frames_detected / self.video_frames if self.video_frames else 0.0


def frame_evidence(boxes: Sequence[Box], labels: Sequence[str]) -> float:
    """Weakest required label's best confidence; 0 if any label is missing.

    Using the minimum makes the score an AND: "person eating a biscuit" needs
    both a person and a biscuit, and is only as certain as the shakier one.
    """
    best = {label: 0.0 for label in labels}
    for b in boxes:
        if b.label in best:
            best[b.label] = max(best[b.label], b.conf)
    return min(best.values()) if best else 0.0


RANKINGS = ("gate", "support")


def verify_hits(
    hits: Sequence[Hit],
    detector: Detector,
    labels: Sequence[str],
    fps: float = 2.0,
    rank: str = "gate",
) -> tuple[list[Verified], VerifyStats]:
    """Run ``detector`` over each hit's segment and return the hits re-ranked.

    Verified segments (all labels seen together in at least one frame) always
    come first. ``rank`` decides the order among them:

    - ``"gate"`` (default): keep the retrieval order. The detector only answers
      "are the objects there?"; the embedding still judges how well the scene
      matches the query. Labels are coarse ("biscuit" covers raw and baked), so
      letting detector confidence reorder results throws away what the query
      said about *which* biscuits.
    - ``"support"``: order by detector support, for comparison.

    Frames are detected independently, without tracking: at a few frames per
    second, small hand-held objects move too far between samples for ByteTrack
    to associate them, and it drops unconfirmed new tracks, so tracking here
    deleted exactly the brief detections verification exists to find.
    """
    if not labels:
        raise ValueError("verify_hits needs at least one label")
    if rank not in RANKINGS:
        raise ValueError(f"rank must be one of {RANKINGS}")
    start = time.perf_counter()
    results, frames_detected = [], 0
    for hit in hits:
        frames = list(sample_frames(hit.video, fps, hit.start, hit.end))
        if not frames:
            continue
        per_frame = detector.detect([f.rgb for f in frames], labels)
        frames_detected += len(frames)
        scores = [frame_evidence(boxes, labels) for boxes in per_frame]
        best = max(range(len(frames)), key=scores.__getitem__)
        counts: dict[str, int] = {}
        for boxes in per_frame:
            for label in {b.label for b in boxes}:
                counts[label] = max(counts.get(label, 0), sum(b.label == label for b in boxes))
        results.append(
            Verified(
                hit=hit,
                support=sum(scores) / len(scores),
                evidence=scores[best],
                best_ts=frames[best].ts if scores[best] > 0 else hit.best_ts,
                boxes=tuple(per_frame[best]),
                counts=counts,
                frames_checked=len(frames),
            )
        )

    if rank == "gate":
        results.sort(key=lambda v: (v.verified, v.hit.score), reverse=True)
    else:
        results.sort(key=lambda v: (v.verified, v.support, v.hit.score), reverse=True)
    videos = {h.video for h in hits}
    stats = VerifyStats(
        candidates=len(hits),
        frames_detected=frames_detected,
        video_frames=sum(video_info(v)[1] for v in videos),
        detect_seconds=round(time.perf_counter() - start, 2),
    )
    return results, stats
