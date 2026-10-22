"""Segment-level semantic index over one or more videos.

Frames are sampled at a fixed rate and embedded; consecutive frames are pooled
into fixed-length segments (mean of normalized frame vectors, re-normalized).
Search ranks segments by cosine similarity, then points at the single
best-matching frame inside each segment so later stages know where to look.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from askvideo.embedder import Embedder
from askvideo.frames import sample_frames

_META = "meta.json"
_ARRAYS = "embeddings.npz"


@dataclass(frozen=True)
class Hit:
    video: str
    start: float
    end: float
    score: float
    best_ts: float  # timestamp of the frame inside the segment closest to the query
    segment_id: int


class VideoIndex:
    def __init__(self, model: str, fps: float = 1.0, segment_sec: float = 4.0):
        self.model = model
        self.fps = fps
        self.segment_sec = segment_sec
        self.frames: list[dict] = []  # {"video", "ts", "index"}
        self.segments: list[dict] = []  # {"video", "start", "end", "frames": [frame ids]}
        self.frame_emb = np.zeros((0, 0), np.float32)
        self.seg_emb = np.zeros((0, 0), np.float32)
        self.stats: dict[str, dict] = {}  # video -> indexing stats, kept for benchmarks
        self._faiss: faiss.Index | None = None

    @property
    def videos(self) -> list[str]:
        return list(self.stats)

    def add_video(
        self,
        path: str | Path,
        embedder: Embedder,
        progress: Callable[[int], None] | None = None,
    ) -> int:
        """Sample, embed, and segment one video. Returns the number of segments added."""
        self._check_model(embedder)
        video = str(Path(path).resolve())
        if video in self.stats:
            return 0

        start = time.perf_counter()
        frames, images = [], []
        for f in sample_frames(video, self.fps):
            frames.append({"video": video, "ts": round(f.ts, 3), "index": f.index})
            images.append(f.rgb)
            if progress:
                progress(len(frames))
        if not frames:
            raise ValueError(f"No frames could be read from {video}")
        n = self.add_embedded(video, frames, embedder.embed_images(images))
        self.stats[video]["index_seconds"] = round(time.perf_counter() - start, 2)
        return n

    def add_embedded(self, video: str, frames: list[dict], emb: np.ndarray) -> int:
        """Add frames that were already embedded elsewhere (e.g. during a live session).

        ``frames`` are ``{"video", "ts", "index"}`` dicts in time order, one per
        row of ``emb``; the caller is responsible for using this index's model.
        """
        video = str(Path(video).resolve())
        if video in self.stats:
            return 0
        if len(frames) != len(emb) or not frames:
            raise ValueError("frames and embeddings must be non-empty and the same length")
        frames = [{**f, "video": video} for f in frames]
        emb = np.asarray(emb, dtype=np.float32)
        base = len(self.frames)
        groups: dict[int, list[int]] = {}
        for i, fr in enumerate(frames):
            groups.setdefault(int(fr["ts"] // self.segment_sec), []).append(base + i)

        all_emb = emb if self.frame_emb.size == 0 else np.vstack([self.frame_emb, emb])
        new_segs, new_seg_emb = [], []
        for slot, ids in sorted(groups.items()):
            new_segs.append(
                {
                    "video": video,
                    "start": slot * self.segment_sec,
                    "end": (slot + 1) * self.segment_sec,
                    "frames": ids,
                }
            )
            new_seg_emb.append(_normalize(all_emb[ids].mean(axis=0)))

        self.frames.extend(frames)
        self.frame_emb = all_emb
        self.segments.extend(new_segs)
        seg = np.stack(new_seg_emb).astype(np.float32)
        self.seg_emb = seg if self.seg_emb.size == 0 else np.vstack([self.seg_emb, seg])
        self._faiss = None
        self.stats[video] = {
            "frames_embedded": len(frames),
            "segments": len(new_segs),
            "duration_sec": round(frames[-1]["ts"], 2),
            "index_seconds": 0.0,
        }
        return len(new_segs)

    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> list[Hit]:
        """Return the ``top_k`` segments most similar to a text query."""
        q = self.embed_query(query, embedder)
        ids, scores = self.rank_segments(q, top_k)
        return [self.make_hit(sid, score, q) for sid, score in zip(ids, scores)]

    def embed_query(self, query: str, embedder: Embedder) -> np.ndarray:
        self._check_model(embedder)
        return embedder.embed_texts([query]).astype(np.float32)[0]

    def rank_segments(self, q: np.ndarray, top_k: int | None = None) -> tuple[list[int], list[float]]:
        """Segment ids by similarity to an embedded query, best first (all if ``top_k`` is None)."""
        if not self.segments:
            return [], []
        k = len(self.segments) if top_k is None else min(top_k, len(self.segments))
        scores, ids = self._index().search(q[None, :], k)
        keep = ids[0] >= 0
        return [int(i) for i in ids[0][keep]], [float(s) for s in scores[0][keep]]

    def make_hit(self, segment_id: int, score: float, q: np.ndarray) -> Hit:
        """A hit for one segment; its best frame is the one closest to the query."""
        seg = self.segments[segment_id]
        fids = seg["frames"]
        best = fids[int(np.argmax(self.frame_emb[fids] @ q))]
        return Hit(
            video=seg["video"],
            start=seg["start"],
            end=seg["end"],
            score=float(score),
            best_ts=self.frames[best]["ts"],
            segment_id=int(segment_id),
        )

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / _ARRAYS, frame_emb=self.frame_emb, seg_emb=self.seg_emb)
        meta = {
            "model": self.model,
            "fps": self.fps,
            "segment_sec": self.segment_sec,
            "frames": self.frames,
            "segments": self.segments,
            "stats": self.stats,
        }
        (d / _META).write_text(json.dumps(meta), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> VideoIndex:
        d = Path(directory)
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        arrays = np.load(d / _ARRAYS)
        idx = cls(meta["model"], meta["fps"], meta["segment_sec"])
        idx.frames, idx.segments, idx.stats = meta["frames"], meta["segments"], meta["stats"]
        idx.frame_emb = arrays["frame_emb"].astype(np.float32)
        idx.seg_emb = arrays["seg_emb"].astype(np.float32)
        return idx

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / _META).is_file()

    def _index(self) -> faiss.Index:
        # Exact inner product on normalized vectors == cosine similarity. Flat is
        # fine up to ~1e6 segments; swap for HNSW beyond that.
        if self._faiss is None:
            self._faiss = faiss.IndexFlatIP(self.seg_emb.shape[1])
            self._faiss.add(self.seg_emb)
        return self._faiss

    def _check_model(self, embedder: Embedder) -> None:
        if embedder.name != self.model:
            raise ValueError(
                f"Index was built with {self.model!r} but embedder is {embedder.name!r}; "
                "embeddings from different models are not comparable"
            )


def hits_to_json(hits: Sequence[Hit]) -> str:
    return json.dumps([asdict(h) for h in hits], indent=2)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)
