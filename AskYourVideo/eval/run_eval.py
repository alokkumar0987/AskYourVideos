"""Reproduce the numbers in docs/TECHNICAL_REPORT.md.

    python eval/run_eval.py            # needs .askvideo_siglip2 (samples/Cooking_cake.mp4)

Six queries on one video, ground-truth windows annotated by hand. A result is
correct when its best frame falls inside the window. Small by design: this is a
sanity check of the pipeline, not a benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from askvideo.detect import YoloeDetector
from askvideo.embedder import OpenClipEmbedder
from askvideo.index import VideoIndex
from askvideo.verify import verify_hits

# query, labels picked by hand (first run), labels taken only from the query's words, window (s)
CASES = [
    ("a man eating a biscuit", ["person", "biscuit"], ["person", "biscuit"], (266, 276)),
    ("kneading dough on a floured counter", ["dough", "hand"], ["dough"], (120, 150)),
    ("cutting dough with a round biscuit cutter", ["dough", "glass"], ["biscuit cutter"], (178, 192)),
    ("brushing butter on biscuits in a cast iron skillet", ["brush", "skillet"], ["brush", "skillet"], (205, 225)),
    ("a package of Challenge butter", ["butter package"], ["butter package"], (236, 246)),
    ("golden baked biscuits", ["biscuit"], ["biscuit"], (0, 12)),
]
CANDIDATES = 10


def main() -> None:
    index = VideoIndex.load(".askvideo_siglip2")
    embedder = OpenClipEmbedder(index.model)
    detector = YoloeDetector(conf=0.2)
    inside = lambda ts, w: w[0] <= ts <= w[1]  # noqa: E731

    rows, totals = [], {"retrieval": 0, "support": 0, "gate": 0, "gate_query_labels": 0}
    frames = video_frames = 0
    seconds = []
    for query, hand_labels, query_labels, window in CASES:
        hits = index.search(query, embedder, top_k=CANDIDATES)
        row = {"query": query, "window": window, "retrieval": inside(hits[0].best_ts, window)}
        for name, labels, rank in (
            ("support", hand_labels, "support"),
            ("gate", hand_labels, "gate"),
            ("gate_query_labels", query_labels, "gate"),
        ):
            t0 = time.perf_counter()
            ranked, stats = verify_hits(hits, detector, labels, rank=rank)
            if name == "gate_query_labels":
                seconds.append(time.perf_counter() - t0)
                frames += stats.frames_detected
                video_frames = stats.video_frames
            row[name] = inside(ranked[0].best_ts, window)
        for k in totals:
            totals[k] += row[k]
        rows.append(row)
        print(
            f"{query[:48]:48s} retrieval {'OK ' if row['retrieval'] else 'BAD'} "
            f"support {'OK ' if row['support'] else 'BAD'} gate {'OK ' if row['gate'] else 'BAD'} "
            f"gate+query-labels {'OK ' if row['gate_query_labels'] else 'BAD'}"
        )

    n = len(CASES)
    summary = {
        "top1": {k: f"{v}/{n}" for k, v in totals.items()},
        "detector_frames_per_query": frames // n,
        "video_frames": video_frames,
        "detector_fraction": round(frames / n / video_frames, 4),
        "verify_seconds_per_query": round(sum(seconds) / n, 1),
    }
    print(json.dumps(summary, indent=2))
    out = Path("eval/results.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
