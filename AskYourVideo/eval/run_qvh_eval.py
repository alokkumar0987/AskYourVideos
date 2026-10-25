"""Zero-shot moment retrieval on a QVHighlights (val) subset.

    python eval/qvh_prepare.py --n 50                      # clips + subset.jsonl
    askvideo index <clips> -i .askvideo_qvh --speech         # see docs/TECHNICAL_REPORT.md
    python eval/run_qvh_eval.py

Nothing here is trained on QVHighlights: SigLIP 2, Whisper and e5 are used
off the shelf, so these numbers are zero-shot and not comparable one-to-one
with models trained on its train split.

Per query (moment localization inside its own clip; the library holds all
clips and results are filtered to the query's clip):
    Hit@1, Hit@5   a top-k 4 s segment overlaps a ground-truth window
    R1@0.5, R1@0.7 IoU of the predicted window with the best-matching ground
                   truth window, the standard QVHighlights moment metric
    Lib@1          the single best segment across the *whole library* is in
                   the right clip and overlaps a window (search without
                   knowing which video)
Predicted window: the top-1 segment grown over contiguous neighbours that also
rank in the clip's top EXPAND_FRAC of segments. The fraction is fixed up front
(not tuned on these queries).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from askvideo import fusion
from askvideo.embedder import OpenClipEmbedder
from askvideo.index import VideoIndex
from askvideo.speech import SpeechIndex, TextEmbedder

ROOT = Path("data/qvhighlights")
EXPAND_FRAC = 0.2
METRICS = ("Hit@1", "Hit@5", "R1@0.5", "R1@0.7", "Lib@1")


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def overlaps(seg: dict, windows) -> bool:
    return any(seg["start"] < e and seg["end"] > s for s, e in windows)


def predicted_window(ranked: list[int], index: VideoIndex) -> tuple[float, float]:
    """Top-1 segment grown over contiguous neighbours that are also highly ranked in the clip."""
    keep = set(ranked[: max(1, round(EXPAND_FRAC * len(ranked)))])
    by_start = {index.segments[i]["start"]: i for i in ranked}
    top = index.segments[ranked[0]]
    start, end = top["start"], top["end"]
    while (prev := by_start.get(start - index.segment_sec)) in keep:
        start = index.segments[prev]["start"]
    while (nxt := by_start.get(end)) in keep:
        end = index.segments[nxt]["end"]
    return start, end


def score(ranked_global: list[int], index: VideoIndex, video: str, windows) -> dict[str, float]:
    ranked = [i for i in ranked_global if index.segments[i]["video"] == video]
    if not ranked:
        return dict.fromkeys(METRICS, 0.0)
    segs = [index.segments[i] for i in ranked]
    window = predicted_window(ranked, index)
    best_iou = max(iou(window, w) for w in windows)
    top = index.segments[ranked_global[0]]
    return {
        "Hit@1": float(overlaps(segs[0], windows)),
        "Hit@5": float(any(overlaps(s, windows) for s in segs[:5])),
        "R1@0.5": float(best_iou >= 0.5),
        "R1@0.7": float(best_iou >= 0.7),
        "Lib@1": float(top["video"] == video and overlaps(top, windows)),
    }


def bootstrap_ci(values: list[float], resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean, by resampling queries with replacement.

    With few queries a single number hides how much it could move; the interval
    says how far (e.g. 50 queries at 30% is roughly 17-43%).
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def random_baseline(index: VideoIndex, cases, trials: int = 200) -> dict[str, float]:
    """Expected metrics if segments were ranked at random (same window rule)."""
    rng = random.Random(0)
    totals = defaultdict(float)
    all_ids = list(range(len(index.segments)))
    for video, windows in cases:
        for _ in range(trials):
            ranked = rng.sample(all_ids, len(all_ids))
            for k, v in score(ranked, index, video, windows).items():
                totals[k] += v / trials
    return {k: totals[k] / len(cases) for k in METRICS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--index-dir", default=".askvideo_qvh")
    p.add_argument("--out", default="eval/qvh_results.json")
    args = p.parse_args()

    index = VideoIndex.load(args.index_dir)
    speech = SpeechIndex.load(args.index_dir) if SpeechIndex.exists(args.index_dir) else None
    embedder = OpenClipEmbedder(index.model)
    text_embedder = TextEmbedder(speech.text_model) if speech else None
    indexed = set(index.videos)

    queries = [json.loads(line) for line in (ROOT / "subset.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = []
    for q in queries:
        video = str((ROOT / "videos" / f"{q['vid']}.mp4").resolve())
        if video in indexed:
            cases.append((q, video, q["relevant_windows"]))
    if not cases:
        raise SystemExit("no indexed clips; run the index step first")

    systems = ["vision"] + (["speech", "fused-equal", "fused-adaptive"] if speech else [])
    totals = {s: defaultdict(float) for s in systems}
    rows = []
    for q, video, windows in cases:
        qv = index.embed_query(q["query"], embedder)
        vision = index.rank_segments(qv)
        rankings = {"vision": vision[0]}
        if speech:
            dense_ids, dense_scores, lexical = speech.rank(q["query"], text_embedder)
            speech_rank = [i for i, _ in fusion.rrf([dense_ids, lexical])]
            # segments without speech still need a place in the speech ranking
            rest = [i for i in vision[0] if i not in set(speech_rank)]
            rankings["speech"] = speech_rank + rest
            for strategy in ("equal", "adaptive"):
                fused, _, _ = fusion.fuse(vision, (dense_ids, dense_scores), lexical, strategy)
                rankings[f"fused-{strategy}"] = [i for i, _ in fused]
        row = {"qid": q["qid"], "query": q["query"], "vid": q["vid"]}
        for system, ranked in rankings.items():
            m = score(ranked, index, video, windows)
            row[system] = m
            for k, v in m.items():
                totals[system][k] += v
        rows.append(row)

    n = len(cases)
    table = {s: {k: round(totals[s][k] / n, 3) for k in METRICS} for s in systems}
    ci = {
        s: {k: [round(x, 3) for x in bootstrap_ci([r[s][k] for r in rows])] for k in METRICS}
        for s in systems
    }
    table["random"] = {k: round(v, 3) for k, v in random_baseline(index, [(v, w) for _, v, w in cases]).items()}
    print(f"QVHighlights val subset: {n} queries over {len(indexed)} clips (zero-shot)")
    print("cells: mean [95% bootstrap interval]\n")
    print(f"{'system':16s} " + " ".join(f"{k:>18s}" for k in METRICS))
    for system, m in table.items():
        cells = [
            f"{m[k]:.2f} [{ci[system][k][0]:.2f}-{ci[system][k][1]:.2f}]" if system in ci else f"{m[k]:.2f}"
            for k in METRICS
        ]
        print(f"{system:16s} " + " ".join(f"{c:>18s}" for c in cells))
    Path(args.out).write_text(
        json.dumps(
            {"n_queries": n, "n_clips": len(indexed), "expand_frac": EXPAND_FRAC, "table": table, "ci95": ci, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
