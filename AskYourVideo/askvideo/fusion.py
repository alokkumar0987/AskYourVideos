"""Merge the visual and speech rankings with Reciprocal Rank Fusion.

SigLIP cosines (~0.1) and text-model scores live on different, uncalibrated
scales, so adding scores would let whichever modality has the wider spread
win. RRF uses only ranks: score(s) = sum over rankings of w / (k + rank).

Fusion is two-level so each modality gets one vote: speech's dense and BM25
rankings are fused first, then that speech ranking is fused with vision.
Fusing all three flat would give speech two votes to vision's one.

How much each modality's vote counts is the ``strategy``:

- "equal": one vote each. Measured to cost top-1 on visual-only queries,
  because speech pulls up segments where nothing relevant is visible.
- "weighted": speech's vote is scaled by a fixed ``speech_weight``.
- "adaptive": per query, each modality's weight is its confidence: how far its
  top score stands above the rest of its scores (a z-score). A query about
  something only said produces a sharp speech peak and a flat visual one, and
  vice versa. No parameter to tune.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from askvideo.index import Hit, VideoIndex

MODES = ("vision", "speech", "fused")
STRATEGIES = ("equal", "weighted", "adaptive")
RRF_K = 60  # the standard constant; dampens the gap between rank 1 and rank 2


def rrf(
    rankings: Sequence[Sequence[int]], k: int = RRF_K, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse rankings of ids (best first) into one, best first. Ties keep first-seen order."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def confidence(scores: Sequence[float]) -> float:
    """How far the best score stands above the others, in standard deviations (>= 0)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size < 2 or s.std() == 0:
        return 0.0
    return max(0.0, float((s.max() - s.mean()) / s.std()))


def modality_weights(
    vision_scores: Sequence[float],
    speech_scores: Sequence[float],
    strategy: str = "equal",
    speech_weight: float = 1.0,
) -> tuple[float, float]:
    """(vision weight, speech weight) for one query; they sum to 2 so "equal" is (1, 1)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "equal":
        return 1.0, 1.0
    if strategy == "weighted":
        return 1.0, speech_weight
    cv, cs = confidence(vision_scores), confidence(speech_scores)
    if cv + cs == 0:
        return 1.0, 1.0
    return 2 * cv / (cv + cs), 2 * cs / (cv + cs)


def fuse(
    vision: tuple[Sequence[int], Sequence[float]],
    dense: tuple[Sequence[int], Sequence[float]],
    lexical: Sequence[int],
    strategy: str = "equal",
    speech_weight: float = 1.0,
) -> tuple[list[tuple[int, float]], list[int], tuple[float, float]]:
    """Fused (id, score) list, the speech ranking, and the weights used.

    ``vision`` and ``dense`` are (ids, scores) best first; ``lexical`` is BM25 ids.
    Pure function of precomputed rankings, so evaluation can try strategies cheaply.
    """
    speech_ranking = [i for i, _ in rrf([dense[0], lexical])]
    weights = modality_weights(vision[1], dense[1], strategy, speech_weight)
    return rrf([vision[0], speech_ranking], weights=weights), speech_ranking, weights


def search(
    index: VideoIndex,
    query: str,
    embedder,
    speech=None,
    text_embedder=None,
    mode: str = "fused",
    top_k: int = 5,
    strategy: str = "equal",
    speech_weight: float = 1.0,
    phonetic: bool = False,
) -> tuple[list[Hit], dict[int, dict[str, int]]]:
    """Retrieve segments by ``mode``; also return each hit's rank per source.

    In "speech" and "fused" mode ``Hit.score`` is the RRF score, not a cosine.
    The best frame of every hit is still the frame closest to the query.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    q = index.embed_query(query, embedder)
    if mode == "vision":
        ids, scores = index.rank_segments(q, top_k)
        hits = [index.make_hit(i, s, q) for i, s in zip(ids, scores)]
        return hits, {h.segment_id: {"vision": r} for r, h in enumerate(hits, 1)}
    if speech is None or text_embedder is None:
        raise ValueError(f"mode {mode!r} needs a speech index; index with --speech first")

    dense_ids, dense_scores, lexical = speech.rank(query, text_embedder, phonetic=phonetic)
    if mode == "speech":
        speech_ranking = [i for i, _ in rrf([dense_ids, lexical])]
        fused, vision = rrf([speech_ranking]), []
    else:
        vision, vision_scores = index.rank_segments(q)
        fused, speech_ranking, _ = fuse(
            (vision, vision_scores), (dense_ids, dense_scores), lexical, strategy, speech_weight
        )

    where = {
        "vision": {s: r for r, s in enumerate(vision, 1)},
        "speech": {s: r for r, s in enumerate(speech_ranking, 1)},
    }
    hits, ranks = [], {}
    for sid, score in fused[:top_k]:
        hits.append(index.make_hit(sid, score, q))
        ranks[sid] = {src: pos[sid] for src, pos in where.items() if sid in pos}
    return hits, ranks
