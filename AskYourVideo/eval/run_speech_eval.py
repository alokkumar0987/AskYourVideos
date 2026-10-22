"""Vision vs speech vs fusion strategies: R@1 and R@5 per query type and video.

    python eval/run_speech_eval.py [-i .askvideo_speech]

Cases live in eval/speech_cases.json:
    {"video": "samples/x.mp4", "query": "...", "type": "visual|spoken|mixed", "window": [start, end]}
"window" may also be a list of [start, end] pairs when the moment occurs more than once.
Cases with a null window are skipped (not annotated yet). A hit is correct when
its segment overlaps a window. The whole library is searched and then filtered
to the case's video, so the other videos act as distractors.

Systems compared:
    vision, speech        one modality alone
    fused-equal           RRF, one vote each
    fused-weighted (LOO)  speech weight chosen on all *other* cases (leave-one-out),
                          so the reported number is never tuned on the case it scores
    fused-adaptive        per-query weights from each modality's score peak; nothing tuned
    fused-adaptive+phon.  the same, with Hinglish skeleton matching in BM25 (speech.index_tokens)
Small, hand-annotated set: report it as such.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from askvideo import fusion
from askvideo.embedder import OpenClipEmbedder
from askvideo.index import VideoIndex
from askvideo.speech import SpeechIndex, TextEmbedder

KS = (1, 5)
WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SYSTEMS = ("vision", "speech", "fused-equal", "fused-weighted (LOO)", "fused-adaptive", "fused-adaptive+phon.")


def _correct(ranked_ids, index, video, windows) -> dict[int, bool]:
    segs = [index.segments[i] for i in ranked_ids if index.segments[i]["video"] == video]
    return {
        k: any(s["start"] < hi and s["end"] > lo for s in segs[:k] for lo, hi in windows) for k in KS
    }


def _objective(results: list[dict[int, bool]]) -> float:
    return sum(r[1] + 0.5 * r[5] for r in results)  # top-1 first, recall@5 as tie-break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--index-dir", default=".askvideo_speech")
    p.add_argument("--cases", default="eval/speech_cases.json")
    p.add_argument("--out", default="eval/speech_results.json")
    args = p.parse_args()

    cases = [c for c in json.loads(Path(args.cases).read_text(encoding="utf-8")) if c.get("window")]
    if not cases:
        raise SystemExit(f"no annotated cases in {args.cases}")
    index, speech = VideoIndex.load(args.index_dir), SpeechIndex.load(args.index_dir)
    embedder, text_embedder = OpenClipEmbedder(index.model), TextEmbedder(speech.text_model)

    # 1) rankings once per case; every system below is computed from these
    per_case = []
    for c in cases:
        q = index.embed_query(c["query"], embedder)
        vision = index.rank_segments(q)
        dense_ids, dense_scores, lexical = speech.rank(c["query"], text_embedder)
        phon_lexical = speech.rank(c["query"], text_embedder, phonetic=True)[2]
        windows = c["window"] if isinstance(c["window"][0], list) else [c["window"]]
        per_case.append(
            (c, str(Path(c["video"]).resolve()), windows, vision, (dense_ids, dense_scores), lexical, phon_lexical)
        )

    def run(i, strategy, w=1.0, phonetic=False):
        c, video, windows, vision, dense, lexical, phon_lexical = per_case[i]
        fused, _, weights = fusion.fuse(vision, dense, phon_lexical if phonetic else lexical, strategy, w)
        return _correct([s for s, _ in fused], index, video, windows), weights

    n = len(per_case)
    weighted = {w: [run(i, "weighted", w)[0] for i in range(n)] for w in WEIGHT_GRID}

    # 2) score every system per case
    hits: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rows, chosen = [], Counter()
    for i, (c, video, windows, vision, dense, lexical, _) in enumerate(per_case):
        speech_ranking = [s for s, _ in fusion.rrf([dense[0], lexical])]
        others = [j for j in range(n) if j != i]
        best_w = max(WEIGHT_GRID, key=lambda w: _objective([weighted[w][j] for j in others]))
        chosen[best_w] += 1
        adaptive, ad_weights = run(i, "adaptive")
        result = {
            "vision": _correct(vision[0], index, video, windows),
            "speech": _correct(speech_ranking, index, video, windows),
            "fused-equal": run(i, "equal")[0],
            "fused-weighted (LOO)": weighted[best_w][i],
            "fused-adaptive": adaptive,
            "fused-adaptive+phon.": run(i, "adaptive", phonetic=True)[0],
        }
        for system, ok in result.items():
            for group in ("all", f"type:{c['type']}", f"video:{Path(video).name}"):
                hits[system][group].append(ok)
        rows.append(
            {**c, "loo_speech_weight": best_w, "adaptive_weights": [round(x, 2) for x in ad_weights]}
            | {s: {f"R@{k}": ok[k] for k in KS} for s, ok in result.items()}
        )
        marks = "  ".join(f"{'✓' if ok[1] else '·'}{'✓' if ok[5] else '·'}" for ok in result.values())
        print(f"[{c['type']:6s}] {c['query'][:46]:46s} {marks}   w_s(adaptive)={ad_weights[1]:.2f}")

    groups = sorted({g for s in hits.values() for g in s}, key=lambda g: (g != "all", g))
    table = {
        g: {
            s: {f"R@{k}": round(sum(r[k] for r in hits[s][g]) / len(hits[s][g]), 3) for k in KS}
            | {"n": len(hits[s][g])}
            for s in SYSTEMS
        }
        for g in groups
    }
    print("\ncolumns: " + " | ".join(SYSTEMS) + "   (cells are R@1 / R@5)")
    for g, by in table.items():
        cells = "  ".join(f"{v['R@1']:.2f}/{v['R@5']:.2f}" for v in by.values())
        print(f"{g + ' (n=' + str(by['vision']['n']) + ')':30s} {cells}")
    print(f"LOO-chosen speech weights: {dict(sorted(chosen.items()))}")
    Path(args.out).write_text(
        json.dumps({"table": table, "loo_weights": dict(chosen), "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
