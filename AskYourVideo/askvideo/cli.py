"""Command-line entry point: `askvideo index` and `askvideo search`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from askvideo import __version__
from askvideo.embedder import MODELS
from askvideo.index import VideoIndex, hits_to_json

DEFAULT_INDEX = ".askvideo"


def _fmt(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="askvideo", description="Search videos by meaning.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ix = sub.add_parser("index", help="Embed videos into a searchable index")
    ix.add_argument("videos", nargs="+", help="Video files to index")
    ix.add_argument("-i", "--index-dir", default=DEFAULT_INDEX)
    ix.add_argument("-m", "--model", default="siglip2", choices=sorted(MODELS))
    ix.add_argument("--fps", type=float, default=1.0, help="Frames sampled per second (default 1)")
    ix.add_argument(
        "--segment", type=float, default=4.0, help="Segment length in seconds (default 4)"
    )
    sp = ix.add_argument_group("speech")
    sp.add_argument("--speech", action="store_true", help="Also transcribe audio (Whisper) and index it")
    sp.add_argument("--asr-model", default="large-v3-turbo", help="faster-whisper model (default large-v3-turbo)")
    sp.add_argument(
        "--text-model", default="intfloat/multilingual-e5-small", help="Embedder for transcript text"
    )

    se = sub.add_parser("search", help="Find the segments that best match a text query")
    se.add_argument("query")
    se.add_argument("-i", "--index-dir", default=DEFAULT_INDEX)
    se.add_argument("-k", "--top-k", type=int, default=5)
    se.add_argument("--json", action="store_true", help="Print hits as JSON")
    se.add_argument("--save-frames", metavar="DIR", help="Save the best frame of each hit here")
    se.add_argument(
        "--mode",
        choices=("vision", "speech", "fused"),
        help="What to search: frames, transcript, or both via RRF (default: fused if the index has speech)",
    )
    se.add_argument(
        "--fusion",
        choices=("adaptive", "equal", "weighted"),
        default="adaptive",
        help="How fused mode weighs the modalities: per-query confidence (default), one vote each, "
        "or a fixed --speech-weight",
    )
    se.add_argument("--speech-weight", type=float, default=1.0, help="Speech vote for --fusion weighted")

    vg = se.add_argument_group("verification (stage 2)")
    vg.add_argument(
        "--verify",
        nargs="+",
        metavar="LABEL",
        help="Confirm hits with YOLOE; all labels must appear together, e.g. --verify person biscuit. "
        "Use only objects named in the query, so usually not for speech-only queries",
    )
    vg.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Segments retrieved for verification before keeping the top -k (default 10)",
    )
    vg.add_argument(
        "--verify-fps", type=float, default=2.0, help="Frames per second checked inside each segment"
    )
    vg.add_argument("--detector", default=None, help="YOLOE weights (default yoloe-26s-seg.pt)")
    vg.add_argument("--conf", type=float, default=0.2, help="Detector confidence threshold")
    vg.add_argument(
        "--rank",
        choices=("gate", "support"),
        default="gate",
        help="gate: verified first, retrieval order kept (default); support: order by detector",
    )

    lv = sub.add_parser("live", help="Track objects on a webcam and record a searchable session")
    lv.add_argument("--source", default="0", help="Camera index or video path (default 0)")
    lv.add_argument(
        "--detect", nargs="*", default=["person"], metavar="LABEL", help="Objects to track"
    )
    lv.add_argument("--watch", nargs="+", default=[], metavar="QUERY", help="Scene queries to alert on")
    lv.add_argument("--watch-threshold", type=float, default=0.10)
    lv.add_argument("-i", "--index-dir", default=".askvideo_live")
    lv.add_argument("-m", "--model", default="siglip2", choices=sorted(MODELS))
    lv.add_argument("--record-dir", default="recordings")
    lv.add_argument("--no-index", action="store_true", help="Detection only: no scene model, max FPS")
    lv.add_argument("--duration", type=float, help="Stop after this many seconds")
    lv.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Detector input size (default 640; 416 lost close-up faces in testing)",
    )
    lv.add_argument("--conf", type=float, default=0.3)
    lv.add_argument("--detector", default=None, help="YOLOE weights (default yoloe-26s-seg.pt)")
    lv.add_argument("--no-window", action="store_true", help="Run without a preview window")
    lv.add_argument("--snapshot", help="Save the last annotated frame to this path")
    return p


def _cmd_index(args) -> int:
    from askvideo.embedder import OpenClipEmbedder

    if VideoIndex.exists(args.index_dir):
        index = VideoIndex.load(args.index_dir)
        if index.model != args.model:
            print(
                f"error: {args.index_dir} uses model {index.model!r}; "
                f"pick another --index-dir for {args.model!r}",
                file=sys.stderr,
            )
            return 2
    else:
        index = VideoIndex(args.model, fps=args.fps, segment_sec=args.segment)

    print(f"loading {args.model} ...", file=sys.stderr)
    embedder = OpenClipEmbedder(args.model)
    for video in args.videos:
        name = Path(video).name
        try:
            n = index.add_video(
                video,
                embedder,
                # One line per 25 frames: `\r` redraws turn into floods of lines in some consoles.
                progress=lambda k: k % 25 == 0 and print(f"  {name}: {k} frames", file=sys.stderr),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if n == 0:
            print(f"  {name}: already indexed, skipped", file=sys.stderr)
        else:
            s = index.stats[str(Path(video).resolve())]
            print(
                f"  {name}: {s['frames_embedded']} frames -> {n} segments "
                f"in {s['index_seconds']}s",
                file=sys.stderr,
            )
    index.save(args.index_dir)
    print(f"saved index -> {args.index_dir}", file=sys.stderr)
    if args.speech:
        return _index_speech(args, index)
    return 0


def _index_speech(args, index: VideoIndex) -> int:
    from askvideo.speech import SpeechIndex, TextEmbedder, transcribe

    speech = (
        SpeechIndex.load(args.index_dir)
        if SpeechIndex.exists(args.index_dir)
        else SpeechIndex(text_model=args.text_model)
    )
    todo = [v for v in args.videos if not speech.has_video(v)]
    for v in set(args.videos) - set(todo):
        print(f"  {Path(v).name}: speech already indexed, skipped", file=sys.stderr)
    if not todo:
        return 0
    print(f"loading {speech.text_model} ...", file=sys.stderr)
    text_embedder = TextEmbedder(speech.text_model)
    for video in todo:
        name = Path(video).name
        print(f"  {name}: transcribing with {args.asr_model} ...", file=sys.stderr)
        t = transcribe(video, args.asr_model)
        n = speech.add_video(index, video, t, text_embedder)
        print(
            f"  {name}: language {t.language} ({t.language_prob:.2f}), {len(t.words)} words, "
            f"{n} segments with speech, {t.dropped_segments} low-confidence lines dropped, "
            f"{t.seconds}s for {t.audio_sec}s of audio (RTF {t.rtf:.2f})",
            file=sys.stderr,
        )
        speech.save(args.index_dir)  # after each video: a crash keeps finished transcripts
    print(f"saved speech index -> {args.index_dir}", file=sys.stderr)
    return 0


def _retrieve(args, index: VideoIndex, embedder, top_k: int):
    """Hits for ``args.query`` in ``args.mode`` (resolved here), plus per-source ranks."""
    from askvideo import fusion
    from askvideo.speech import SpeechIndex, TextEmbedder

    has_speech = SpeechIndex.exists(args.index_dir)
    mode = args.mode or ("fused" if has_speech else "vision")
    if mode != "vision" and not has_speech:
        raise ValueError(f"--mode {mode} needs speech; run `askvideo index ... --speech` first")
    speech = text_embedder = None
    if mode != "vision":
        speech = SpeechIndex.load(args.index_dir)
        text_embedder = TextEmbedder(speech.text_model)
    print(f"mode: {mode}" + (f" ({args.fusion})" if mode == "fused" else ""), file=sys.stderr)
    return fusion.search(
        index, args.query, embedder, speech, text_embedder, mode=mode, top_k=top_k,
        strategy=args.fusion, speech_weight=args.speech_weight,
    )


def _cmd_search(args) -> int:
    from askvideo.embedder import OpenClipEmbedder

    if not VideoIndex.exists(args.index_dir):
        print(f"error: no index at {args.index_dir}; run `askvideo index` first", file=sys.stderr)
        return 2
    index = VideoIndex.load(args.index_dir)
    if args.verify:
        return _search_verified(args, index, OpenClipEmbedder(index.model))
    try:
        hits, ranks = _retrieve(args, index, OpenClipEmbedder(index.model), args.top_k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(hits_to_json(hits))
    else:
        for rank, h in enumerate(hits, 1):
            src = ", ".join(f"{k} #{r}" for k, r in ranks.get(h.segment_id, {}).items())
            print(
                f"{rank:>2}. {h.score:+.3f}  {Path(h.video).name}  "
                f"{_fmt(h.start)}-{_fmt(h.end)}  (best frame {_fmt(h.best_ts)})  [{src}]"
            )

    if args.save_frames:
        import cv2

        from askvideo.frames import read_frame_at
        from askvideo.visualize import render_hit

        out = Path(args.save_frames)
        out.mkdir(parents=True, exist_ok=True)
        for rank, h in enumerate(hits, 1):
            rgb = read_frame_at(h.video, h.best_ts)
            if rgb is not None:
                path = out / f"{rank:02d}_{Path(h.video).stem}_{h.best_ts:.0f}s.jpg"
                img = render_hit(rgb, (), query=args.query, ts=h.best_ts, footnote=Path(h.video).name)
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"saved frames -> {out}", file=sys.stderr)
    return 0


def _search_verified(args, index: VideoIndex, embedder) -> int:
    import json

    import cv2

    from askvideo.detect import DEFAULT_MODEL, YoloeDetector
    from askvideo.frames import read_frame_at
    from askvideo.verify import verify_hits
    from askvideo.visualize import render_hit

    try:
        hits, _ = _retrieve(args, index, embedder, max(args.candidates, args.top_k))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    retrieval_rank = {h.segment_id: r for r, h in enumerate(hits, 1)}
    print(f"verifying {len(hits)} candidates for {args.verify} ...", file=sys.stderr)
    detector = YoloeDetector(args.detector or DEFAULT_MODEL, conf=args.conf)
    ranked, stats = verify_hits(hits, detector, args.verify, fps=args.verify_fps, rank=args.rank)
    top = ranked[: args.top_k]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "video": v.hit.video,
                        "start": v.hit.start,
                        "end": v.hit.end,
                        "best_ts": v.best_ts,
                        "verified": v.verified,
                        "support": round(v.support, 4),
                        "evidence": round(v.evidence, 4),
                        "retrieval_score": round(v.hit.score, 4),
                        "retrieval_rank": retrieval_rank[v.hit.segment_id],
                        "counts": v.counts,
                        "boxes": [
                            {"label": b.label, "conf": round(b.conf, 3), "xyxy": b.xyxy}
                            for b in v.boxes
                        ],
                    }
                    for v in top
                ],
                indent=2,
            )
        )
    else:
        for rank, v in enumerate(top, 1):
            mark = "OK " if v.verified else "-- "
            counts = ", ".join(f"{k}x{n}" for k, n in sorted(v.counts.items()))
            print(
                f"{rank:>2}. {mark} score {v.hit.score:+.3f} support {v.support:.2f}  {Path(v.hit.video).name}  "
                f"{_fmt(v.hit.start)}-{_fmt(v.hit.end)}  (best frame {_fmt(v.best_ts)}, "
                f"was #{retrieval_rank[v.hit.segment_id]})  {counts}"
            )
    print(
        f"detector ran on {stats.frames_detected} of {stats.video_frames} frames "
        f"({stats.fraction:.2%}) in {stats.detect_seconds}s",
        file=sys.stderr,
    )

    if args.save_frames:
        out = Path(args.save_frames)
        out.mkdir(parents=True, exist_ok=True)
        for rank, v in enumerate(top, 1):
            rgb = read_frame_at(v.hit.video, v.best_ts)
            if rgb is not None:
                path = out / f"{rank:02d}_{Path(v.hit.video).stem}_{v.best_ts:.0f}s.jpg"
                img = render_hit(
                    rgb, v.boxes if v.verified else (), query=args.query, ts=v.best_ts,
                    labels=args.verify, verified=v.verified,
                    footnote=f"Detector checked {stats.fraction:.2%} of frames  ·  {Path(v.hit.video).name}",
                )
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"saved frames -> {out}", file=sys.stderr)
    return 0


def _cmd_live(args) -> int:
    import time

    from askvideo.detect import DEFAULT_MODEL, YoloeDetector
    from askvideo.live import run_live

    if args.no_index and args.watch:
        print("error: --watch needs the scene model; drop --no-index", file=sys.stderr)
        return 2
    index = None
    if not args.no_index and VideoIndex.exists(args.index_dir):
        index = VideoIndex.load(args.index_dir)
        if index.model != args.model:
            print(f"error: {args.index_dir} uses model {index.model!r}", file=sys.stderr)
            return 2

    embedder = None
    if not args.no_index:
        from askvideo.embedder import OpenClipEmbedder

        print(f"loading {args.model} ...", file=sys.stderr)
        embedder = OpenClipEmbedder(args.model)
    detector = None
    if args.detect:
        print("loading detector ...", file=sys.stderr)
        detector = YoloeDetector(args.detector or DEFAULT_MODEL, conf=args.conf, imgsz=args.imgsz)

    source = int(args.source) if args.source.isdigit() else args.source
    record = Path(args.record_dir) / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    print(f"live on {source!r} -> recording {record}" + ("" if args.no_window else "  (press q to stop)"), file=sys.stderr)
    try:
        result = run_live(
            source,
            detector,
            args.detect,
            record,
            embedder=embedder,
            watch=args.watch,
            watch_threshold=args.watch_threshold,
            duration=args.duration,
            show=not args.no_window,
            snapshot=args.snapshot,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"session {result.seconds:.1f}s: video {result.fps:.1f} FPS, "
        f"detector {result.detect_fps:.1f} FPS, recorded -> {result.record_path}",
        file=sys.stderr,
    )
    for a in result.alerts:
        print(f"  alert at {_fmt(a.ts)}: {a.query!r} ({a.score:+.3f})", file=sys.stderr)
    if result.embeddings is not None:
        index = index or VideoIndex(args.model)
        n = index.add_embedded(str(result.record_path), result.frames, result.embeddings)
        index.save(args.index_dir)
        print(
            f"indexed {len(result.frames)} frames -> {n} segments in {args.index_dir}; "
            f'search it with: askvideo search "..." -i {args.index_dir}',
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return {"index": _cmd_index, "search": _cmd_search, "live": _cmd_live}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
