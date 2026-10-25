"""Browser UI: search an index, add videos to it, and run a live webcam session.

    askvideo-ui            # then open http://127.0.0.1:7860

Everything runs locally; models load once on first use and stay in memory.
"""

from __future__ import annotations

import html
import threading
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from askvideo.index import VideoIndex

DEFAULT_MODEL = "siglip2"
RECORD_DIR = Path("recordings")

EXAMPLES = [
    ["a man eating a biscuit", "person, biscuit"],
    ["cutting dough with a biscuit cutter", "biscuit cutter"],
    ["people walking on a street", "person"],
]


def _fmt(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def _index_dirs() -> list[str]:
    """Libraries in the working directory, largest first so the default is the main one."""
    found = sorted(Path(".").glob(".askvideo*/meta.json"), key=lambda p: -p.with_name("embeddings.npz").stat().st_size)
    return [p.parent.name for p in found] or [".askvideo"]


def _labels(text: str) -> list[str]:
    return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]


@lru_cache(maxsize=2)
def _embedder(model: str):
    from askvideo.embedder import OpenClipEmbedder

    return OpenClipEmbedder(model)


@lru_cache(maxsize=4)
def _detector(conf: float, imgsz: int = 640):
    from askvideo.detect import DEFAULT_MODEL as WEIGHTS
    from askvideo.detect import YoloeDetector

    return YoloeDetector(WEIGHTS, conf=conf, imgsz=imgsz)


@lru_cache(maxsize=1)
def _text_embedder(name: str):
    from askvideo.speech import TextEmbedder

    return TextEmbedder(name)


def _has_speech(index_dir: str) -> bool:
    from askvideo.speech import SpeechIndex

    return SpeechIndex.exists(index_dir)


def _retrieve(index_dir: str, index: VideoIndex, query: str, embedder, mode: str, top_k: int):
    """Hits, per-source ranks and the mode actually used (vision when there is no speech)."""
    from askvideo import fusion
    from askvideo.speech import SpeechIndex

    mode = mode.lower()
    if mode == "vision" or not _has_speech(index_dir):
        hits, ranks = fusion.search(index, query, embedder, mode="vision", top_k=top_k)
        return hits, ranks, "vision"
    speech = SpeechIndex.load(index_dir)
    hits, ranks = fusion.search(
        index, query, embedder, speech, _text_embedder(speech.text_model), mode=mode, top_k=top_k,
        strategy="adaptive",  # best overall top-1 in eval/run_speech_eval.py, nothing tuned
    )
    return hits, ranks, mode


def _sources(ranks: dict[str, int]) -> str:
    return " · ".join(f"{k} #{r}" for k, r in ranks.items())


def _load_or_new(index_dir: str) -> VideoIndex:
    return VideoIndex.load(index_dir) if VideoIndex.exists(index_dir) else VideoIndex(DEFAULT_MODEL)


# ---------------------------------------------------------------- html bits


def _stats(*items: tuple[str, str]) -> str:
    cells = "".join(
        f'<div class="av-stat"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
        for k, v in items
    )
    return f'<div class="av-stats">{cells}</div>'


def _note(text: str, kind: str = "info") -> str:
    return f'<div class="av-note av-{kind}">{text}</div>'


def _index_summary(index_dir: str) -> str:
    if not VideoIndex.exists(index_dir):
        return _stats(("Index", index_dir), ("Status", "empty"), ("Videos", "0"))
    index = VideoIndex.load(index_dir)
    seconds = sum(s.get("duration_sec", 0) for s in index.stats.values())
    speech = "—"
    if _has_speech(index_dir):
        from askvideo.speech import SpeechIndex

        speech = f"{len(SpeechIndex.load(index_dir).transcripts)}/{len(index.videos)}"
    return _stats(
        ("Model", index.model),
        ("Videos", str(len(index.videos))),
        ("Footage", _fmt(seconds)),
        ("Segments", str(len(index.segments))),
        ("Speech", speech),
    )


def _alerts_html(alerts) -> str:
    if not alerts:
        return '<div class="av-empty">No alerts yet</div>'
    items = "".join(
        f'<li><span class="av-time">{_fmt(a.ts)}</span>{html.escape(a.query)}'
        f'<span class="av-score">{a.score:+.2f}</span></li>'
        for a in reversed(alerts)
    )
    return f'<ul class="av-alerts">{items}</ul>'


# ---------------------------------------------------------------- search


def search(index_dir: str, query: str, verify: str, top_k: int, mode: str = "Vision"):
    from askvideo.frames import read_frame_at
    from askvideo.visualize import render_hit

    if not query.strip():
        raise gr.Error("Describe the moment you are looking for.")
    if not VideoIndex.exists(index_dir):
        raise gr.Error(f"No index at {index_dir}. Add a video first.")
    index = VideoIndex.load(index_dir)
    embedder = _embedder(index.model)
    labels = _labels(verify)
    t0 = time.perf_counter()

    gallery, rows = [], []
    if labels:
        from askvideo.verify import verify_hits

        hits, ranks, used = _retrieve(index_dir, index, query, embedder, mode, max(10, top_k))
        ranked, stats = verify_hits(hits, _detector(0.2), labels)
        top = ranked[:top_k]
        for rank, v in enumerate(top, 1):
            rgb = read_frame_at(v.hit.video, v.best_ts)
            if rgb is not None:
                img = render_hit(
                    rgb, v.boxes if v.verified else (), query=query, ts=v.best_ts, labels=labels,
                    verified=v.verified,
                    footnote=f"{Path(v.hit.video).name}  ·  {_sources(ranks.get(v.hit.segment_id, {}))}",
                )
                mark = "✓ verified" if v.verified else "not verified"
                gallery.append((img, f"#{rank} · {_fmt(v.best_ts)} · {mark}"))
            rows.append(
                [rank, "✓" if v.verified else "—", Path(v.hit.video).name,
                 f"{_fmt(v.hit.start)}–{_fmt(v.hit.end)}", _fmt(v.best_ts), round(v.hit.score, 3)]
            )
        summary = _stats(
            ("Results", str(len(rows))),
            ("Verified", f"{sum(v.verified for v in top)}/{len(top)}"),
            ("Frames checked", f"{stats.fraction:.2%}"),
            ("Mode", used),
            ("Time", f"{time.perf_counter() - t0:.1f}s"),
        )
    else:
        hits, ranks, used = _retrieve(index_dir, index, query, embedder, mode, top_k)
        for rank, h in enumerate(hits, 1):
            rgb = read_frame_at(h.video, h.best_ts)
            if rgb is not None:
                foot = f"{Path(h.video).name}  ·  {_sources(ranks.get(h.segment_id, {}))}"
                img = render_hit(rgb, (), query=query, ts=h.best_ts, footnote=foot)
                gallery.append((img, f"#{rank} · {_fmt(h.best_ts)} · {Path(h.video).name}"))
            rows.append(
                [rank, "", Path(h.video).name, f"{_fmt(h.start)}–{_fmt(h.end)}", _fmt(h.best_ts),
                 round(h.score, 3)]
            )
        summary = _stats(
            ("Results", str(len(rows))),
            ("Best match", rows[0][4] if rows else "—"),
            ("Mode", used),
            ("Time", f"{time.perf_counter() - t0:.1f}s"),
        ) + _note("Add objects from your query (e.g. <code>biscuit</code>) to confirm hits with YOLOE.")
    return gallery, rows, summary


# ---------------------------------------------------------------- add video


def add_videos(index_dir: str, files, with_speech: bool = False, progress=gr.Progress()):
    if not files:
        raise gr.Error("Choose one or more video files.")
    index = _load_or_new(index_dir)
    progress(0, desc=f"Loading {index.model} …")
    embedder = _embedder(index.model)
    lines = []
    for i, f in enumerate(files):
        path = f if isinstance(f, str) else f.name
        name = html.escape(Path(path).name)
        progress(i / len(files), desc=f"Indexing {Path(path).name}")
        try:
            n = index.add_video(path, embedder)
        except (FileNotFoundError, ValueError) as exc:
            lines.append(_note(f"<b>{name}</b> — {html.escape(str(exc))}", "error"))
            continue
        s = index.stats.get(str(Path(path).resolve()), {})
        lines.append(
            _note(f"<b>{name}</b> — already in this index", "info") if n == 0
            else _note(
                f"<b>{name}</b> — {s['frames_embedded']} frames → {n} segments "
                f"in {s['index_seconds']}s", "ok",
            )
        )
    index.save(index_dir)
    if with_speech:
        lines += _add_speech(index_dir, index, files, progress)
    return "".join(lines), _index_summary(index_dir)


def _add_speech(index_dir: str, index: VideoIndex, files, progress) -> list[str]:
    from askvideo.speech import SpeechIndex, transcribe

    speech = SpeechIndex.load(index_dir) if SpeechIndex.exists(index_dir) else SpeechIndex()
    lines = []
    for i, f in enumerate(files):
        path = f if isinstance(f, str) else f.name
        name = html.escape(Path(path).name)
        if speech.has_video(path) or str(Path(path).resolve()) not in index.stats:
            continue
        progress(i / len(files), desc=f"Transcribing {Path(path).name} (Whisper)")
        t = transcribe(path)
        n = speech.add_video(index, path, t, _text_embedder(speech.text_model))
        speech.save(index_dir)
        lines.append(_note(
            f"<b>{name}</b> — speech: {t.language}, {len(t.words)} words, {n} segments, "
            f"{t.seconds:.0f}s (RTF {t.rtf:.2f})", "ok",
        ))
    return lines


# ---------------------------------------------------------------- live

_live_stop = threading.Event()
_live_lock = threading.Lock()


def live(index_dir: str, detect: str, watch: str, threshold: float):
    from askvideo.live import run_live

    if not _live_lock.acquire(blocking=False):
        raise gr.Error("A live session is already running.")
    try:
        index = _load_or_new(index_dir)
        idle = gr.update(interactive=False)
        yield None, _note("Loading models (SigLIP 2 + YOLOE) …"), "", idle, idle
        embedder = _embedder(index.model)
        labels = _labels(detect)
        queries = [q.strip() for q in watch.splitlines() if q.strip()]
        detector = _detector(0.3) if labels else None

        state = {"frame": None, "alerts": [], "result": None, "error": None}
        _live_stop.clear()
        record = RECORD_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        def on_frame(rgb, alerts):
            state["frame"], state["alerts"] = rgb, list(alerts)

        def worker():
            try:
                state["result"] = run_live(
                    0, detector, labels, record, embedder=embedder, watch=queries,
                    watch_threshold=threshold, show=False, on_frame=on_frame, stop=_live_stop,
                )
            except Exception as exc:  # surfaced in the UI below
                state["error"] = exc

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        on = gr.update(interactive=True)
        while t.is_alive():
            time.sleep(0.08)
            yield (
                state["frame"],
                '<div class="av-live"><span class="av-dot"></span>Live · recording</div>',
                _alerts_html(state["alerts"]),
                idle,
                on,
            )

        start_again = gr.update(interactive=True)
        if state["error"] is not None:
            yield state["frame"], _note(html.escape(str(state["error"])), "error"), "", start_again, idle
            return
        r = state["result"]
        summary = _stats(
            ("Duration", _fmt(r.seconds)),
            ("Video", f"{r.fps:.1f} FPS"),
            ("Detector", f"{r.detect_fps:.1f}/s"),
            ("Alerts", str(len(r.alerts))),
        )
        if r.embeddings is not None:
            n = index.add_embedded(str(r.record_path), r.frames, r.embeddings)
            index.save(index_dir)
            summary += _note(
                f"Saved <code>{html.escape(str(r.record_path))}</code> and indexed it ({n} segments). "
                "Search it in the <b>Search</b> tab.", "ok",
            )
        yield state["frame"], summary, _alerts_html(r.alerts), start_again, idle
    finally:
        _live_lock.release()


def stop_live():
    _live_stop.set()


# ---------------------------------------------------------------- layout

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#f6f7fb",
    body_background_fill_dark="#0b0d12",
    block_background_fill="#ffffff",
    block_background_fill_dark="#12151c",
    block_border_width="1px",
    block_border_color="#e4e7ee",
    block_border_color_dark="#1f2430",
    block_shadow="0 1px 2px rgba(16, 24, 40, 0.04)",
    block_label_text_weight="600",
    block_title_text_weight="600",
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    button_large_radius="10px",
    input_background_fill_dark="#0f1218",
)

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
footer { display: none !important; }

.av-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  flex-wrap: wrap; padding: 22px 26px; border-radius: 14px; margin-bottom: 4px;
  background: linear-gradient(120deg, #1e1b4b 0%, #312e81 55%, #4338ca 100%); color: #fff; }
.av-brand { display: flex; align-items: center; gap: 14px; }
.av-logo { width: 44px; height: 44px; border-radius: 11px; display: grid; place-items: center;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22); }
.av-brand h1 { margin: 0 !important; font-size: 22px !important; font-weight: 700; color: #fff !important;
  letter-spacing: -0.01em; }
.av-brand p { margin: 2px 0 0 !important; font-size: 13.5px; color: #c7d2fe !important; }
.av-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.av-chip { font-size: 12px; font-weight: 500; padding: 5px 10px; border-radius: 999px;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2); color: #e0e7ff; }

.av-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
.av-stat { padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); }
.av-stat span { display: block; font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--body-text-color-subdued); font-weight: 600; }
.av-stat b { display: block; margin-top: 4px; font-size: 18px; font-weight: 650;
  color: var(--body-text-color); font-variant-numeric: tabular-nums; }

.av-note { margin-top: 10px; padding: 10px 14px; border-radius: 10px; font-size: 13.5px;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.av-info { border-left: 3px solid #6366f1; }
.av-ok { border-left: 3px solid #10b981; }
.av-error { border-left: 3px solid #ef4444; }
.av-note code { font-size: 12.5px; }

.av-section { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .07em;
  color: var(--body-text-color-subdued); margin: 4px 0 -4px; }

.av-live { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px;
  padding: 8px 12px; border-radius: 999px; background: rgba(239,68,68,.1); color: #ef4444; }
.av-dot { width: 9px; height: 9px; border-radius: 50%; background: #ef4444;
  animation: av-pulse 1.2s ease-in-out infinite; }
@keyframes av-pulse { 50% { opacity: .25; } }

.av-alerts { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.av-alerts li { display: flex; align-items: center; gap: 10px; padding: 9px 12px; font-size: 13.5px;
  border-radius: 9px; border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.av-time { font-family: var(--font-mono); font-size: 12px; padding: 2px 7px; border-radius: 6px;
  background: rgba(99,102,241,.12); color: #6366f1; }
.av-score { margin-left: auto; font-family: var(--font-mono); font-size: 12px;
  color: var(--body-text-color-subdued); }
.av-empty { font-size: 13px; color: var(--body-text-color-subdued); padding: 6px 2px; }

.av-library input { color: var(--body-text-color) !important; -webkit-text-fill-color: var(--body-text-color) !important; }

.av-help { font-size: 13px; color: var(--body-text-color-subdued); line-height: 1.55; }
.av-help b { color: var(--body-text-color); }
"""

LOGO = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="14" height="14" rx="2"/>'
    '<path d="m16 10 6-3v10l-6-3"/><circle cx="9" cy="12" r="2.5"/></svg>'
)

HEADER = f"""
<div class="av-header">
  <div class="av-brand">
    <div class="av-logo">{LOGO}</div>
    <div><h1>AskYourVideo</h1><p>Find any moment in your videos by describing it.</p></div>
  </div>
  <div class="av-chips">
    <span class="av-chip">SigLIP 2 retrieval</span>
    <span class="av-chip">YOLOE verification</span>
    <span class="av-chip">ByteTrack live</span>
    <span class="av-chip">Runs locally</span>
  </div>
</div>
"""


def build() -> gr.Blocks:
    dirs = _index_dirs()
    with gr.Blocks(title="AskYourVideo", theme=THEME, css=CSS) as app:
        gr.HTML(HEADER)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260):
                index_dir = gr.Dropdown(
                    dirs, value=dirs[0], label="Video library", allow_custom_value=True,
                    info="Type a new name to start a new library", elem_classes="av-library",
                )
            with gr.Column(scale=3):
                summary = gr.HTML(_index_summary(dirs[0]))
        index_dir.change(_index_summary, index_dir, summary)
        app.load(lambda: gr.update(choices=_index_dirs()), outputs=index_dir, show_progress="hidden")

        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        query = gr.Textbox(
                            label="Describe the moment", placeholder="a man eating a biscuit", lines=2,
                        )
                        verify = gr.Textbox(
                            label="Confirm with objects", placeholder="person, biscuit",
                            info="Optional. Use words from your query; YOLOE checks they appear together.",
                        )
                        top_k = gr.Slider(1, 10, value=3, step=1, label="Number of results")
                        mode = gr.Radio(
                            ["Vision", "Fused", "Speech"], value="Vision", label="Search in",
                            info="Vision is best for what you can see. Use Fused or Speech for what was said.",
                        )
                        go = gr.Button("Search", variant="primary", size="lg")
                        gr.Examples(EXAMPLES, [query, verify], label="Try")
                    with gr.Column(scale=2):
                        status = gr.HTML(
                            _note("Results appear here. Plain search takes about a second; "
                                  "confirming with objects takes ~25 s on CPU.")
                        )
                        gallery = gr.Gallery(
                            label="Best frames", columns=3, height=420, object_fit="contain",
                            show_label=False, preview=False,
                        )
                        with gr.Accordion("Result details", open=False):
                            table = gr.Dataframe(
                                headers=["#", "Verified", "Video", "Segment", "Best frame", "Score"],
                                interactive=False,
                            )
                for trigger in (go.click, query.submit):
                    trigger(search, [index_dir, query, verify, top_k, mode], [gallery, table, status])

            with gr.Tab("Add videos"):
                with gr.Row():
                    with gr.Column(scale=2):
                        files = gr.File(
                            label="Drop video files here", file_count="multiple", file_types=["video"],
                            height=220,
                        )
                        with_speech = gr.Checkbox(
                            label="Also index speech (Whisper)", value=False,
                            info="Transcribes the audio so spoken moments become searchable. Slower.",
                        )
                        add = gr.Button("Add to library", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div class="av-help"><b>How indexing works</b><br>'
                            "Each video is sampled at 1 frame per second, embedded with SigLIP 2, "
                            "and grouped into 4-second segments. A 5-minute video takes a few "
                            "minutes on CPU. Videos already in the library are skipped.</div>"
                        )
                        add_log = gr.HTML()
                add.click(add_videos, [index_dir, files, with_speech], [add_log, summary])

            with gr.Tab("Live camera"):
                with gr.Row():
                    with gr.Column(scale=2):
                        frame = gr.Image(
                            label="Camera", interactive=False, height=480, show_label=False,
                            placeholder="Press Start to open the camera",
                        )
                    with gr.Column(scale=1, min_width=300):
                        live_status = gr.HTML(_note("Camera is off."))
                        with gr.Row():
                            start = gr.Button("Start", variant="primary")
                            stop = gr.Button("Stop", variant="stop", interactive=False)
                        detect = gr.Textbox(value="person, cell phone, cup", label="Track objects")
                        watch = gr.Textbox(
                            value="a person holding a phone\na person drinking from a cup",
                            lines=3, label="Alert when", info="One description per line",
                        )
                        threshold = gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Alert sensitivity",
                                              info="Lower = more alerts")
                        gr.HTML('<div class="av-section">Alerts</div>')
                        alerts = gr.HTML(_alerts_html([]))
                start.click(
                    live, [index_dir, detect, watch, threshold], [frame, live_status, alerts, start, stop]
                ).then(_index_summary, index_dir, summary)
                stop.click(stop_live)
    return app


def main() -> None:
    build().queue().launch(inbrowser=True)


if __name__ == "__main__":
    main()
