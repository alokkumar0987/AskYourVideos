# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

All code lives in `AskYourVideo/` (Python package `askvideo`). Run every command from that directory. The workspace root holds nothing else.

## Environment (Windows, read first)

- The C: drive is nearly full. Keep every cache, temp file and download on D:. `AskYourVideo/env.ps1` sets `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `TEMP`/`TMP` to `AskYourVideo/.cache/` and activates `.venv`. In a fresh PowerShell, dot-source it first: `. .\env.ps1`. In tool calls, set those variables inline instead, because shell state does not persist. Model weights are already cached, so add `HF_HUB_OFFLINE=1` to avoid network calls.
- Use the venv interpreter directly: `.venv\Scripts\python.exe`. There is no system `gh` CLI.
- Quoting gets mangled when you pass multi-line Python to `python -c` through PowerShell. Write a script to the scratchpad and run that instead.

## Commands

```powershell
.venv\Scripts\python.exe -m pytest -q                              # all tests (offline, ~4 s, no model downloads)
.venv\Scripts\python.exe -m pytest -q tests/test_verify.py::test_name  # single test
.venv\Scripts\python.exe -m pip install -e ".[ui,dev]"             # reinstall after changing entry points

askvideo index VIDEO... -i .askvideo_siglip2                       # embed videos into an index dir
askvideo search "query" -i .askvideo_siglip2 -k 3 --save-frames out\
askvideo search "a man eating a biscuit" -i .askvideo_siglip2 --verify person biscuit   # + YOLOE stage
askvideo live --detect person "cell phone" --watch "a person holding a phone"            # webcam
askvideo-ui                                                        # Gradio UI at http://127.0.0.1:7860
start_ui.bat                                                       # same, with env vars set; works from any cwd
```

There is no linter or formatter configured.

## Architecture

The pipeline has two stages. Cheap semantic retrieval runs over everything. An expensive detector then runs only on the retrieved candidates. The whole point of the project is that the detector touches a tiny fraction of frames (0.89% in the reference run), so keep that property when you change anything.

- `embedder.py`: `OpenClipEmbedder` puts images and text into one shared space (`siglip2` by default, `clip` for comparison). All outputs are L2-normalized. Queries are wrapped in `QUERY_TEMPLATE`. Anything that has `name`, `dim`, `embed_images` and `embed_texts` satisfies the `Embedder` protocol, and the tests rely on that with fake color embedders.
- `index.py`: `VideoIndex` samples frames at `fps` (default 1), embeds them, and mean-pools them into fixed `segment_sec` (default 4 s) segments. Search uses FAISS `IndexFlatIP` over the segment embeddings. Each `Hit.best_ts` is the frame inside the segment closest to the query. An index directory holds `meta.json` and `embeddings.npz`, and it records the model name. Searching with a different embedder raises an error by design. `add_embedded` takes precomputed embeddings (live mode uses this).
- `frames.py`: `sample_frames` decodes only the sampled frames and calls `grab()` for the rest. `read_frame_at` seeks the same way, so a timestamp from sampling decodes to the exact same frame and drawn boxes line up. Keep the two consistent.
- `detect.py`: `YoloeDetector` is text-prompted, open-vocabulary YOLOE (default weights `yoloe-26s-seg.pt`). Changing the label set rebuilds the predictor, which also resets tracker state. `track_stream` keeps ByteTrack state across calls, for live use. `draw_boxes` returns **BGR**.
- `verify.py`: `verify_hits` re-samples each candidate segment at 2 FPS and requires **all** labels together in a frame (per-frame evidence is the min over labels). `rank="gate"` (default) puts verified segments first but keeps the retrieval order. `"support"` exists only for comparison and performed worse.
- `live.py`: `run_live` runs the display/record loop at camera rate. A `_DetectWorker` thread (camera sources only) and an `_EmbedWorker` thread (~1 frame/s, watch-query alerts) each process only the newest frame. File sources are detected synchronously so that tests are deterministic. The recording is padded against the session clock, so search timestamps match file time. The `on_frame`/`stop` hooks let the UI drive it.
- `visualize.py`: `render_hit` produces the presentation frame (boxes, zoom inset for small objects, caption strip) and returns **RGB**. The UI and `--save-frames` use it. The live overlay stays on the faster `draw_boxes`.
- `ui.py`: Gradio Blocks app with Search, Add videos and Live camera tabs. Models load lazily through `lru_cache` and stay in memory. Index directories are discovered as `.askvideo*/meta.json` in the cwd.
- `cli.py`: argparse front end with subcommands `index`, `search` and `live`.

## Design decisions backed by experiments (don't undo without new evidence)

- The detector is a **gate, not a ranker**. Re-ranking by detector support dropped top-1 from 5/6 to 3/6 on the reference queries.
- Verification labels must come **from the query's own words**. Extra hand-picked labels hurt results. A planned query parser should follow this rule.
- **No tracking during verification.** At 2 FPS, ByteTrack drops small hand-held objects. Tracking is only for the continuous live stream.
- Live detection defaults to `imgsz=640`. 416 was not faster on this CPU and missed close-up faces.

## Data and outputs

- `.askvideo_siglip2/` indexes `samples/Cooking_cake.mp4`, the reference video for all reported numbers. `.askvideo_live/` holds webcam sessions.
- `recordings/` contains webcam footage of the user. It is git-ignored and must never be published.
- `out/` contains generated frames, screenshots and `architecture.html`/`.png`.
- `eval/run_eval.py` reproduces the reported numbers (6 hand-annotated queries, ~3 min on CPU) and writes `eval/results.json`. `docs/TECHNICAL_REPORT.md` and `docs/INTERVIEW_QA.md` quote those numbers, so update them together. `docs/ARCHITECTURE.md` holds the ASCII diagrams. Update it when the pipeline changes.
- Speech path: `speech.py` (Whisper, e5, BM25, `SpeechIndex` keyed by the visual segment ids) and `fusion.py` (two-level RRF, one vote per modality). `askvideo index ... --speech` builds it, and `askvideo search --mode vision|speech|fused` queries it. The ablation lives in `eval/run_speech_eval.py` with `eval/speech_cases.json`. The YouTube cases are filled in by the user from `eval/dump_transcripts.py` output. The YouTube samples are third-party videos: don't publish their frames or transcripts (see `samples/SOURCES.md`).
- Search defaults to `--mode vision`: fused/speech hurt visual queries on QVHighlights (Hit@1 0.84 → 0.74, Lib@1 0.76 → 0.40) and only help queries about what was said.
- Public benchmark: `eval/qvh_prepare.py --hf` fetches QVHighlights clips from a Hugging Face mirror (YouTube bot-checks yt-dlp on this network), `.askvideo_qvh/` indexes them, and `eval/run_qvh_eval.py` scores them with bootstrap CIs.

## Roadmap

Done: retrieval, YOLOE verification, live mode, UI. Next:
- Face identity (SCRFD + ArcFace). It needs weights `scrfd_2.5g_bnkps.onnx` and `arcface_r100.pth` from the user's face repo.
- A query parser.
- A Whisper speech index with score fusion, plus a vision-only vs. vision+speech ablation.
- A proper benchmark against brute-force per-frame detection.
