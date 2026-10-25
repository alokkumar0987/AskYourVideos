# AskYourVideo: Architecture

How the system is put together, in three views: what runs **once per video** (index time), what runs **per question** (inference time), and what runs **continuously** (live). Each box names the code that does the work.

Numbers come from the reference run in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) (laptop CPU, no GPU). Speech-path numbers come from three videos (15:19 of footage) and a preliminary 26-query evaluation (see the technical report, section 8).

---

## 1. System overview

```
                         ┌──────────────────────────────────────────┐
   video files ─────────►│  INDEX TIME  (offline, once per video)   │
                         │  frames → SigLIP 2     audio → Whisper   │
                         └───────────────┬──────────────────────────┘
                                         │ writes
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │  LIBRARY  (.askvideo_*/ directory)       │
                         │  visual index + speech index, same       │
                         │  segment ids                             │
                         └───────────────┬──────────────────────────┘
                                         │ reads
                                         ▼
   text question ───────►┌──────────────────────────────────────────┐
                         │  INFERENCE TIME  (per query)             │──► timestamps
                         │  retrieve (vision ⊕ speech, RRF)         │    + evidence frames
                         │  → optional YOLOE verification gate      │
                         └──────────────────────────────────────────┘

   webcam ──────────────►┌──────────────────────────────────────────┐
                         │  LIVE  (continuous)                      │──► boxes, alerts,
                         │  YOLOE + ByteTrack, SigLIP watch alerts  │    recording
                         └───────────────┬──────────────────────────┘
                                         │ on stop: add_embedded()
                                         └──────────► LIBRARY

   Front ends: CLI (cli.py) · Gradio UI (ui.py) · eval scripts (eval/)
```

---

## 2. Index time (offline, once per video)

```
                                   video.mp4
                                       │
              ┌────────────────────────┴─────────────────────────┐
              │ frames                                           │ audio track
              ▼                                                  ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────────┐
 │ sample_frames(fps=1)         │ frames.py    │ transcribe()            speech.py│
 │ decode 1 frame/s,            │              │ faster-whisper large-v3-turbo    │
 │ grab() (not decode) the rest │              │ int8, CPU                        │
 └──────────────┬───────────────┘              │  ├ VAD: skip silence and music   │
                │ RGB frames                   │  ├ word timestamps               │
                ▼                              │  └ drop no_speech_prob > 0.6     │
 ┌──────────────────────────────┐              │       or avg_logprob < -1.0      │
 │ OpenClipEmbedder             │ embedder.py  └────────────────┬─────────────────┘
 │ SigLIP 2 ViT-B/16 image enc. │                               │ words [start, end, text, prob]
 │ → 768-d, L2-normalized       │                               │ + language, RTF
 └──────────────┬───────────────┘                               ▼
                │ frame_emb  (F × 768)         ┌──────────────────────────────────┐
                ▼                              │ segment_texts(context ±2 s)      │
 ┌──────────────────────────────┐              │ words → text per segment id      │
 │ VideoIndex.add_embedded()    │ index.py     │ (uses the SAME segment ids as    │◄──┐
 │ group by ts // 4 s           │              │  the visual index)               │   │
 │ mean-pool → re-normalize     │──────────────┴──────────────┬───────────────────┘   │
 │ = 1 vector per 4 s segment   │   segment ids               │                       │
 └──────────────┬───────────────┘   (shared key)       ┌──────┴────────┐              │
                │ seg_emb  (S × 768)                   ▼               ▼              │
                ▼                         ┌─────────────────┐  ┌─────────────────┐    │
 ┌──────────────────────────────┐         │ TextEmbedder    │  │ tokenize()      │    │
 │ FAISS IndexFlatIP            │         │ multilingual-e5 │  │ lowercase, \w + │    │
 │ (built lazily at query time; │         │ small           │  │ Devanagari      │    │
 │  inner product = cosine)     │         │ "passage: " pfx │  │ (for BM25)      │    │
 └──────────────┬───────────────┘         │ → 384-d, L2-norm│  └────────┬────────┘    │
                │                         └────────┬────────┘           │             │
                ▼                                  ▼                    ▼             │
 ┌───────────────────────────────────────────────────────────────────────────────┐   │
 │ LIBRARY DIRECTORY  .askvideo_<name>/                                          │   │
 │   meta.json          model, fps, segment_sec, frames, segments, stats         │───┘
 │   embeddings.npz     frame_emb, seg_emb                                       │
 │   speech.json        text_model, segment_ids, texts, transcripts (words, RTF) │
 │   speech_emb.npz     transcript embeddings (rows ↔ segment_ids)               │
 └───────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Reference cost (CPU) |
|---|---|
| Frames + SigLIP 2, 5-min video (300 frames → 75 segments) | 128 s (~0.43 s/frame) |
| Whisper large-v3-turbo transcription (int8) | RTF 0.53–2.20 across 3 videos (141–785 s per video); Hindi was slowest |
| e5 embeddings for transcript segments | under 1 s per video (60–70 segments) |

**Why speech reuses the visual segment ids:** the two modalities then describe the same 4-second units, so fusing them is a merge of two rankings over one id space. No time alignment is needed at query time.

---

## 3. Inference time (per query)

```
  query: "a man eating a biscuit"            labels (optional): person, biscuit
     │                                                   │
     ├───────────────────────────┬──────────────────────┐│
     ▼                           ▼                      ▼│
┌──────────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ SigLIP 2 text enc.   │ │ e5 "query: "     │ │ tokenize()      │   STAGE 1
│ "a photo of {q}"     │ │ → 384-d          │ │                 │   RETRIEVE
│ → 768-d              │ └────────┬─────────┘ └────────┬────────┘   ~0.5–1 s warm
└──────────┬───────────┘          │                    │            (vision or fused)
           ▼                      ▼                    ▼
┌──────────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ FAISS search over    │ │ cosine vs        │ │ BM25 vs segment │
│ ALL segments         │ │ transcript emb.  │ │ texts           │
│ → vision ranking     │ │ → dense ranking  │ │ → lexical rank  │
└──────────┬───────────┘ └────────┬─────────┘ └────────┬────────┘
           │                      └────────┬───────────┘
           │                               ▼
           │                  ┌─────────────────────────┐
           │                  │ RRF  (speech level)     │  fusion.py
           │                  │ Σ 1/(60 + rank)         │
           │                  │ → speech ranking        │
           │                  └────────────┬────────────┘
           └──────────────┬────────────────┘
                          ▼
             ┌─────────────────────────┐
             │ RRF  (modality level)   │  mode = vision (default) | speech | fused
             │ vision ⊕ speech, weights│  strategy = adaptive (default):
             │ w_v, w_s per query      │  weight ∝ z-score of each
             │                         │  modality's top score
             └────────────┬────────────┘
                          ▼
             ┌─────────────────────────┐
             │ top-K segments (K = 10) │  index.make_hit(): best frame =
             │ + best frame per segment│  argmax(frame_emb · q) in the segment
             └────────────┬────────────┘
                          │
            no labels ────┼──── labels given
                │         ▼
                │  ┌──────────────────────────────────────────┐
                │  │ verify_hits()                  verify.py │   STAGE 2
                │  │ for each of the K segments:              │   VERIFY
                │  │   re-sample at 2 FPS        (frames.py)  │   ~19 s (warm)
                │  │   YOLOE, text-prompted     (detect.py)   │
                │  │   frame evidence = min over labels of    │   80 of 8,983
                │  │     best confidence  (soft AND)          │   frames = 0.89%
                │  │   verified = any frame evidence > 0      │
                │  │   no tracking (ByteTrack drops small,    │
                │  │     brief objects at 2 FPS)              │
                │  └───────────────────┬──────────────────────┘
                │                      ▼
                │  ┌──────────────────────────────────────────┐
                │  │ GATE: verified first, retrieval order    │
                │  │ kept within each group (not re-ranked    │
                │  │ by detector confidence: 5/6 → 3/6)       │
                │  └───────────────────┬──────────────────────┘
                └──────────┬───────────┘
                           ▼
             ┌──────────────────────────────┐
             │ render_hit()   visualize.py  │  boxes, zoom inset for small
             │ → result frames + caption    │  objects, query / time /
             └──────────────┬───────────────┘  verified / source ranks
                            ▼
              timestamps · frames · per-source ranks (CLI, UI, JSON)
```

### Cost model per query

```
  retrieval     O(N · d)     N = segments in library, d = 768 (flat index)
                             → swap to HNSW / IVF-PQ beyond ~10^6 segments
  speech rank   O(N_s · 384) + BM25 over N_s texts (N_s = segments with speech)
  verification  O(K · S · f) K = 10 segments, S = 4 s, f = 2 FPS → 80 frames
                             independent of library size
```

**Recall ceiling:** verification can only re-order what retrieval returned. A moment outside the top-K cannot be recovered, so K trades recall against detector cost.

---

## 4. Live (continuous, webcam)

```
                     ┌────────────────────────────────────────────────┐
  camera ── read() ─►│ MAIN LOOP (camera rate, ~15 FPS)       live.py │
                     │  ├─ write frame to recording (padded to the    │──► recordings/live_*.mp4
                     │  │   wall clock: file time = session time)     │
                     │  ├─ submit newest frame to both workers ─────┐ │
                     │  ├─ draw latest boxes + alert overlay        │ │──► window / UI stream
                     │  └─ on_frame() hook (UI)                     │ │    (on_frame)
                     └──────────────────────────────────────────────┼─┘
                                                                    │
                ┌───────────────────────────────────────────────────┴──────────┐
                ▼  queue size 1: stale frames are dropped                      ▼
  ┌────────────────────────────────┐                     ┌────────────────────────────────┐
  │ _DetectWorker (thread)         │                     │ _EmbedWorker (thread)          │
  │ YOLOE + ByteTrack, persistent  │                     │ every ~1 s: SigLIP 2 image emb │
  │ track ids                      │                     │ score vs "alert when ..." texts│
  │ ~1–2 detections/s on CPU       │                     │ ≥ threshold (0.10) → Alert     │
  │ → latest boxes                 │                     │ keeps (ts, embedding) records  │
  └────────────────────────────────┘                     └───────────────┬────────────────┘
                                                                         │ on stop
                                                                         ▼
                                                      VideoIndex.add_embedded()  (no re-embedding)
                                                      → the session is searchable at once
```

Live recordings have **no audio** (the OpenCV writer is video-only), so live sessions are vision-only in the library.

---

## 5. Code map

| Module | Role | Used at |
|---|---|---|
| `frames.py` | `sample_frames`, `read_frame_at` (seek identically, so boxes land on the right frame) | index, verify, render |
| `embedder.py` | SigLIP 2 / CLIP image and text encoders, shared space | index, query, live |
| `index.py` | `VideoIndex`: segments, FAISS, `rank_segments`, `make_hit`, persistence | index, query |
| `speech.py` | Whisper transcription, `segment_texts`, e5 `TextEmbedder`, BM25, `SpeechIndex` | index, query |
| `fusion.py` | two-level RRF, `search(mode=vision\|speech\|fused)` | query |
| `detect.py` | YOLOE (text-prompted), `track_stream` (ByteTrack), `draw_boxes` (BGR) | verify, live |
| `verify.py` | `verify_hits`: 2 FPS re-sampling, min-over-labels evidence, gate ranking | query |
| `visualize.py` | `render_hit`: presentation frames (RGB) | query output |
| `live.py` | camera loop + detect/embed workers, recording, alerts | live |
| `cli.py` / `ui.py` | front ends | all |

## 6. Invariants worth keeping

- One library = one image model. Mixing embedders is refused, because their vectors are not comparable.
- Speech rows are keyed by visual segment ids. Never re-segment one modality without the other.
- The detector **gates**, it does not rank. Verification labels come **from the query's words**.
- RRF is two-level, so each modality gets one vote, scaled by the fusion strategy. A single fixed speech weight was shown to fail: leave-one-out picked ≈0.1, which erases speech on spoken queries.
- The work the detector does depends on K, never on library size.
