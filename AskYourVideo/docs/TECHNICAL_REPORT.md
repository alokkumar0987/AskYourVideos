# AskYourVideo: Technical Report

**Retrieve, then verify: open-vocabulary moment search that runs a detector on under 1% of a video.**

Author: Alok Kumar · Status: working prototype, small-scale evaluation · Hardware: laptop CPU only

Component-level diagrams (index time, inference time, live) are in [ARCHITECTURE.md](ARCHITECTURE.md).

![Architecture](architecture.png)

---

## 1. Problem

Given a long video and a free-text question ("a man eating a biscuit"), return the timestamp where it happens, along with visual evidence.

The usual baseline runs an object detector on every frame and then searches the detections. This approach has two problems:

1. **Cost scales with footage, not with the question.** The 5-minute reference video has 8,983 frames. On this CPU, YOLOE takes 245–360 ms per frame, so a full pass takes about **35–55 min**. This is an estimate from measured per-frame latency; I did not run the full pass end to end.
2. **Closed vocabulary.** A per-frame detector only knows the labels it was run with. If someone asks about something new, every frame has to be processed again.

## 2. Approach: two stages

| Stage | Model | Runs on | Cost per query |
|---|---|---|---|
| **Index** (offline, once per video) | SigLIP 2 image encoder (ViT-B/16, 768-d) | 1 frame/s | none (paid once) |
| **Retrieve** | SigLIP 2 text encoder + FAISS inner product | all segments | about 1 s |
| **Verify** | YOLOE, text-prompted open-vocabulary detector | top-K segments only, 2 FPS | K × 4 s × 2 FPS = 80 frames for K=10 |

### 2.1 Indexing

- Frames are sampled at 1 FPS. Skipped frames are `grab()`-ed rather than decoded.
- Each frame is embedded with SigLIP 2's image encoder and L2-normalized.
- Frame embeddings are mean-pooled into fixed **4-second segments** and normalized again. The segment is the unit of retrieval. The frame embeddings are kept so the best frame inside a segment can be picked later.
- Storage is `meta.json` plus `embeddings.npz`. The index records which model built it, and querying with a different model is refused, because embeddings from different models are not comparable.

Reference video: 299 s → 300 frames → 75 segments. Indexing took 128 s on CPU (about 0.43 s per frame).

### 2.2 Retrieval

SigLIP 2 is trained contrastively, so its image and text encoders map into **one shared embedding space**. The text query is wrapped in a template ("a photo of …"), encoded by the **text** encoder, normalized, and searched against the segment embeddings with FAISS `IndexFlatIP`. Inner product on normalized vectors is cosine similarity.

**No text is extracted from the video.** There is no ASR, OCR or captioning. Pixels go through the image encoder and only the user's query goes through the text encoder, so the system works on silent footage.

### 2.3 Verification

For each of the top-K retrieved segments (K=10):

- Re-sample at **2 FPS** and run YOLOE with the query's object labels as text prompts.
- **Per-frame evidence = min over required labels of the best confidence for that label.** This works as a soft AND: "person + biscuit" scores only as high as the less certain of the two, and scores 0 if either is missing.
- A segment is **verified** if any frame has evidence > 0. The frame with the highest evidence becomes the result frame, with its boxes.

### 2.4 Ranking: gate, not ranker

Verified segments come first. **Within each group, the retrieval order is kept.** The detector answers *"are the required objects present?"*, while the embedding still judges *"how well does the scene match the whole sentence?"*. Section 4 shows why this matters.

## 3. Results

Reproduce with `python eval/run_eval.py` (output in `eval/results.json`). The test set has six queries on one cooking video. A result is correct if its best frame falls inside a ground-truth window that I annotated by hand.

| Query | Retrieval only | Verify, rank by support | Verify, gate | Gate + labels from query |
|---|:-:|:-:|:-:|:-:|
| a man eating a biscuit | ✗ | ✓ | ✓ | ✓ |
| kneading dough on a floured counter | ✓ | ✓ | ✓ | ✓ |
| cutting dough with a round biscuit cutter | ✓ | ✗ | ✗ | ✓ |
| brushing butter on biscuits in a cast iron skillet | ✓ | ✓ | ✓ | ✓ |
| a package of Challenge butter | ✓ | ✗ | ✓ | ✓ |
| golden baked biscuits | ✓ | ✗ | ✓ | ✓ |
| **Top-1 correct** | **5/6** | **3/6** | **5/6** | **6/6** |

| Cost (per query, K=10) | Value |
|---|---|
| Frames seen by detector | **80 of 8,983 (0.89%)** |
| Verification time, CPU | ~19 s with warm models (~25 s on the first query) |
| Retrieval-only time | about 1 s |
| Full per-frame detection (estimate) | ~35–55 min |

**How to read this table honestly.** Retrieval alone and gate verification both score 5/6, but they fail on *different* queries:
- Verification fixes "eating a biscuit". Retrieval's #1 there was a man holding up a butter pack, and the detector found no biscuit in it.
- With the hand-picked labels `dough + glass`, verification breaks "biscuit cutter", because the glass is never detected. Using the query's own words (`biscuit cutter`) gives 6/6.

| Retrieval's #1, rejected by the detector (no biscuit) | Promoted to #1 after verification |
|---|---|
| ![rejected](result_rejected.jpg) | ![verified](result_verified.jpg) |

The takeaway is not "verification is +1 query". It is that verification is a precision filter whose quality depends on choosing the right labels.

## 4. Findings

1. **Gate, don't rank.** Re-ranking by detector support dropped top-1 from 5/6 to 3/6. Detector labels are coarse: "biscuit" covers raw dough and golden baked biscuits alike, so detector confidence carries none of the query's adjectives. The embedding does. The detector should only filter.
2. **Labels must come from the query.** Labels I added by guessing ("glass", "hand") made results worse. YOLOE detects "hand" poorly, and the glass was not visible. This rule is the spec for the planned query parser.
3. **Do not track during verification.** At 2 FPS, small hand-held objects move too far between samples for ByteTrack to associate them, and ByteTrack drops unconfirmed new tracks. Tracking removed exactly the brief detections that verification exists to find. Tracking is used only in live mode, where frames are continuous.
4. **Exact frame alignment matters.** The saved frame and the drawn boxes must be the same decoded frame. `sample_frames` and `read_frame_at` seek identically for that reason. An early bug drew boxes on a neighbouring frame.

## 5. Live mode

Webcam → three concurrent loops:
- **Display and record** at camera rate (~14–16 FPS).
- **YOLOE + ByteTrack** in a worker thread (~1–2 detections/s while sharing the CPU). Boxes trail the video by one detection.
- **SigLIP 2 every ~1 s**, scoring the frame against "alert when …" text prompts.

Both workers always take the *newest* frame and drop stale ones, so a slow model lowers its own rate instead of stalling the video. The recording is padded to wall-clock time, and the embeddings computed during the session go straight into the index, so a session can be searched as soon as it ends. The alert threshold (0.10 cosine) was set by eye from a few minutes of testing.

## 6. Complexity and scaling

With N segments in the corpus, K candidates, segment length S and verification rate f:

- Indexing: O(frames), paid once per video.
- Retrieval: O(N·d) with a flat index. This is fine up to about 10⁶ segments. Beyond that, switch to HNSW or IVF-PQ for sublinear search.
- Verification: **O(K·S·f), independent of corpus size.** This is the main property of the design.

The cost of this design is that **recall is capped by retrieval**. If the right segment is not in the top-K, verification cannot bring it back. K trades recall against detector cost.

## 7. Limitations

- **Tiny evaluation:** one video, six queries, ground truth annotated by the author. These are sanity checks, not benchmark numbers.
- **Vision only.** Moments defined by speech ("when he explains the recipe") cannot be found. My expectation is a meaningful recall loss on speech-heavy content, but I have not measured it.
- **64-token query limit** in the SigLIP 2 text encoder. Typical moment queries are about 5–15 tokens.
- Fixed 4 s segments can split an event across a boundary, and mean pooling blurs short actions.
- Retrieval works on single frames, so motion and temporal order ("before / after", "starts to") are not modelled.
- Verification labels are currently given by hand.
- CPU latency: ~19 s per verified query.

## 8. Speech modality (preliminary)

Prompted by a reviewer's hypothesis that vision-only indexing loses about 20% recall, I added a speech index (`speech.py`) and two-level RRF fusion (`fusion.py`). The design is in [ARCHITECTURE.md](ARCHITECTURE.md).

**Setup.**
- Three videos: the cooking video plus two narrated YouTube recipes, one English and one Hindi, credited in `samples/SOURCES.md`.
- 26 queries: 12 visual, 8 spoken, 6 mixed. Spoken queries paraphrase what was said rather than copying it.
- Each search covers the whole library, then is filtered to the query's video, so the other videos act as distractors.
- Reproduce with `python eval/run_speech_eval.py`.
- **Caveat:** the 20 YouTube queries and their windows were drafted by the model from transcripts and 8-second thumbnails, and still await human verification. Treat every number here as preliminary.

**Transcription (Whisper large-v3-turbo, int8, CPU).** Language detection was correct for all three videos (en 1.00, hi 0.94, en 1.00). RTF was 0.53–2.20, slowest on Hindi. The Hindi transcript is in Devanagari, including the English words the speaker used.

| System | All (R@1 / R@5) | Spoken | Visual | Mixed |
|---|---|---|---|---|
| Vision only | 0.69 / 0.81 | 0.12 / 0.38 | **0.92 / 1.00** | 1.00 / 1.00 |
| Speech only | 0.39 / 0.65 | 0.38 / 0.75 | 0.33 / 0.50 | 0.50 / 0.83 |
| Fused, equal votes | 0.61 / **0.89** | 0.38 / 0.75 | 0.58 / 0.92 | 1.00 / 1.00 |
| Fused, fixed weight (leave-one-out) | 0.65 / 0.81 | 0.12 / 0.38 | 0.83 / 1.00 | 1.00 / 1.00 |
| **Fused, adaptive** (strategy used in fused mode) | **0.69** / 0.85 | 0.38 / 0.75 | 0.75 / 0.83 | 1.00 / 1.00 |

**Findings.**
1. **The reviewer was right for spoken queries.** Vision alone found only 38% of spoken-only moments in the top 5. Adding speech doubled that to 75%. How much overall recall changes depends on the query mix: here +4 points R@5, with 31% spoken queries.
2. **Fusing with equal votes costs precision.** Visual-query R@1 fell from 0.92 to 0.58, because speech promoted segments where nothing relevant was visible.
3. **A single global weight can't serve both query types.** Leave-one-out chose a speech weight of about 0.1 in 24 of 26 folds, which silences speech on exactly the queries that need it.
4. **Adaptive per-query weighting** (each modality weighted by the z-score of its top score, nothing tuned) is the only strategy that matched vision's overall R@1 and also kept the spoken-query gain. It still lost visual R@5 (1.00 → 0.83), so the routing signal is weak.
5. **Cross-script mismatch.** Romanized Hinglish and English queries against a Devanagari transcript failed at R@1 in every system. BM25 shares no tokens across scripts, and e5 alone does not bridge the gap.
6. **Phonetic Hinglish matching did not help. It's kept off by default.** I romanized Devanagari words (नाश्ता → naashtaa) and matched words by a consonant "skeleton" (nashta / नाष्टा → `nst`). Results:

   | Fused-adaptive, R@1 / R@5 | All | Hindi video | Spoken |
   |---|---|---|---|
   | Without phonetic matching (default) | **0.69 / 0.85** | **0.70 / 0.90** | **0.38 / 0.75** |
   | Skeletons for every query word | 0.58 / 0.77 | 0.40 / 0.70 | 0.25 / 0.62 |
   | Skeletons for out-of-vocabulary words only | 0.61 / 0.81 | 0.50 / 0.80 | 0.25 / 0.62 |

   - **Why it hurt:** short skeletons from English query words ("and" → `nd`) collide with unrelated Hindi words. Restricting expansion to words not found verbatim in any transcript reduced the noise but did not remove it, because English queries on the Hindi video are themselves out of vocabulary.
   - **Why it couldn't help the target query:** "subah jaldi mein … nashta" is not a matching problem. "Nashta" is spoken throughout the video, so even a perfect match leaves the query ambiguous.
   - The code stays available, opt-in, via `SpeechIndex.rank(..., phonetic=True)`.
7. **A query-type classifier isn't worth building yet.** I measured the headroom first. An oracle that routes every visual or mixed query to vision and every spoken query to speech would reach R@1 0.77 and R@5 0.92, against 0.69 / 0.85 for adaptive fusion. That is only 2 more correct queries out of 26. A real classifier makes mistakes, and its expected R@1 at 90 / 80 / 70% accuracy would be 0.72 / 0.68 / 0.63, so it must be over about 85% accurate just to beat parameter-free adaptive fusion. With 8 spoken queries, such a classifier can be neither trained nor reliably evaluated. Revisit with 100+ queries (30–40 spoken), comparing a zero-shot LLM router and a keyword rule against adaptive fusion on held-out queries.

## 9. Public benchmark: QVHighlights (pilot)

QVHighlights (Lei et al., NeurIPS 2021) pairs 150-second YouTube clips with human-written queries and human-marked time windows (median 16 s). The annotations come from the authors' GitHub repo (`jayleicn/moment_detr`, CC BY-NC-SA 4.0). The clips are the authors' released files, fetched from a Hugging Face mirror (`ayushsdev/qvhighlights-videos`), because YouTube bot-checked yt-dlp on both Wi-Fi and mobile data.

**Setup.**
- 50 validation queries over 50 clips, chosen by a seeded shuffle (`eval/qvh_prepare.py --n 50 --hf`).
- One library holds all 50 clips. Each query is scored inside its own clip, and *Lib@1* also scores the search across the whole library.
- Frames are embedded with SigLIP 2. Speech uses Whisper `small` (RTF ≈ 0.28 on this CPU, vs 0.5–2.2 for turbo).
- **Zero-shot:** nothing is trained on QVHighlights, so these numbers are not directly comparable with published models trained on its train split, and those are usually reported on the test split.
- The predicted window is the top segment grown over adjacent segments that also rank in the clip's top 20%. This rule was fixed in advance, not tuned.
- Brackets are 95% bootstrap intervals over the 50 queries. Reproduce with `python eval/run_qvh_eval.py`.

| System | Hit@1 | Hit@5 | R1@0.5 | R1@0.7 | Lib@1 |
|---|---|---|---|---|---|
| **Vision** | **0.84** [0.74–0.94] | 0.98 [0.94–1.00] | **0.56** [0.42–0.70] | **0.30** [0.18–0.44] | **0.76** [0.64–0.88] |
| Speech | 0.36 [0.22–0.50] | 0.76 [0.64–0.88] | 0.10 [0.02–0.18] | 0.00 | 0.08 [0.02–0.16] |
| Fused, equal | 0.74 [0.62–0.86] | 0.98 [0.94–1.00] | 0.40 [0.28–0.54] | 0.24 [0.14–0.36] | 0.36 [0.24–0.48] |
| Fused, adaptive | 0.74 [0.62–0.86] | 1.00 | 0.42 [0.28–0.56] | 0.24 [0.14–0.36] | 0.40 [0.26–0.54] |
| Random | 0.28 | 0.72 | 0.02 | 0.00 | 0.01 |

**Findings.**
1. **Zero-shot vision localizes well.** R1@0.5 is 0.56, against 0.02 for random. Searching all 50 clips without being told which video lands on the right clip and moment 76% of the time.
2. **Hit@1 flatters every system**, since random already gets 0.28 because ground-truth windows are long. R1@0.5 and R1@0.7 are the honest metrics.
3. **Speech hurts here.** QVHighlights queries describe what is *seen* ("a woman shows her outfit"), and vlog chatter matches many queries loosely. Across the library, fusion drops Lib@1 from 0.76 to 0.40.
4. Together with section 8: speech helps when the query is about what was said (spoken R@5 0.38 → 0.75), and hurts on visual queries, most of all at library scale. **The default search mode is therefore `vision`.** `fused` and `speech` stay available for queries about speech.
5. **Language detection.** Whisper `small` labelled 17 of the first 23 clips English. Confident non-English detections (Korean, Tagalog, Hindi at ≥ 0.94) looked like genuine non-English vlogs. Two low-confidence "German" detections (0.54, 0.63) produced doubtful text, and one clip had no words at all (music only).

**Limits.** 50 queries give intervals of ±10–14 points, so this is a pilot. Next: 300+ clips vision-only, and speech on a 100-clip subset.

## 10. Next steps

1. **Scale the benchmark:** QVHighlights to 300+ clips (vision-only), plus Charades-STA (`lmms-eval/charades_sta` on Hugging Face) for a second domain. Add mAP, and a per-frame detection baseline for cost.
2. **Query parser:** split a query into a short visual description (for SigLIP) and object labels taken only from the query (for YOLOE).
3. **Speech, continued:**
   - Get the 20 drafted queries verified by a human.
   - Find a stronger routing signal than the score z-score, for example a small query-type classifier trained on held-out queries.
   - Close the cross-script gap by *translating* the query or the transcript (e.g. into English) before matching. Phonetic romanization was tried and hurt (finding 6). Also try more Hinglish queries, since the current set has only 2.
4. **Identity:** SCRFD + ArcFace, for queries like "Alok at the laptop".
5. Cache detections per (segment, label) so repeated queries cost nothing. Use overlapping segments or temporal smoothing at boundaries.

## 11. Reproduce

```powershell
cd AskYourVideo
. .\env.ps1
askvideo index samples\Cooking_cake.mp4 -i .askvideo_siglip2
askvideo search "a man eating a biscuit" -i .askvideo_siglip2 --verify person biscuit -k 3 --save-frames out\
python eval\run_eval.py
askvideo index samples\Cooking_cake.mp4 samples\yt_8YNa-WLQ0gg.mp4 samples\yt_xH2iedsq0qA.mp4 -i .askvideo_speech --speech
python eval\run_speech_eval.py
python eval\qvh_prepare.py --n 50 --hf
askvideo index data\qvhighlights\videos\*.mp4 -i .askvideo_qvh --speech --asr-model small
python eval\run_qvh_eval.py
python -m pytest -q
```
