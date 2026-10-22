# AskYourVideo: Interview Preparation

For ML research and multimodal roles. Each answer has a **short version** to say out loud (about 30 s) and **depth** for follow-ups. All numbers come from [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) and `eval/results.json`.

---

## 0. The 60-second pitch

> Running an object detector on every frame of a video is expensive, and it only answers questions about labels you chose in advance. I built a two-stage system instead. SigLIP 2 embeds one frame per second into a shared image-text space, offline. At query time, a text query retrieves the top-10 four-second segments in about a second. Then YOLOE, an open-vocabulary detector, checks only those segments for the objects named in the query. On my test video the detector looked at 80 of 8,983 frames, 0.89%, and a verified answer took about 19 seconds on a laptop CPU, versus an estimated 35 to 55 minutes for per-frame detection. The most interesting finding was that the detector must be a gate, not a ranker: re-ranking by detector confidence dropped top-1 accuracy from 5/6 to 3/6. It also runs live on a webcam with tracking and text-defined alerts. The evaluation is small, one video and six queries, so a proper benchmark is next.

### Numbers to remember

| | |
|---|---|
| Video | 299 s, 8,983 frames, 300 sampled, 75 segments (4 s) |
| Embeddings | SigLIP 2 ViT-B/16, 768-d, L2-normalized |
| Detector cost per query | 80 frames = **0.89%** (K=10 × 4 s × 2 FPS) |
| Latency | retrieval about 1 s · verification about 19 s (CPU, warm) |
| Top-1 (6 queries) | retrieval 5/6 · rank-by-detector 3/6 · gate 5/6 · gate + query labels 6/6 |
| Live | ~15 FPS display · detector ~1–2/s · SigLIP every ~1 s |

---

## 1. Questions I was actually asked (LinkedIn, by a practitioner)

### Q1. "If the query decides which objects YOLOE looks for, don't you have to run YOLOE over every video in the corpus at query time?"
**Short:** No. The only stage that touches the whole corpus is retrieval, and it runs against an embedding index built offline. YOLOE runs only on the top-K retrieved segments, about 80 frames, so detection cost depends on K and not on corpus size.

**Depth:** Verification is O(K·S·f): K candidates, S seconds each, f frames per second. Retrieval is a flat FAISS search, O(N·d), and becomes sublinear with HNSW or IVF-PQ. The trade-off is that **recall is capped by retrieval**: if the right segment isn't in the top-K, the detector can't recover it. So K is the knob between recall and detector cost. The fair point in the question: if you need guaranteed recall for a fixed label set, offline per-frame detection is the right tool. You lose open vocabulary, though. A hybrid would cache detections per (segment, label), so a repeated label costs nothing.

### Q2. "The index stores visual SigLIP features and the query is text. How can FAISS compare them?"
**Short:** SigLIP 2 is a vision-language model. Its image and text encoders are trained contrastively into one shared space. I encode the query with the text encoder, L2-normalize it, and do an inner-product search, which is cosine similarity, over the normalized visual embeddings. It's the same zero-shot mechanism as CLIP.

**Depth:** CLIP uses a softmax contrastive loss (InfoNCE) over the batch. SigLIP replaces it with a **pairwise sigmoid loss**: every image-text pair is an independent binary match/no-match decision. That removes the batch-wide normalization and scales better. SigLIP 2 adds further training objectives and multilingual data on top. One practical consequence: absolute cosine scores are low (around 0.1) and not calibrated probabilities. So my alert threshold (0.10) is empirical, and ranking is more reliable than thresholding.

### Q3. "Then someone has to extract text from the videos first. What if there's no speech?"
**Short:** No text comes from the video at all. Frames go through the image encoder and only the query goes through the text encoder. There's no ASR, OCR or captioning, so it works on silent footage. The biscuit example is matched purely from pixels.

### Q4. "SigLIP 2 queries are limited to 64 tokens."
**Short:** Correct, the tokenizer's context is 64. Moment queries are usually 5–15 tokens, so it hasn't been a constraint. For long queries, the plan is a parser that splits them into a short visual description for SigLIP and object labels for YOLOE.

**Depth:** Other options are splitting a long query into sub-queries and fusing scores (sum, or a temporal AND), or using a text encoder with a longer context. That second option breaks the shared space unless you train a projection.

### Q5. "I suspect recall drops around 20% if you encode only the visual domain. Evaluate both."
**Short:** I built it and measured it. On spoken-only queries, vision alone found the moment in the top 5 only 38% of the time, and adding speech doubled that to 75%. So for that query type the reviewer was right, and by more than 20%. But fusing with equal votes cut top-1 on visual queries from 0.92 to 0.58. A per-query adaptive weighting kept overall top-1 at the vision level (0.69) and kept the spoken gain. *Caveat: 26 queries, labels drafted and awaiting verification.*

**Depth:**
- The overall gain depends on the query mix: +4 points R@5 here, with 31% spoken queries. So "20%" isn't one number.
- A single global weight fails. Leave-one-out picked a speech weight of about 0.1 in 24 of 26 folds, which silences speech exactly where it's needed. That's evidence for query-dependent fusion.
- Implementation: Whisper large-v3-turbo with VAD and word timestamps. Words map onto the same 4 s segment ids as the visual index. Each segment's text gets multilingual-e5 embeddings and BM25. Fusion is two-level RRF (rank-based, because SigLIP and e5 scores aren't calibrated against each other).
- An open problem it exposed: romanized or English queries against a Devanagari Hindi transcript fail. BM25 shares no tokens across scripts. Related evidence from my other project, a video format classifier: speech *statistics* (words/min, coverage) added only +0.006 macro-F1. That was a different task and didn't use transcript content, so it doesn't answer this question.

---

### Q6. "Why not train a classifier that decides whether a query is visual or spoken?"
**Short:** I measured the ceiling first. A perfect router would add 8 points of R@1 (0.69 → 0.77), which is 2 queries out of 26. At 80% accuracy it would score 0.68, *worse* than my parameter-free adaptive fusion. So it only pays off above about 85% accuracy, and with 8 spoken queries I can't train or verify that.

**Depth:** Adaptive fusion already gets part of the benefit with no training: each modality's weight follows how sharply its top score stands out. Many queries are genuinely mixed ("adding tomato puree" is both seen and said), and a hard router has to pick one side for them, while fusion doesn't. Next step, once there are 100+ labelled queries: compare a zero-shot LLM router (e.g. a small local Qwen) and a keyword rule against adaptive fusion on held-out queries.

---

## 2. Method and design

### Why two stages instead of one strong model?
Each stage does what it's cheap and good at. Embeddings are fast and capture whole-scene semantics, including adjectives and context, but they're fuzzy: a butter pack can look "biscuit-like". Detectors are precise about object presence and give boxes as evidence, but they're expensive and blind to the rest of the sentence. Cascading them gets the embedding's recall and semantics, plus the detector's precision, at the cost of K segments.

### Why does "gate" beat "rank by detector"?
Detector labels are coarse: "biscuit" fires on raw dough and golden baked biscuits equally. Once you sort by detector confidence, you throw away what the query said about *which* biscuits. "golden baked biscuits" went wrong under support ranking for exactly this reason. The gate uses the detector as a binary filter and keeps the semantic order. Measured: 5/6 → 3/6 with support ranking.

### Why the minimum over labels?
A query like "person eating a biscuit" needs both objects. Min-confidence is a soft AND: a frame is only as convincing as its weakest required object, and it scores 0 if any object is missing. A product or mean would let one very confident object hide a missing one.

### Why 1 FPS for indexing, 2 FPS for verification, 4 s segments?
- 1 FPS keeps indexing cheap (about 0.43 s per frame on CPU) and is enough for scene-level retrieval.
- Verification runs denser because small objects appear briefly.
- 4 s balances localization against the number of segments.

These are reasonable defaults, not tuned values. Sweeping them is part of the benchmark plan.

### Why mean-pool frames into segments? Isn't that lossy?
It is: a short action can get averaged away. I keep frame embeddings, though, so the best frame inside a retrieved segment is chosen by max similarity. Better options: max-sim / late interaction over frames, attention pooling, or a real video encoder (e.g. InternVideo) that models motion. Mean pooling was the simplest baseline that worked.

### Why YOLOE and not a VLM (Gemini, Qwen-VL) as the verifier?
YOLOE runs locally, is fast (~0.25 s per frame on CPU), is deterministic, and returns boxes I can show as evidence. A VLM could check *relations* ("is he eating it?"), which YOLOE can't: it only checks that a person and a biscuit are both present. A good future design is YOLOE as a cheap gate on the top-10, then a VLM as a final judge on the top-2 or top-3.

### Why no tracking during verification?
At 2 FPS, small hand-held objects move too far between samples for ByteTrack's motion association. ByteTrack also drops unconfirmed new tracks, which deleted the brief detections I was looking for. I found this by comparing tracked vs untracked runs. Tracking stays in live mode, where frames are continuous.

---

## 3. Evaluation and research rigor

### How did you evaluate, and how much should we trust it?
One cooking video, six queries, hand-annotated time windows, top-1 hit if the best frame falls in the window. I'd call it a sanity check: n=6, single domain, author-made ground truth, and no confidence intervals. One wrong query moves accuracy by 17 points. I say this up front.

### Retrieval and gate both score 5/6. So did verification help at all?
Not on aggregate with my first labels. They fail on different queries. Verification fixed "eating a biscuit" (retrieval's #1 was a butter pack), but my hand-chosen labels `dough + glass` broke "biscuit cutter". With labels taken from the query's own words, it's 6/6. The honest conclusion: verification is a precision filter, and its value depends on label quality. That's why the query parser is the next component.

### How would you build a proper benchmark?
- **Data:** public moment-retrieval sets with temporal annotations, such as Charades-STA, QVHighlights or ActivityNet Captions, plus my own long-video set.
- **Metrics:** R@1/R@5 at temporal IoU 0.5 and 0.7, mAP. For efficiency: detector frames per query, wall time, and the R@K curve to choose K.
- **Baselines:** retrieval only; per-frame detection with the same labels; gate vs support ranking.
- **Ablations:** K, segment length, sampling FPS, label source (hand vs parsed), vision vs vision+speech.
- **Protocol:** fix hyperparameters on a validation split, report confidence intervals or bootstrap, and include a separate test of how often the true segment falls outside the top-K (the retrieval recall ceiling).

### What's the biggest weakness of the approach?
The retrieval recall ceiling. Verification can only re-order what retrieval found. Second is that frame-level embeddings ignore motion and temporal order, so "starts to pour" vs "stops pouring" look the same.

---

## 4. Systems and engineering

### How does live mode stay smooth on a CPU?
Three loops. Display and recording run at camera rate. The detector and SigLIP each run in a worker thread that only takes the **newest** frame and drops stale ones, so slow models lower their own rate instead of blocking video. Boxes trail by about one detection. Recording is padded to wall-clock time, so a timestamp from a later search matches the file.

### A bug you found and fixed?
The saved result frame and the boxes drawn on it came from neighbouring frames, because sampling and single-frame reads seeked differently. I made both seek by frame number, so a timestamp from sampling decodes to exactly that frame. A second one: a smaller detector input (416 px) was supposed to be faster, but it wasn't faster on this CPU and it missed close-up faces. Measuring it reversed my assumption, so the default stays at 640.

### How does it scale to 10,000 hours of video?
- Indexing is embarrassingly parallel: batch it on GPU and shard by video.
- Switch from a flat index to HNSW or IVF-PQ, possibly with product quantization to keep memory in check.
- Verification cost per query stays O(K).
- Cache detections per (segment, label).
- Keep frame embeddings on disk (memory-mapped) and only segment embeddings in RAM.

---

## 5. Curveball questions

**"Isn't this just CLIP retrieval plus a filter?"**
Architecturally, yes. Retrieve-then-verify is a well-known pattern, and I don't claim novelty. The contribution is engineering and measurement: an open-vocabulary detector as the verifier, the evidence that it must gate rather than rank, the rule that labels come from the query, and the cost accounting (0.89% of frames).

**"Why should a frame-level model understand an action like 'eating'?"**
It doesn't really. It matches the *appearance* of eating (a hand near the mouth, food in hand). Actions that need motion would need a video encoder or a VLM judge. That limitation is listed in the report.

**"Your threshold is 0.10. Where did that come from?"**
Picked by eye from a few minutes of webcam testing. SigLIP cosines aren't calibrated, so a proper version would calibrate per query against background scores (z-score) or learn the threshold on labelled clips.

**"What would you do with a GPU and a month?"**
1. Run the public benchmark with the ablations above.
2. Add the query parser and the speech index, and measure the fusion gain.
3. Try a video encoder for the retrieval stage.
4. Add a VLM judge on the top-3.

---

## 6. Words to use precisely

- **Shared embedding space / zero-shot retrieval:** text and image vectors are directly comparable.
- **Cascade / retrieve-then-verify / two-stage:** cheap high-recall stage, then an expensive high-precision stage.
- **Recall ceiling:** the second stage can't exceed the first stage's recall at K.
- **Open-vocabulary detection:** labels given as text at inference time, not fixed at training.
- **Late fusion:** combining scores from separate modality indexes, not their features.
- **Temporal IoU:** overlap between the predicted and ground-truth time windows.
