# Ask Your Video

Find a moment in a video by describing it. The object detector runs only where it matters.

![Architecture](docs/architecture.png)

1. **Index (offline):** sample 1 frame/s, embed it with SigLIP 2, and pool into 4-second segments in a FAISS index.
2. **Retrieve:** the text query goes through SigLIP 2's text encoder and returns the top-10 segments in about 1 s.
3. **Verify:** open-vocabulary YOLOE checks only those segments for the objects named in the query. Verified segments go first, and the retrieval order is kept.
4. **Live:** on a webcam, YOLOE + ByteTrack track objects, SigLIP 2 raises text-defined alerts, and the session is indexed when it ends.

On the reference video, the detector processed **80 of 8,983 frames (0.89%)** per query (CPU, about 19 s). Details, results and limitations are in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Usage

```powershell
pip install -e ".[ui,dev]"
askvideo index samples\Cooking_cake.mp4 -i .askvideo_siglip2
askvideo search "a man eating a biscuit" -i .askvideo_siglip2 --verify person biscuit -k 3 --save-frames out\
askvideo live --detect person "cell phone" --watch "a person holding a phone"
askvideo-ui                      # browser UI at http://127.0.0.1:7860
python eval\run_eval.py          # reproduce the reported numbers
python -m pytest -q
```

For `--verify`, use object words that appear in the query. Extra guessed labels hurt accuracy.

## Docs

- [Architecture](docs/ARCHITECTURE.md): ASCII diagrams for index time, inference time and live mode, plus a code map.
- [Technical report](docs/TECHNICAL_REPORT.md): method, results, findings, limitations, next steps.
- [Interview preparation](docs/INTERVIEW_QA.md): expected questions with short and in-depth answers.

## Roadmap

Benchmark on public moment-retrieval data · query parser · Whisper speech index with score fusion · face identity (SCRFD + ArcFace).
