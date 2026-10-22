"""Write each video's transcript, grouped into 4-second segments, for annotation.

    python eval/dump_transcripts.py [-i .askvideo_speech]

Output goes to out/transcripts/<video>.txt (git-ignored: transcripts of other
people's videos are not ours to publish). Read it, then write queries in
eval/speech_cases.json; paraphrase, don't copy, so BM25 is not handed the answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from askvideo.index import VideoIndex
from askvideo.speech import SpeechIndex


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--index-dir", default=".askvideo_speech")
    p.add_argument("-o", "--out", default="out/transcripts")
    args = p.parse_args()

    index, speech = VideoIndex.load(args.index_dir), SpeechIndex.load(args.index_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for video, info in speech.transcripts.items():
        words = speech.words(video)
        lines = [
            f"# {Path(video).name}",
            f"# language {info['language']} ({info['language_prob']}), {len(words)} words, "
            f"RTF {info['rtf']}, {info['dropped_segments']} low-confidence lines dropped",
            "",
        ]
        for seg in (s for s in index.segments if s["video"] == video):
            said = " ".join(w.text for w in words if seg["start"] <= w.start < seg["end"])
            lines.append(f"[{_fmt(seg['start'])}-{_fmt(seg['end'])}] {said}")
        path = out / f"{Path(video).stem}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"{path}: {len(words)} words, language {info['language']}")


if __name__ == "__main__":
    main()
