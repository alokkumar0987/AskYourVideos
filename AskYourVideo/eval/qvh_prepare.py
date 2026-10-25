"""Download a reproducible QVHighlights (val) subset for the public benchmark.

    python eval/qvh_prepare.py --n 50

QVHighlights (Lei et al., NeurIPS 2021) pairs 150-second YouTube clips with
human-written queries and the time windows they refer to. Annotations are
CC BY-NC-SA 4.0 (non-commercial research); the videos stay on YouTube and are
fetched here only as the 150 s clip each annotation refers to.

Clips are chosen by a seeded shuffle, so the subset is the same on every run.
Videos that are no longer available are skipped and the next one is tried.
Writes data/qvhighlights/subset.jsonl (queries whose clip downloaded) and
data/qvhighlights/videos/<vid>.mp4.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/qvhighlights")
ANNOTATIONS = ROOT / "highlight_val_release.jsonl"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
)
YT_DLP = Path(sys.executable).with_name("yt-dlp.exe" if sys.platform == "win32" else "yt-dlp")


def parse_vid(vid: str) -> tuple[str, float, float]:
    """'NUsG9BgSes0_210.0_360.0' -> ('NUsG9BgSes0', 210.0, 360.0); YouTube ids may contain '_'."""
    yt, start, end = vid.rsplit("_", 2)
    return yt, float(start), float(end)


class Blocked(RuntimeError):
    """YouTube is rate-limiting this connection; continuing would only extend the block."""


_BLOCK_SIGNS = ("HTTP Error 429", "Too Many Requests", "Sign in to confirm", "Please sign in")


def download(vid: str, out: Path, ffmpeg: str | None) -> bool:
    """True if the clip downloaded, False if the video is gone; raises Blocked on rate limits."""
    yt, start, end = parse_vid(vid)
    cmd = [
        str(YT_DLP), "--no-playlist", "--quiet", "--no-warnings", "--sleep-requests", "2",
        "-f", "bv*[height<=360][vcodec^=avc1]+ba[ext=m4a]/b[height<=360]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start:g}-{end:g}",
        # cut exactly at the annotated start; otherwise the clip begins at the
        # previous keyframe and every ground-truth window is shifted
        "--force-keyframes-at-cuts",
        "-o", str(out),
        f"https://www.youtube.com/watch?v={yt}",
    ]
    if ffmpeg:
        cmd[1:1] = ["--ffmpeg-location", ffmpeg]
    deno = YT_DLP.with_name("deno.exe" if sys.platform == "win32" else "deno")
    if deno.is_file():  # YouTube's JS challenges need a runtime; without one it asks to sign in
        cmd[1:1] = ["--js-runtimes", f"deno:{deno}"]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True, errors="replace")
    except subprocess.CalledProcessError as exc:
        out.unlink(missing_ok=True)
        if any(sign in (exc.stderr or "") for sign in _BLOCK_SIGNS):
            raise Blocked((exc.stderr or "").strip().splitlines()[-1]) from exc
        return False
    except subprocess.TimeoutExpired:
        out.unlink(missing_ok=True)
        return False
    return out.is_file() and out.stat().st_size > 0


def cut_local(vid: str, source_dir: Path, out: Path, ffmpeg: str | None) -> bool:
    """Cut the annotated 150 s from a full video the user downloaded as <youtube_id>.<ext>."""
    yt, start, end = parse_vid(vid)
    sources = [p for p in source_dir.glob(f"{yt}.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not sources:
        return False
    exe = str(Path(ffmpeg) / "ffmpeg") if ffmpeg else "ffmpeg"
    # Re-encode instead of stream copy so the clip starts exactly at `start`
    # (a copy would snap to the previous keyframe and shift every window).
    cmd = [
        exe, "-v", "error", "-y", "-ss", f"{start:g}", "-to", f"{end:g}", "-i", str(sources[0]),
        "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        out.unlink(missing_ok=True)
        return False
    return out.is_file() and out.stat().st_size > 0


HF_MIRROR = "ayushsdev/qvhighlights-videos"  # the authors' released clips, sharded by first character


def download_hf(vid: str, out: Path) -> bool:
    """Fetch the already-cut 150 s clip from a Hugging Face mirror of the official release.

    No YouTube involved, so it works where yt-dlp gets bot-checked.
    """
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for shard in dict.fromkeys((vid[0], vid[0].lower())):
        try:
            path = hf_hub_download(HF_MIRROR, f"{shard}/{vid}.mp4", repo_type="dataset")
        except EntryNotFoundError:
            continue
        shutil.copyfile(path, out)
        return out.stat().st_size > 0
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="Number of clips to download")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ffmpeg", default=None, help="Folder with ffmpeg, if it is not on PATH")
    p.add_argument("--pause", type=float, default=25.0, help="Seconds to wait between downloads")
    p.add_argument(
        "--local",
        type=Path,
        help="Folder of full videos named <youtube_id>.mp4 (downloaded by hand); cut clips "
        "from these instead of downloading. See out/qvh_video_list.txt for the list.",
    )
    p.add_argument(
        "--hf", action="store_true",
        help=f"Download the released clips from the Hugging Face mirror {HF_MIRROR} instead of YouTube",
    )
    args = p.parse_args()

    if not ANNOTATIONS.is_file():
        import urllib.request

        ROOT.mkdir(parents=True, exist_ok=True)
        print(f"downloading annotations from {ANNOTATIONS_URL}")
        urllib.request.urlretrieve(ANNOTATIONS_URL, ANNOTATIONS)

    by_vid: dict[str, list[dict]] = defaultdict(list)
    for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        by_vid[q["vid"]].append(q)
    vids = sorted(by_vid)
    random.Random(args.seed).shuffle(vids)

    videos = ROOT / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    kept, tried, blocked = [], 0, None
    for vid in vids:
        if len(kept) == args.n:
            break
        tried += 1
        out = videos / f"{vid}.mp4"
        cached = out.is_file()
        if args.local or args.hf:
            ok = cached or (download_hf(vid, out) if args.hf else cut_local(vid, args.local, out, args.ffmpeg))
            print(f"[{len(kept) + ok:>3}/{args.n}] {'ok  ' if ok else 'missing'} {vid}", flush=True)
            if ok:
                kept.append(vid)
            continue
        try:
            ok = cached or download(vid, out, args.ffmpeg)
        except Blocked as exc:
            blocked = exc
            break
        print(f"[{len(kept) + ok:>3}/{args.n}] {'ok  ' if ok else 'skip'} {vid}", flush=True)
        if ok:
            kept.append(vid)
        if not cached:
            time.sleep(args.pause)

    subset = [q for vid in kept for q in by_vid[vid]]
    with open(ROOT / "subset.jsonl", "w", encoding="utf-8") as f:
        for q in subset:
            f.write(json.dumps(q) + "\n")
    print(f"{len(kept)} clips ({tried - len(kept) - bool(blocked)} unavailable), {len(subset)} queries -> {ROOT / 'subset.jsonl'}")
    if blocked:
        print(f"stopped early, YouTube is rate-limiting: {blocked}\nre-run later; finished clips are kept")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
