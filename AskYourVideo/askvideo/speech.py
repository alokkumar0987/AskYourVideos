"""Speech modality: transcribe a video's audio and index the words by segment.

Moments that are only *said* ("when she explains why the pan must be hot") are
invisible to the frame index. Here the audio is transcribed with Whisper,
word timestamps are mapped onto the same segment ids ``VideoIndex`` already
uses, and each segment's text is searchable two ways:

- dense: a multilingual text embedder (e5), so paraphrases and Hindi/English
  mixes still match;
- lexical: BM25, for exact names, numbers and quantities.

Keying by the existing segment ids is what makes fusion with the visual
ranking a simple merge (see ``fusion.py``).
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_ASR = "large-v3-turbo"
DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-small"
_META = "speech.json"
_ARRAYS = "speech_emb.npz"

# Whisper segments below these are dropped: on music or silence Whisper tends
# to invent text ("Thanks for watching!"), and those lines would match queries.
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0

# \w misses Devanagari vowel signs (they are combining marks), which would split
# every Hindi word apart, so the whole Devanagari block counts as a word char.
_TOKEN = re.compile(r"[\wऀ-ॿ]+")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    prob: float


@dataclass
class Transcript:
    words: list[Word]
    language: str
    language_prob: float
    audio_sec: float
    seconds: float  # wall time spent transcribing
    asr_model: str
    dropped_segments: int = 0

    @property
    def rtf(self) -> float:
        """Real-time factor: seconds of compute per second of audio (lower is faster)."""
        return self.seconds / self.audio_sec if self.audio_sec else 0.0


@lru_cache(maxsize=2)
def _whisper(model: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="cpu", compute_type=compute_type)


def transcribe(video: str | Path, model: str = DEFAULT_ASR, compute_type: str = "int8") -> Transcript:
    """Transcribe a video's audio with word timestamps (language auto-detected)."""
    start = time.perf_counter()
    segments, info = _whisper(model, compute_type).transcribe(
        str(video),
        vad_filter=True,  # skip silence and music before decoding
        word_timestamps=True,
        condition_on_previous_text=False,  # stops one hallucination from seeding the next
    )
    words, dropped = [], 0
    for seg in segments:  # a generator: decoding happens while iterating
        if seg.no_speech_prob > MAX_NO_SPEECH_PROB or seg.avg_logprob < MIN_AVG_LOGPROB:
            dropped += 1
            continue
        for w in seg.words or ():
            text = w.word.strip()
            if text:
                words.append(Word(round(w.start, 2), round(w.end, 2), text, round(w.probability, 3)))
    return Transcript(
        words=words,
        language=info.language,
        language_prob=round(info.language_probability, 3),
        audio_sec=round(info.duration, 2),
        seconds=round(time.perf_counter() - start, 2),
        asr_model=model,
        dropped_segments=dropped,
    )


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --- Hinglish matching -------------------------------------------------------
# Whisper writes Hindi speech in Devanagari, while people type Hindi queries in
# Roman script ("subah jaldi mein nashta"), so BM25 sees no shared tokens. Each
# Devanagari word is therefore also indexed under a phonetic "skeleton" of its
# romanization, and Roman query words are looked up by the same skeleton. The
# skeleton keeps only consonants (no vowels, no h), which absorbs the spelling
# drift on both sides: nashta / naashta / नाश्ता / नाष्टा all become "nst".

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "य़": "y",
}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
# Unicode keeps nukta letters (ज़) decomposed under NFC, but text may arrive in
# either form; normalizing keys and input the same way makes both match.
_CONSONANTS = {unicodedata.normalize("NFC", k): v for k, v in _CONSONANTS.items()}
_VIRAMA, _NUKTA = "्", "़"
_NASAL = {"ं": "n", "ँ": "n", "ः": "h"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def romanize(word: str) -> str:
    """Devanagari → loose Roman (Hinglish-style) spelling, e.g. नाश्ता → naashtaa."""
    word = unicodedata.normalize("NFC", word)
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if i + 1 < len(word) and word[i + 1] == _NUKTA and ch + _NUKTA in _CONSONANTS:
            ch, i = ch + _NUKTA, i + 1
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            if nxt not in _MATRAS and nxt != _VIRAMA and nxt != _NUKTA:
                out.append("a")  # inherent vowel
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _NASAL:
            out.append(_NASAL[ch])
        elif ch not in (_VIRAMA, _NUKTA):
            out.append(ch)
        i += 1
    s = "".join(out)
    # Hindi drops the final inherent vowel: कर is "kar", not "kara".
    if len(s) > 2 and s.endswith("a") and not s.endswith("aa") and word[-1] in _CONSONANTS:
        s = s[:-1]
    return s


def skeleton(word: str) -> str:
    """Consonant key shared by spelling variants: nashta, naashtaa, nasta → "nst"."""
    s = romanize(word) if _DEVANAGARI.search(word) else word.lower()
    s = s.replace("ph", "f").replace("w", "v").replace("z", "j").replace("q", "k")
    s = re.sub(r"[aeiouyh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)  # tt → t


def index_tokens(text: str, vocab: set[str] | None = None) -> list[str]:
    """BM25 tokens: the words themselves, plus "~skeleton" keys for Hinglish matching.

    Documents (``vocab`` None) get keys only for Devanagari words, so English
    transcripts gain no extra matches. A query (``vocab`` = every plain word in
    the transcripts) gets a key only for words that are *out of vocabulary*:
    "adding" occurs as-is in an English transcript and is matched exactly, while
    "nashta" occurs nowhere in Roman script and needs the phonetic fallback.
    Expanding every query word was measured to hurt more: short English
    skeletons ("and" → "nd") collide with many Hindi words. Even OOV-only, the
    net effect was negative, so ``SpeechIndex.rank`` uses this only when asked.
    Keys shorter than two consonants are dropped (हैं, के).
    """
    words = tokenize(text)
    if vocab is None:
        keyed = [w for w in words if _DEVANAGARI.search(w)]
    else:
        keyed = [w for w in words if w not in vocab]
    return words + [f"~{k}" for k in map(skeleton, keyed) if len(k) >= 2]


def segment_texts(
    words: Sequence[Word], segments: Sequence[tuple[int, float, float]], context_sec: float = 2.0
) -> dict[int, str]:
    """Text for each ``(segment_id, start, end)``: words starting in the segment
    widened by ``context_sec`` on both sides, so a sentence crossing a boundary
    is not cut in half. Segments with no speech are left out."""
    starts = np.array([w.start for w in words])
    out = {}
    for sid, s, e in segments:
        lo, hi = np.searchsorted(starts, [s - context_sec, e + context_sec])
        if hi > lo:
            out[sid] = " ".join(w.text for w in words[lo:hi])
    return out


def bm25_scores(query: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 of one tokenized query against tokenized documents."""
    n = len(docs)
    if n == 0:
        return np.zeros(0)
    lengths = np.array([len(d) for d in docs], dtype=np.float64)
    avg = lengths.mean() or 1.0
    counts = [Counter(d) for d in docs]
    scores = np.zeros(n)
    for term in set(query):
        df = sum(term in c for c in counts)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = np.array([c[term] for c in counts], dtype=np.float64)
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg))
    return scores


class TextEmbedder:
    """Multilingual e5 sentence embeddings, L2-normalized.

    e5 was trained with "query: " / "passage: " prefixes and loses accuracy
    without them, so they are added here rather than left to callers.
    """

    def __init__(self, name: str = DEFAULT_TEXT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(name, device=device)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32
        )


@dataclass
class SpeechIndex:
    text_model: str = DEFAULT_TEXT_MODEL
    context_sec: float = 2.0
    segment_ids: list[int] = field(default_factory=list)  # row i of ``emb`` belongs to this segment
    texts: list[str] = field(default_factory=list)
    emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    transcripts: dict[str, dict] = field(default_factory=dict)  # video -> stats + words

    def has_video(self, video: str | Path) -> bool:
        return str(Path(video).resolve()) in self.transcripts

    def add_video(self, video_index, video: str | Path, transcript: Transcript, text_embedder) -> int:
        """Attach a transcript to the segments ``video_index`` holds for ``video``.

        Returns how many segments got text. The video must already be in the
        visual index: speech reuses its segment ids.
        """
        if text_embedder.name != self.text_model:
            raise ValueError(f"Speech index uses {self.text_model!r}, embedder is {text_embedder.name!r}")
        video = str(Path(video).resolve())
        if video in self.transcripts:
            return 0
        segs = [
            (sid, s["start"], s["end"]) for sid, s in enumerate(video_index.segments) if s["video"] == video
        ]
        if not segs:
            raise ValueError(f"{video} is not in the visual index; index its frames first")
        texts = segment_texts(transcript.words, segs, self.context_sec)
        if texts:
            ids = sorted(texts)
            emb = text_embedder.embed_passages([texts[i] for i in ids])
            self.emb = emb if self.emb.size == 0 else np.vstack([self.emb, emb])
            self.segment_ids.extend(ids)
            self.texts.extend(texts[i] for i in ids)
        stats = {k: v for k, v in asdict(transcript).items() if k != "words"}
        self.transcripts[video] = {
            **stats,
            "rtf": round(transcript.rtf, 3),
            "segments_with_speech": len(texts),
            "words": [[w.start, w.end, w.text, w.prob] for w in transcript.words],
        }
        return len(texts)

    def words(self, video: str | Path) -> list[Word]:
        return [Word(*w) for w in self.transcripts[str(Path(video).resolve())]["words"]]

    def rank(
        self, query: str, text_embedder, top_k: int | None = None, phonetic: bool = False
    ) -> tuple[list[int], list[float], list[int]]:
        """Dense ranking (ids, cosine scores) and BM25 ranking (ids; only segments with a term match).

        ``phonetic`` adds Hinglish skeleton matching to BM25 (see ``index_tokens``).
        Off by default: on the 26-query evaluation it lowered fused top-1 on the
        Hindi video (0.70 → 0.50), because skeletons of English query words
        collide with unrelated Hindi words.
        """
        if not self.segment_ids:
            return [], [], []
        if text_embedder.name != self.text_model:
            raise ValueError(f"Speech index uses {self.text_model!r}, embedder is {text_embedder.name!r}")
        dense = self.emb @ text_embedder.embed_query(query)
        order = np.argsort(-dense, kind="stable")[:top_k]
        if phonetic:
            docs = [index_tokens(t) for t in self.texts]
            vocab = {w for d in docs for w in d if not w.startswith("~")}
            bm = bm25_scores(index_tokens(query, vocab), docs)
        else:
            bm = bm25_scores(tokenize(query), [tokenize(t) for t in self.texts])
        bm_ids = [self.segment_ids[i] for i in np.argsort(-bm, kind="stable") if bm[i] > 0]
        return [self.segment_ids[i] for i in order], [float(dense[i]) for i in order], bm_ids[:top_k]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / _ARRAYS, emb=self.emb)
        meta = {
            "text_model": self.text_model,
            "context_sec": self.context_sec,
            "segment_ids": self.segment_ids,
            "texts": self.texts,
            "transcripts": self.transcripts,
        }
        (d / _META).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> SpeechIndex:
        d = Path(directory)
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        return cls(
            text_model=meta["text_model"],
            context_sec=meta["context_sec"],
            segment_ids=meta["segment_ids"],
            texts=meta["texts"],
            emb=np.load(d / _ARRAYS)["emb"].astype(np.float32),
            transcripts=meta["transcripts"],
        )

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / _META).is_file()
