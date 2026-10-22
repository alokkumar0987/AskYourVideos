"""Speech index and fusion, with fake embedders and an injected transcript (no models)."""

from __future__ import annotations

import numpy as np
from conftest import COLORS

from askvideo import fusion
from askvideo.index import VideoIndex
from askvideo.speech import (
    SpeechIndex,
    Transcript,
    Word,
    bm25_scores,
    index_tokens,
    romanize,
    segment_texts,
    skeleton,
    tokenize,
)


class ColorEmbedder:
    name = "fake"
    dim = 3

    def embed_images(self, images):
        v = np.array([im.reshape(-1, 3).mean(axis=0) for im in images], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts):
        v = np.array([COLORS.get(t, (1, 1, 1)) for t in texts], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class BagOfWordsEmbedder:
    """Hashes tokens into a small vector: texts sharing words are similar."""

    name = "bow"

    def _vec(self, text):
        v = np.zeros(64, np.float32)
        for t in tokenize(text.split(": ", 1)[-1]):
            v[hash(t) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _transcript(words):
    return Transcript(
        words=[Word(s, s + 0.4, t, 0.9) for s, t in words],
        language="en", language_prob=0.99, audio_sec=12.0, seconds=3.0, asr_model="fake",
    )


def test_tokenize_keeps_hindi_words_whole():
    assert tokenize("नमस्ते दुनिया, Add SALT!") == ["नमस्ते", "दुनिया", "add", "salt"]


def test_romanize_hindi():
    assert romanize("नाश्ता") == "naashtaa"
    assert romanize("कर") == "kar"  # final inherent vowel dropped
    assert romanize("ज़रूरत") == romanize("ज़रूरत")  # precomposed and nukta-decomposed forms agree


def test_skeleton_joins_spelling_variants_across_scripts():
    assert skeleton("nashta") == skeleton("naashta") == skeleton("नाश्ता") == skeleton("नाष्टा") == "nst"
    assert skeleton("subah") == skeleton("सुबह")
    assert skeleton("batter") == skeleton("बैटर")


def test_hinglish_query_matches_devanagari_transcript_via_bm25():
    docs = [index_tokens("सुबह की जल्दी में आसान नाश्ता"), index_tokens("तवा पर तेल डालेंगे")]
    vocab = {w for d in docs for w in d if not w.startswith("~")}
    scores = bm25_scores(index_tokens("subah ka aasan nashta", vocab), docs)
    assert scores.argmax() == 0 and scores[1] == 0
    # English transcripts get no skeleton keys, so they can't pick up accidental matches
    assert not [t for t in index_tokens("preheat the oven") if t.startswith("~")]


def test_query_words_found_verbatim_are_not_expanded():
    vocab = {"add", "the", "oil"}
    keys = [t for t in index_tokens("add the nashta", vocab) if t.startswith("~")]
    assert keys == ["~nst"]  # only the out-of-vocabulary word gets a phonetic key


def test_bm25_prefers_the_document_with_the_term():
    docs = [tokenize("heat the pan"), tokenize("add salt and salt again"), tokenize("serve hot")]
    scores = bm25_scores(tokenize("salt"), docs)
    assert scores.argmax() == 1 and scores[0] == scores[2] == 0


def test_segment_texts_maps_words_to_segments_with_context():
    words = [Word(1.0, 1.4, "heat", 1), Word(5.0, 5.4, "salt", 1), Word(7.9, 8.2, "stir", 1)]
    segs = [(0, 0.0, 4.0), (1, 4.0, 8.0), (2, 8.0, 12.0)]
    assert segment_texts(words, segs, context_sec=0) == {0: "heat", 1: "salt stir"}
    # with context, the word just before a boundary also reaches the next segment
    assert segment_texts(words, segs, context_sec=0.5)[2] == "stir"


def test_rrf_rewards_agreement_between_rankings():
    fused = fusion.rrf([[1, 2, 3], [3, 2, 1], [2]])
    assert fused[0][0] == 2


def test_weights_follow_the_strategy():
    flat, peaked = [0.10, 0.11, 0.10, 0.12, 0.11], [0.9, 0.1, 0.1, 0.1, 0.1]
    assert fusion.modality_weights(flat, peaked, "equal") == (1.0, 1.0)
    assert fusion.modality_weights(flat, peaked, "weighted", 0.25) == (1.0, 0.25)
    wv, ws = fusion.modality_weights(flat, peaked, "adaptive")
    assert ws > wv and abs(wv + ws - 2) < 1e-9  # the peaked modality gets more say


def test_zero_speech_weight_reduces_to_vision():
    vision = ([3, 1, 2], [0.3, 0.2, 0.1])
    fused, _, _ = fusion.fuse(vision, ([2, 1], [0.9, 0.1]), [2], "weighted", 0.0)
    assert [s for s, _ in fused][:3] == [3, 1, 2]


def _indexes(rgb_video, tmp_path):
    index = VideoIndex("fake", fps=1.0, segment_sec=4.0)
    index.add_video(rgb_video, ColorEmbedder())
    speech = SpeechIndex(text_model="bow", context_sec=0)
    # 0-4 s red, 4-8 s green, 8-12 s blue; the oven is only *said*, during the red part
    n = speech.add_video(index, rgb_video, _transcript([(1.0, "preheat"), (1.5, "the"), (2.0, "oven"),
                                                       (9.0, "serve"), (9.5, "warm")]), BagOfWordsEmbedder())
    assert n == 2
    speech.save(tmp_path / "ix")
    return index, SpeechIndex.load(tmp_path / "ix")


def test_speech_index_round_trip(rgb_video, tmp_path):
    _, speech = _indexes(rgb_video, tmp_path)
    assert speech.segment_ids == [0, 2] and speech.texts == ["preheat the oven", "serve warm"]
    assert speech.has_video(rgb_video) and speech.words(rgb_video)[2].text == "oven"


def test_modes_find_what_each_modality_can_see(rgb_video, tmp_path):
    index, speech = _indexes(rgb_video, tmp_path)
    bow = BagOfWordsEmbedder()

    hits, _ = fusion.search(index, "green", ColorEmbedder(), speech, bow, mode="vision", top_k=1)
    assert hits[0].start == 4.0  # vision finds the green part

    hits, ranks = fusion.search(index, "oven", ColorEmbedder(), speech, bow, mode="speech", top_k=1)
    assert hits[0].start == 0.0 and ranks[0] == {"speech": 1}  # only speech knows about the oven

    hits, ranks = fusion.search(index, "oven", ColorEmbedder(), speech, bow, mode="fused", top_k=3)
    assert hits[0].start == 0.0 and set(ranks[0]) == {"vision", "speech"}


def test_vision_search_unchanged_by_refactor(rgb_video):
    index = VideoIndex("fake", fps=1.0, segment_sec=4.0)
    index.add_video(rgb_video, ColorEmbedder())
    plain = index.search("blue", ColorEmbedder(), top_k=3)
    via_fusion, _ = fusion.search(index, "blue", ColorEmbedder(), mode="vision", top_k=3)
    assert plain == via_fusion and plain[0].start == 8.0
