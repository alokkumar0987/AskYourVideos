"""Image/text embedders in one shared space (SigLIP 2 by default, CLIP for comparison)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image

# name -> (open_clip architecture, pretrained tag)
MODELS: dict[str, tuple[str, str]] = {
    "siglip2": ("ViT-B-16-SigLIP2", "webli"),
    "clip": ("ViT-B-32", "laion2b_s34b_b79k"),
}

# Contrastive image-text models match captions better than bare nouns.
QUERY_TEMPLATE = "a photo of {}"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray: ...
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenClipEmbedder:
    """Wraps an open_clip model; all outputs are L2-normalized float32."""

    def __init__(self, name: str = "siglip2", device: str = "cpu", batch_size: int = 16):
        if name not in MODELS:
            raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
        import open_clip
        import torch

        arch, pretrained = MODELS[name]
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self._torch = torch
        self._model = model.eval().to(device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)
        self._device = device
        self.batch_size = batch_size
        self.name = name
        with torch.no_grad():
            self.dim = int(self._model.encode_text(self._tokenizer(["x"]).to(device)).shape[-1])

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = torch.stack(
                    [self._preprocess(Image.fromarray(im)) for im in images[i : i + self.batch_size]]
                ).to(self._device)
                out.append(self._model.encode_image(batch).float().cpu().numpy())
        return _normalize(np.concatenate(out) if out else np.zeros((0, self.dim), np.float32))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tokens = self._tokenizer([QUERY_TEMPLATE.format(t) for t in texts]).to(self._device)
            return _normalize(self._model.encode_text(tokens).float().cpu().numpy())


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
