from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from .config import Settings


LATIN_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9_'’-]*|\d+(?:\.\d+)?)(?![A-Za-z0-9_])")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")


def _feature_tokens(text: str) -> list[str]:
    folded = text.casefold()
    tokens = LATIN_TOKEN_RE.findall(folded)
    for run in CJK_RUN_RE.findall(folded):
        tokens.extend([run] if len(run) == 1 else (run[index : index + 2] for index in range(len(run) - 1)))
    return tokens


class Embedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...


class DeterministicEmbedder(Embedder):
    """Feature hashing for offline demos/tests. It is not a production embedding model."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.model_name = f"demo-feature-hash-{dimension}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            # Keep Latin terms separate from adjacent CJK text and represent
            # contiguous Chinese prose with character bigrams.
            tokens = _feature_tokens(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


class OpenAICompatibleEmbedder(Embedder):
    def __init__(self, base_url: str, api_key: str, model: str, dimension: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        values = [item["embedding"] for item in sorted(response.json()["data"], key=lambda x: x["index"])]
        array = np.asarray(values, dtype=np.float32)
        self.dimension = int(array.shape[1])
        return _normalize(array)


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, model: str, dimension: int):
        self.api_key = api_key
        self.model_name = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.embed_content(model=self.model_name, contents=texts)
        array = np.asarray([item.values for item in response.embeddings], dtype=np.float32)
        self.dimension = int(array.shape[1])
        return _normalize(array)


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "deterministic":
        return DeterministicEmbedder(settings.embedding_dimension)
    if settings.embedding_provider == "gemini":
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings")
        return GeminiEmbedder(settings.google_api_key, settings.embedding_model, settings.embedding_dimension)
    if not settings.openai_embedding_base_url or not settings.openai_embedding_api_key:
        raise RuntimeError("OpenAI-compatible embedding URL and API key are required")
    return OpenAICompatibleEmbedder(
        settings.openai_embedding_base_url,
        settings.openai_embedding_api_key,
        settings.embedding_model,
        settings.embedding_dimension,
    )
