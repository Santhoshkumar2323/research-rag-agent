from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from research.processing import Paper


class PaperEmbedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        expected_dimension: int = settings.embedding_dimension,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        device: str | None = None,
        min_paper_text_length: int = 20,
    ) -> None:
        if expected_dimension <= 0:
            raise ValueError(
                f"expected_dimension must be greater than 0, got {expected_dimension}"
            )

        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be greater than 0, got {batch_size}"
            )

        if min_paper_text_length <= 0:
            raise ValueError(
                "min_paper_text_length must be greater than 0, "
                f"got {min_paper_text_length}"
            )

        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.min_paper_text_length = min_paper_text_length

        resolved_device = self._resolve_device(device)

        self.model = SentenceTransformer(
            model_name,
            device=resolved_device,
        )

        actual_dimension = self.model.get_embedding_dimension()

        if actual_dimension != expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model returned {actual_dimension}, "
                f"expected {expected_dimension}"
            )

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device is None:
            return "cpu"

        device = device.strip()

        if not device:
            raise ValueError("device must not be empty")

        return device

    def _validate_paper_texts(
        self,
        texts: Sequence[str],
    ) -> None:
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Paper text at index {index} must be a string"
                )

            if len(text.strip()) < self.min_paper_text_length:
                raise ValueError(
                    "Paper text at index "
                    f"{index} is too short: "
                    f"{len(text.strip())} characters"
                )

    @staticmethod
    def _validate_queries(
        queries: Sequence[str],
    ) -> None:
        for index, query in enumerate(queries):
            if not isinstance(query, str):
                raise ValueError(
                    f"Query at index {index} must be a string"
                )

            if not query.strip():
                raise ValueError(
                    f"Query at index {index} must not be empty"
                )

    def _encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError(
                "Embedding model returned an invalid shape: "
                f"{vectors.shape}"
            )

        if vectors.shape[0] != len(texts):
            raise ValueError(
                "Embedding count mismatch: "
                f"received {vectors.shape[0]} vectors "
                f"for {len(texts)} inputs"
            )

        if vectors.shape[1] != self.expected_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"received {vectors.shape[1]}, "
                f"expected {self.expected_dimension}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embedding output contains NaN or infinite values"
            )

        return vectors.tolist()

    def encode_papers(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        self._validate_paper_texts(texts)

        return self._encode(
            texts,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(
        self,
        query: str,
    ) -> list[float]:
        self._validate_queries([query])

        vectors = self._encode([query])

        return vectors[0]

    def embed_papers(
        self,
        papers: Sequence[Paper],
        *,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        texts = [paper.embedding_text for paper in papers]

        return self.encode_papers(
            texts,
            show_progress_bar=show_progress_bar,
        )