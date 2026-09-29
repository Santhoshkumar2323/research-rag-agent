from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from research.database import ResearchDatabase


DEFAULT_TOP_K = 10
MAX_TOP_K = 20
MAX_SELECTED_PAPERS = 2

DEFAULT_DISCOVERY_PER_QUERY = 5
MAX_DISCOVERY_PAPERS = 10


class ResearchRetriever:

    def __init__(
        self,
        database: ResearchDatabase | None = None,
    ) -> None:
        self.database = database or ResearchDatabase()

    def search(
        self,
        vector: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        self._validate_vector(vector)

        if top_k <= 0:
            raise ValueError(
                f"top_k must be greater than 0, got {top_k}"
            )

        top_k = min(top_k, MAX_TOP_K)

        rows = self.database.search(
            vector=list(vector),
            limit=top_k,
        )

        return [
            self._format_search_result(row)
            for row in rows
        ]

    def discover(
        self,
        query_vectors: Sequence[Sequence[float]],
        *,
        total: int = MAX_DISCOVERY_PAPERS,
        per_query: int = DEFAULT_DISCOVERY_PER_QUERY,
    ) -> list[dict[str, Any]]:
        if not query_vectors:
            raise ValueError("query_vectors must not be empty")

        if total <= 0:
            raise ValueError(
                f"total must be greater than 0, got {total}"
            )

        if per_query <= 0:
            raise ValueError(
                f"per_query must be greater than 0, got {per_query}"
            )

        total = min(total, MAX_DISCOVERY_PAPERS)

        candidate_groups: list[list[dict[str, Any]]] = []

        for vector in query_vectors:
            self._validate_vector(vector)

            rows = self.database.search(
                vector=list(vector),
                limit=per_query,
            )

            candidates = [
                self._format_search_result(row)
                for row in rows
            ]

            candidate_groups.append(candidates)

        discovered: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        max_group_length = max(
            len(group)
            for group in candidate_groups
        )

        for position in range(max_group_length):
            for group in candidate_groups:
                if position >= len(group):
                    continue

                paper = group[position]
                paper_id = str(paper.get("id", ""))

                if not paper_id:
                    continue

                if paper_id in seen_ids:
                    continue

                seen_ids.add(paper_id)
                discovered.append(paper)

                if len(discovered) >= total:
                    return discovered

        return discovered

    def get_selected_papers(
        self,
        selected_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        ids = self.validate_selected_ids(selected_ids)

        if not ids:
            return []

        rows = self.database.get_by_ids(ids)

        rows_by_id = {
            str(row.get("id")): row
            for row in rows
        }

        missing_ids = [
            paper_id
            for paper_id in ids
            if paper_id not in rows_by_id
        ]

        if missing_ids:
            raise ValueError(
                "Selected paper IDs were not found: "
                + ", ".join(missing_ids)
            )

        return [
            self._format_search_result(rows_by_id[paper_id])
            for paper_id in ids
        ]

    @staticmethod
    def validate_selected_ids(
        selected_ids: Sequence[str],
    ) -> list[str]:
        ids = list(selected_ids)

        if len(ids) > MAX_SELECTED_PAPERS:
            raise ValueError(
                "A maximum of "
                f"{MAX_SELECTED_PAPERS} papers can be selected"
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Selected paper IDs must be unique"
            )

        for index, paper_id in enumerate(ids):
            if not isinstance(paper_id, str):
                raise ValueError(
                    f"Selected paper ID at index {index} must be a string"
                )

            if not paper_id.strip():
                raise ValueError(
                    f"Selected paper ID at index {index} must not be empty"
                )

        return ids

    def _validate_vector(
        self,
        vector: Sequence[float],
    ) -> None:
        if not isinstance(vector, Sequence):
            raise ValueError("vector must be a sequence")

        if len(vector) != self.database.vector_dimension:
            raise ValueError(
                "Vector dimension mismatch: "
                f"received {len(vector)}, "
                f"expected {self.database.vector_dimension}"
            )

        for index, value in enumerate(vector):
            if not isinstance(value, Real):
                raise ValueError(
                    f"Vector value at index {index} must be numeric"
                )

        numeric_vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if numeric_vector.ndim != 1:
            raise ValueError(
                "Vector must be one-dimensional"
            )

        if not np.isfinite(numeric_vector).all():
            raise ValueError(
                "Vector contains NaN or infinite values"
            )

    @staticmethod
    def _format_search_result(
        row: dict[str, Any],
    ) -> dict[str, Any]:
        distance = row.get("_distance")

        cosine_similarity: float | None = None

        if isinstance(distance, Real):
            cosine_similarity = 1.0 - float(distance)

            cosine_similarity = max(
                -1.0,
                min(1.0, cosine_similarity),
            )

        return {
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "url": row.get("url", ""),
            "pub_date": row.get("pub_date", ""),
            "source": row.get("source", ""),
            "doi": row.get("doi", ""),
            "pmid": row.get("pmid", ""),
            "pmcid": row.get("pmcid", ""),
            "cosine_similarity": cosine_similarity,
        }