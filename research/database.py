from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa

from config.settings import settings


@dataclass(frozen=True)
class StoredPaper:
    id: str
    title: str
    abstract: str
    url: str
    pub_date: str
    source: str
    doi: str
    pmid: str
    pmcid: str
    vector: list[float]


class ResearchDatabase:
    def __init__(
        self,
        db_path: Path = settings.lancedb_path,
        table_name: str = settings.lancedb_table,
        vector_dimension: int = settings.embedding_dimension,
    ) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.vector_dimension = vector_dimension

        if self.vector_dimension <= 0:
            raise ValueError(
                "vector_dimension must be greater than 0"
            )

        self.db_path.mkdir(parents=True, exist_ok=True)

        self.db = lancedb.connect(str(self.db_path))

    def _build_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("title", pa.string(), nullable=False),
                pa.field("abstract", pa.string(), nullable=False),
                pa.field("url", pa.string(), nullable=False),
                pa.field("pub_date", pa.string(), nullable=False),
                pa.field("source", pa.string(), nullable=False),
                pa.field("doi", pa.string(), nullable=False),
                pa.field("pmid", pa.string(), nullable=False),
                pa.field("pmcid", pa.string(), nullable=False),
                pa.field(
                    "vector",
                    pa.list_(
                        pa.float32(),
                        self.vector_dimension,
                    ),
                    nullable=False,
                ),
            ]
        )

    def table_exists(self) -> bool:
        tables = self.db.list_tables()

        if hasattr(tables, "tables"):
            table_names = tables.tables
        else:
            table_names = tables

        return self.table_name in table_names

    def create_table(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            raise ValueError(
                "Cannot create research table from empty rows"
            )

        self._validate_rows(rows)

        if self.table_exists():
            raise ValueError(
                f"Table already exists: {self.table_name}"
            )

        self.db.create_table(
            self.table_name,
            data=rows,
            schema=self._build_schema(),
            mode="create",
        )

    def replace_table(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            raise ValueError(
                "Cannot replace research table with empty rows"
            )

        self._validate_rows(rows)

        self.db.create_table(
            self.table_name,
            data=rows,
            schema=self._build_schema(),
            mode="overwrite",
        )

    def open_table(self):
        if not self.table_exists():
            raise FileNotFoundError(
                f"Research table does not exist: {self.table_name}"
            )

        return self.db.open_table(self.table_name)

    def search(
        self,
        vector: list[float],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        self._validate_vector(
            vector=vector,
            field_name="search vector",
        )

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than 0"
            )

        table = self.open_table()

        results = (
            table.search(vector)
            .metric("cosine")
            .limit(limit)
            .to_list()
        )

        return results

    def get_by_ids(
        self,
        paper_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not paper_ids:
            return []

        self._validate_ids(paper_ids)

        table = self.open_table()

        escaped_ids = [
            paper_id.replace("'", "''")
            for paper_id in paper_ids
        ]

        condition = ", ".join(
            f"'{paper_id}'"
            for paper_id in escaped_ids
        )

        return (
            table.search()
            .where(f"id IN ({condition})")
            .limit(len(paper_ids))
            .to_list()
        )

    def count(self) -> int:
        table = self.open_table()
        return table.count_rows()

    def _validate_ids(
        self,
        paper_ids: list[str],
    ) -> None:
        for index, paper_id in enumerate(paper_ids):
            if not isinstance(paper_id, str):
                raise ValueError(
                    f"paper_ids[{index}] must be a string"
                )

            if not paper_id.strip():
                raise ValueError(
                    f"paper_ids[{index}] must not be empty"
                )

    def _validate_vector(
        self,
        vector: Any,
        field_name: str,
    ) -> None:
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{field_name} must be a list, tuple, or numpy array"
            )

        if len(vector) != self.vector_dimension:
            raise ValueError(
                f"{field_name} must contain "
                f"{self.vector_dimension} values, "
                f"got {len(vector)}"
            )

        try:
            numeric_vector = np.asarray(
                vector,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} contains non-numeric values"
            ) from exc

        if numeric_vector.ndim != 1:
            raise ValueError(
                f"{field_name} must be one-dimensional"
            )

        if not np.isfinite(numeric_vector).all():
            raise ValueError(
                f"{field_name} contains NaN or infinite values"
            )

    def _validate_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        required_fields = {
            "id",
            "title",
            "abstract",
            "url",
            "pub_date",
            "source",
            "doi",
            "pmid",
            "pmcid",
            "vector",
        }

        ids: set[str] = set()

        for index, row in enumerate(rows):
            missing = required_fields - row.keys()

            if missing:
                raise ValueError(
                    f"Row {index} is missing fields: "
                    f"{sorted(missing)}"
                )

            paper_id = row["id"]

            if not isinstance(paper_id, str):
                raise ValueError(
                    f"Row {index} has a non-string paper ID"
                )

            if not paper_id.strip():
                raise ValueError(
                    f"Row {index} has an empty paper ID"
                )

            if paper_id in ids:
                raise ValueError(
                    f"Duplicate paper ID in snapshot: {paper_id}"
                )

            ids.add(paper_id)

            self._validate_vector(
                vector=row["vector"],
                field_name=f"Row {index} vector",
            )