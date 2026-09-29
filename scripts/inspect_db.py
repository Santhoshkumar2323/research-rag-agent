import math

from config.settings import settings
from research.database import ResearchDatabase


def _display_value(
    row: dict,
    key: str,
    default: str = "N/A",
) -> str:
    value = row.get(key)

    if value is None:
        return default

    value = str(value).strip()

    return value if value else default


def validate_all_vectors(
    database: ResearchDatabase,
) -> None:
    table = database.open_table()

    expected_dimension = database.vector_dimension
    expected_rows = database.count()

    vector_data = (
        table.search()
        .select(["vector"])
        .limit(expected_rows)
        .to_arrow()
    )

    if vector_data.num_rows != expected_rows:
        raise RuntimeError(
            "Vector validation row-count mismatch: "
            f"expected {expected_rows}, "
            f"got {vector_data.num_rows}"
        )

    vectors = vector_data["vector"]

    for row_index in range(vectors.length()):
        vector = vectors[row_index].as_py()

        if vector is None:
            raise RuntimeError(
                f"Row {row_index} contains a null vector"
            )

        if len(vector) != expected_dimension:
            raise RuntimeError(
                f"Row {row_index} has invalid vector dimension: "
                f"expected {expected_dimension}, "
                f"got {len(vector)}"
            )

        for value_index, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise RuntimeError(
                    f"Row {row_index}, vector position "
                    f"{value_index} is not numeric"
                )

            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Row {row_index}, vector position "
                    f"{value_index} contains NaN or infinity"
                )

    print(
        f"Validated {expected_rows} vectors: "
        f"{expected_dimension} dimensions each"
    )


def print_sample_papers(
    database: ResearchDatabase,
) -> None:
    table = database.open_table()

    rows = (
        table.search()
        .select(
            [
                "id",
                "title",
                "source",
                "pub_date",
                "doi",
                "pmid",
                "pmcid",
                "url",
            ]
        )
        .limit(5)
        .to_list()
    )

    print()
    print("=== Sample Papers ===")

    if not rows:
        print("No papers found")
        return

    for index, row in enumerate(rows, start=1):
        print()
        print(
            f"{index}. "
            f"{_display_value(row, 'title')}"
        )
        print(
            f"   ID:      "
            f"{_display_value(row, 'id')}"
        )
        print(
            f"   Source:  "
            f"{_display_value(row, 'source')}"
        )
        print(
            f"   Date:    "
            f"{_display_value(row, 'pub_date')}"
        )
        print(
            f"   DOI:     "
            f"{_display_value(row, 'doi')}"
        )
        print(
            f"   PMID:    "
            f"{_display_value(row, 'pmid')}"
        )
        print(
            f"   PMCID:   "
            f"{_display_value(row, 'pmcid')}"
        )
        print(
            f"   URL:     "
            f"{_display_value(row, 'url')}"
        )


def main() -> None:
    database = ResearchDatabase()

    print()
    print("=== Research Database ===")
    print(f"Path:       {database.db_path}")
    print(f"Table:      {database.table_name}")
    print(f"Dimensions: {database.vector_dimension}")

    if not database.table_exists():
        print()
        print("Status: database table does not exist")
        print("Run: python scripts/ingest.py")
        return

    table = database.open_table()
    row_count = database.count()

    print(f"Rows:       {row_count}")

    print()
    print("=== Schema ===")

    print(table.schema)

    print_sample_papers(database)

    print()
    print("=== Full Vector Validation ===")

    validate_all_vectors(database)

    print()
    print("Database inspection completed successfully")


if __name__ == "__main__":
    main()