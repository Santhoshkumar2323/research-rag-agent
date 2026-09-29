import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings
from research.database import ResearchDatabase
from research.embeddings import PaperEmbedder
from research.processing import paper_to_dict, process_papers
from research.source import fetch_papers


logger = logging.getLogger(__name__)


TEMP_DB_PREFIX = "lancedb_build_"
BACKUP_DB_PREFIX = "lancedb_backup_"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    noisy_loggers = [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "urllib3",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(
            logger_name
        ).setLevel(logging.WARNING)


def build_rows() -> list[dict[str, Any]]:
    logger.info("Research ingestion")
    logger.info("────────────────────────────────────────")

    logger.info("→ Fetching research papers")

    raw_records = fetch_papers()

    logger.info(
        "✓ Fetched %d papers",
        len(raw_records),
    )

    papers = process_papers(raw_records)

    if not papers:
        raise RuntimeError(
            "No valid papers remained after processing"
        )

    logger.info(
        "✓ Processed %d papers",
        len(papers),
    )

    logger.info("→ Loading embedding model")

    embedder = PaperEmbedder()

    logger.info("✓ Embedding model ready")

    logger.info(
        "→ Embedding %d papers",
        len(papers),
    )

    vectors = embedder.embed_papers(
        papers,
        show_progress_bar=True,
    )

    if len(vectors) != len(papers):
        raise RuntimeError(
            "Paper/vector count mismatch: "
            f"{len(papers)} papers vs "
            f"{len(vectors)} vectors"
        )

    logger.info(
        "✓ Generated %d embeddings",
        len(vectors),
    )

    rows: list[dict[str, Any]] = []

    for paper, vector in zip(
        papers,
        vectors,
        strict=True,
    ):
        row = paper_to_dict(paper)

        row.pop("embedding_text")

        row["vector"] = vector

        rows.append(row)

    logger.info(
        "✓ Prepared %d database rows",
        len(rows),
    )

    return rows


def create_snapshot(
    rows: list[dict[str, Any]],
    snapshot_path: Path,
) -> None:
    logger.info("→ Creating database snapshot")

    database = ResearchDatabase(
        db_path=snapshot_path,
    )

    database.create_table(rows)

    stored_count = database.count()

    if stored_count != len(rows):
        raise RuntimeError(
            "Snapshot row-count validation failed: "
            f"expected {len(rows)}, "
            f"stored {stored_count}"
        )

    logger.info(
        "✓ Snapshot validated: %d rows",
        stored_count,
    )


def _remove_directory(
    path: Path,
) -> None:
    if not path.exists():
        return

    try:
        shutil.rmtree(path)

    except Exception:
        logger.exception(
            "Failed to remove directory: %s",
            path,
        )
        raise


def _move_directory(
    source: Path,
    destination: Path,
) -> None:
    if not source.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source}"
        )

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}"
        )

    try:
        shutil.move(
            str(source),
            str(destination),
        )

    except OSError:
        logger.warning(
            "Direct move failed; using copy fallback"
        )

        shutil.copytree(
            source,
            destination,
        )

        try:
            shutil.rmtree(source)

        except Exception:
            logger.exception(
                "Copy succeeded but source cleanup failed: %s",
                source,
            )
            raise


def activate_snapshot(
    snapshot_path: Path,
    live_path: Path,
) -> None:
    parent = live_path.parent

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_path = parent / (
        f"{BACKUP_DB_PREFIX}{uuid.uuid4().hex}"
    )

    old_database_moved = False
    new_database_moved = False

    try:
        if live_path.exists():
            logger.info(
                "→ Backing up current database"
            )

            _move_directory(
                source=live_path,
                destination=backup_path,
            )

            old_database_moved = True

        logger.info(
            "→ Activating new database"
        )

        _move_directory(
            source=snapshot_path,
            destination=live_path,
        )

        new_database_moved = True

    except Exception:
        logger.exception(
            "✗ Snapshot activation failed"
        )

        if new_database_moved and live_path.exists():
            try:
                _remove_directory(live_path)
            except Exception:
                logger.exception(
                    "Failed to remove partially "
                    "activated database"
                )

        if old_database_moved and backup_path.exists():
            try:
                _move_directory(
                    source=backup_path,
                    destination=live_path,
                )
            except Exception:
                logger.exception(
                    "CRITICAL: failed to restore "
                    "previous database"
                )

        raise

    else:
        if backup_path.exists():
            logger.info(
                "→ Removing old database backup"
            )

            _remove_directory(backup_path)

        logger.info(
            "✓ New database activated"
        )


def run_ingestion(
    target_path: Path | None = None,
) -> None:
    live_path = Path(
        target_path
        if target_path is not None
        else settings.lancedb_path
    ).expanduser()

    live_path = live_path.resolve()

    project_root = live_path.parent

    snapshot_path = project_root / (
        f"{TEMP_DB_PREFIX}{uuid.uuid4().hex}"
    )

    logger.info(
        "Target: %s",
        live_path,
    )

    try:
        rows = build_rows()

        create_snapshot(
            rows=rows,
            snapshot_path=snapshot_path,
        )

        activate_snapshot(
            snapshot_path=snapshot_path,
            live_path=live_path,
        )

        logger.info("✓ Research ingestion completed")
        logger.info("")

    except Exception:
        logger.exception(
            "✗ Research ingestion failed"
        )

        if snapshot_path.exists():
            try:
                _remove_directory(snapshot_path)
            except Exception:
                logger.exception(
                    "Failed to clean temporary snapshot"
                )

        raise


if __name__ == "__main__":
    configure_logging()
    run_ingestion()