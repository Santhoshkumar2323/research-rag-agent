import logging
import shutil
from pathlib import Path

from config.settings import PROJECT_ROOT, settings


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def _validate_reset_target(
    target_path: Path,
) -> Path:
    project_root = PROJECT_ROOT.resolve()
    resolved_target = target_path.expanduser().resolve()

    if resolved_target == project_root:
        raise ValueError(
            "Refusing to delete the project root."
        )

    try:
        resolved_target.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            "Refusing to delete database path outside "
            f"the project directory: {resolved_target}"
        ) from exc

    return resolved_target


def reset_database(
    target_path: Path | None = None,
) -> None:
    requested_path = (
        target_path
        if target_path is not None
        else settings.lancedb_path
    )

    db_path = _validate_reset_target(
        Path(requested_path)
    )

    if not db_path.exists():
        logger.info(
            "Database does not exist: %s",
            db_path,
        )
        return

    if not db_path.is_dir():
        raise ValueError(
            f"Reset target is not a directory: {db_path}"
        )

    logger.info(
        "Removing research database: %s",
        db_path,
    )

    try:
        shutil.rmtree(db_path)
    except Exception:
        logger.exception(
            "Failed to remove database: %s",
            db_path,
        )
        raise

    logger.info(
        "✓ Research database removed"
    )


if __name__ == "__main__":
    configure_logging()
    reset_database()