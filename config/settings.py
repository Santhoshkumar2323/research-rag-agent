from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, got {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0, got {value!r}"
        )

    return value


def _required_string(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()

    if not value:
        raise ValueError(
            f"{name} must not be empty"
        )

    return value


def _validated_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)

    path = (
        Path(raw_value).expanduser()
        if raw_value
        else default
    )

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if not os.access(parent, os.W_OK):
        raise ValueError(
            f"{name} parent directory is not writable: {parent}"
        )

    return path


@dataclass(frozen=True)
class Settings:
    lancedb_path: Path
    lancedb_table: str

    europe_pmc_url: str
    europe_pmc_query: str
    max_papers: int
    request_timeout: int

    embedding_model: str
    embedding_dimension: int


def create_settings() -> Settings:
    return Settings(
        lancedb_path=_validated_path(
            "LANCEDB_PATH",
            PROJECT_ROOT / "data" / "lancedb",
        ),
        lancedb_table=_required_string(
            "LANCEDB_TABLE",
            "research_papers",
        ),
        europe_pmc_url=_required_string(
            "EUROPE_PMC_URL",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        ),
        europe_pmc_query=_required_string(
            "EUROPE_PMC_QUERY",
            "(women OR female) AND (health OR medicine)",
        ),
        max_papers=_positive_int("MAX_PAPERS", 60),
        request_timeout=_positive_int("REQUEST_TIMEOUT", 30),
        embedding_model=_required_string(
            "EMBEDDING_MODEL",
            "BAAI/bge-large-en-v1.5",
        ),
        embedding_dimension=_positive_int(
            "EMBEDDING_DIMENSION",
            1024,
        ),
    )

settings = create_settings()