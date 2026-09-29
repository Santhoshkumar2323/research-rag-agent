from functools import lru_cache
import os

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from config.settings import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
def _get_required_string(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise ValueError(
            f"{name} is required. Add it to the project .env file."
        )

    return value


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default))

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number, got {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than 0, got {value!r}"
        )

    return value


def _get_int(name: str, default: int) -> int:
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


def create_rate_limiter() -> InMemoryRateLimiter:
    requests_per_second = _get_float(
        "GROQ_REQUESTS_PER_SECOND",
        0.5,
    )

    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def create_llm() -> ChatGroq:
    api_key = _get_required_string("GROQ_API_KEY")

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    ).strip()

    if not model_name:
        raise ValueError("GROQ_MODEL must not be empty")

    max_retries = _get_int(
        "GROQ_MAX_RETRIES",
        2,
    )

    max_tokens = _get_int(
        "GROQ_MAX_TOKENS",
        1200,
    )

    timeout = _get_float(
        "GROQ_TIMEOUT",
        60,
    )

    rate_limiter = create_rate_limiter()

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.0,
        max_retries=max_retries,
        timeout=timeout,
        max_tokens=max_tokens,
        reasoning_format="hidden",
        rate_limiter=rate_limiter,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return create_llm()