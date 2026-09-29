import logging
import time
from typing import Any

import requests

from config.settings import settings


logger = logging.getLogger(__name__)


class ResearchSourceError(RuntimeError):
    """Raised when the research source cannot be read safely."""


def fetch_papers(
    url: str = settings.europe_pmc_url,
    query: str = settings.europe_pmc_query,
    max_papers: int = settings.max_papers,
    timeout: int = settings.request_timeout,
    max_retries: int = 3,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = "*"

    for page_number in range(1, max_pages + 1):
        if len(records) >= max_papers:
            break

        remaining = max_papers - len(records)

        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(1000, remaining),
            "cursorMark": cursor,
        }

        payload = _request_with_retry(
            url=url,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
        )

        result_list = payload.get("resultList")

        if not isinstance(result_list, dict):
            raise ResearchSourceError(
                "Europe PMC response is missing 'resultList'"
            )

        batch = result_list.get("result", [])

        if not isinstance(batch, list):
            raise ResearchSourceError(
                "Europe PMC response contains an invalid 'result' field"
            )

        if not batch:
            logger.info(
                "Europe PMC returned no more records after %d pages",
                page_number,
            )
            break

        records.extend(batch)

        next_cursor = payload.get("nextCursorMark")

        if not next_cursor:
            break

        if next_cursor == cursor:
            raise ResearchSourceError(
                "Europe PMC pagination cursor stopped changing"
            )

        cursor = next_cursor

    else:
        raise ResearchSourceError(
            f"Europe PMC pagination exceeded max_pages={max_pages}"
        )

    logger.info(
        "Fetched %d research records from Europe PMC",
        len(records),
    )

    return records[:max_papers]


def _request_with_retry(
    url: str,
    params: dict[str, Any],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
            )

            if response.status_code == 429 or (500 <= response.status_code < 600):
                raise requests.HTTPError(
                    f"Server error: HTTP {response.status_code}",
                    response=response,
                )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise ResearchSourceError(
                    "Europe PMC returned a non-object JSON response"
                )

            return payload

        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                raise ResearchSourceError(
                    f"Europe PMC request failed after "
                    f"{max_retries + 1} attempts"
                ) from exc

            delay = 2 ** attempt

            logger.warning(
                "Europe PMC request failed "
                "(attempt %d/%d): %s. Retrying in %ds",
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )

            time.sleep(delay)

    raise ResearchSourceError("Unexpected retry state")