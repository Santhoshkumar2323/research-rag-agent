from dataclasses import asdict, dataclass
import hashlib
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    url: str
    pub_date: str
    source: str
    doi: str
    pmid: str
    pmcid: str
    embedding_text: str


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _build_id(
    source: str,
    external_id: str,
    doi: str,
    title: str,
) -> str:
    if external_id:
        return f"{source}:{external_id}".lower()

    if doi:
        return f"doi:{doi.lower()}"

    if not title:
        raise ValueError("Paper has no usable identifier or title")

    return "title:" + hashlib.sha256(
        title.lower().encode("utf-8")
    ).hexdigest()


def _build_url(
    pmcid: str,
    pmid: str,
    doi: str,
) -> str:
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"

    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    if doi:
        return f"https://doi.org/{doi}"

    return ""


def process_paper(record: dict[str, Any]) -> Paper:
    title = _clean_text(record.get("title"))
    abstract = _clean_text(record.get("abstractText"))
    source = _clean_text(record.get("source"))
    external_id = _clean_text(record.get("id"))
    doi = _clean_text(record.get("doi"))
    pmid = _clean_text(record.get("pmid"))
    pmcid = _clean_text(record.get("pmcid"))

    if not title:
        raise ValueError("Paper has no title")

    paper_id = _build_id(
        source=source,
        external_id=external_id,
        doi=doi,
        title=title,
    )

    url = _build_url(
        pmcid=pmcid,
        pmid=pmid,
        doi=doi,
    )

    embedding_text = f"Title: {title}\nAbstract: {abstract}"

    return Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        url=url,
        pub_date=_clean_text(
            record.get("firstPublicationDate")
            or record.get("pubYear")
        ),
        source=source,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        embedding_text=embedding_text,
    )


def process_papers(
    records: list[dict[str, Any]],
) -> list[Paper]:
    papers: list[Paper] = []
    seen_ids: set[str] = set()

    for record in records:
        try:
            paper = process_paper(record)
        except ValueError as exc:
            logger.warning(
                "Skipped malformed paper record: %s",
                exc,
            )
            continue

        if paper.id in seen_ids:
            logger.warning(
                "Skipped duplicate paper record: %s",
                paper.id,
            )
            continue

        seen_ids.add(paper.id)
        papers.append(paper)

    return papers


def paper_to_dict(paper: Paper) -> dict[str, Any]:
    return asdict(paper)