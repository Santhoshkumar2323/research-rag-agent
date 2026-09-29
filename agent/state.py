from operator import add
from typing import Annotated, TypedDict


class SelectedPaper(TypedDict):
    id: str
    title: str
    abstract: str
    url: str
    pub_date: str
    source: str
    doi: str
    pmid: str
    pmcid: str
    cosine_similarity: float | None


class Evidence(TypedDict):
    paper_id: str
    text: str
    claim: str
    source: str


class Citation(TypedDict):
    paper_id: str
    evidence_text: str


class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int


def merge_token_usage(
    left: TokenUsage | None,
    right: TokenUsage | None,
) -> TokenUsage:
    left = left or {}
    right = right or {}

    return {
        "input_tokens": (
            int(left.get("input_tokens", 0) or 0)
            + int(right.get("input_tokens", 0) or 0)
        ),
        "output_tokens": (
            int(left.get("output_tokens", 0) or 0)
            + int(right.get("output_tokens", 0) or 0)
        ),
        "reasoning_tokens": (
            int(left.get("reasoning_tokens", 0) or 0)
            + int(right.get("reasoning_tokens", 0) or 0)
        ),
        "total_tokens": (
            int(left.get("total_tokens", 0) or 0)
            + int(right.get("total_tokens", 0) or 0)
        ),
    }
class AgentState(TypedDict):
    question: str
    selected_papers: list[SelectedPaper]

    evidence: Annotated[
        list[Evidence],
        add,
    ]

    citations: Annotated[
        list[Citation],
        add,
    ]

    analysis: str
    answer: str

    token_usage: Annotated[
        TokenUsage,
        merge_token_usage,
    ]

    error: str | None