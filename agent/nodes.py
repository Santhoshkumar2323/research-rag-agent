from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from agent.state import (
    AgentState,
    Citation,
    Evidence,
    TokenUsage,
)
from config.llm import get_llm


# ---------------------------------------------------------------------
# Structured evidence schema
# ---------------------------------------------------------------------

class ExtractedEvidence(BaseModel):
    paper_id: str = Field(
        description="Exact ID of the paper supplying the evidence."
    )
    claim: str = Field(
        description=(
            "A concise claim directly supported by the paper abstract."
        )
    )
    evidence_text: str = Field(
        description=(
            "Evidence directly supported by the paper abstract."
        )
    )


class EvidenceExtractionResponse(BaseModel):
    evidence: list[ExtractedEvidence] = Field(
        description=(
            "Evidence extracted only from the selected paper abstracts."
        )
    )


# ---------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------

def _extract_token_usage(
    response: Any,
) -> TokenUsage:
    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if not isinstance(usage, dict):
        usage = {}

    input_tokens = int(
        usage.get("input_tokens", 0) or 0
    )

    output_tokens = int(
        usage.get("output_tokens", 0) or 0
    )

    reasoning_tokens = int(
        usage.get("reasoning_tokens", 0) or 0
    )

    total_tokens = int(
        usage.get(
            "total_tokens",
            input_tokens + output_tokens,
        )
        or 0
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------
# Paper context
# ---------------------------------------------------------------------

def _build_paper_context(
    selected_papers: list[dict[str, Any]],
) -> str:
    sections: list[str] = []

    for index, paper in enumerate(
        selected_papers,
        start=1,
    ):
        sections.append(
            "\n".join(
                [
                    f"Paper {index}",
                    f"ID: {paper['id']}",
                    f"Title: {paper['title']}",
                    f"Publication date: {paper['pub_date']}",
                    f"Source: {paper['source']}",
                    f"DOI: {paper['doi']}",
                    f"PMID: {paper['pmid']}",
                    f"PMCID: {paper['pmcid']}",
                    f"Abstract: {paper['abstract']}",
                ]
            )
        )

    return "\n\n".join(sections)


def _build_selected_paper_summary(
    selected_papers: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            (
                f"- Paper ID: {paper['id']} | "
                f"Title: {paper['title']}"
            )
            for paper in selected_papers
        ]
    )


# ---------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------

def _validate_evidence_ids(
    extracted: list[ExtractedEvidence],
    selected_papers: list[dict[str, Any]],
) -> list[ExtractedEvidence]:
    valid_ids = {
        str(paper["id"])
        for paper in selected_papers
    }

    validated: list[ExtractedEvidence] = []

    for item in extracted:
        paper_id = item.paper_id.strip()

        if paper_id not in valid_ids:
            continue

        validated.append(
            item.model_copy(
                update={
                    "paper_id": paper_id,
                }
            )
        )

    return validated


# ---------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------

def extract_evidence(
    state: AgentState,
) -> dict[str, Any]:
    question = state["question"].strip()
    selected_papers = state["selected_papers"]

    if not question:
        return {
            "error": "The research question is empty."
        }

    if not selected_papers:
        return {
            "error": "No papers were selected."
        }

    llm = get_llm()

    # IMPORTANT:
    # Use Groq JSON Schema structured output rather than
    # LangChain tool/function calling.
    structured_llm = llm.with_structured_output(
        EvidenceExtractionResponse,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

    paper_context = _build_paper_context(
        selected_papers
    )

    prompt = f"""
You are extracting research evidence from selected scientific papers.

Research question:
{question}

Selected papers:
{paper_context}

Rules:

1. Use ONLY the paper abstracts and metadata supplied above.
2. Do not use outside knowledge.
3. Do not infer information that is absent from the abstracts.
4. Extract only claims that are directly supported by the abstracts.
5. Every paper_id must exactly match one of the supplied paper IDs.
6. evidence_text must represent information actually present in that
   paper's abstract.
7. If an abstract does not contain useful evidence for the question,
   do not invent evidence for it.
8. Return the required structured JSON response.
"""

    try:
        response = structured_llm.invoke(prompt)

    except Exception as exc:
        return {
            "error": (
                f"Evidence extraction failed: {exc}"
            )
        }

    # include_raw=True returns:
    #
    # {
    #     "raw": AIMessage(...),
    #     "parsed": EvidenceExtractionResponse(...),
    #     "parsing_error": ...
    # }
    raw_response = response.get("raw")
    parsed_response = response.get("parsed")

    if parsed_response is None:
        parsing_error = response.get(
            "parsing_error"
        )

        return {
            "error": (
                "Evidence extraction returned no valid "
                f"structured response: {parsing_error}"
            )
        }

    token_usage = _extract_token_usage(
        raw_response
    )

    validated_items = _validate_evidence_ids(
        parsed_response.evidence,
        selected_papers,
    )

    if not validated_items:
        return {
            "error": (
                "The model returned no evidence linked "
                "to the selected papers."
            ),
            "token_usage": token_usage,
        }

    evidence: list[Evidence] = []

    for item in validated_items:
        claim = item.claim.strip()
        evidence_text = item.evidence_text.strip()

        if not claim or not evidence_text:
            continue

        evidence.append(
            {
                "paper_id": item.paper_id,
                "claim": claim,
                "text": evidence_text,
                "source": "abstract",
            }
        )

    if not evidence:
        return {
            "error": "No usable validated evidence was returned.",
            "token_usage": token_usage,
        }

    citations: list[Citation] = [
        {
            "paper_id": item["paper_id"],
            "evidence_text": item["text"],
        }
        for item in evidence
    ]

    return {
        "evidence": evidence,
        "citations": citations,
        "token_usage": token_usage,
        "error": None,
    }


# ---------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------

def _extract_message_text(
    response: Any,
) -> str:
    if isinstance(response, BaseMessage):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ---------------------------------------------------------------------
# Analysis generation
# ---------------------------------------------------------------------

def generate_analysis(
    state: AgentState,
) -> dict[str, Any]:
    question = state["question"].strip()
    evidence = state["evidence"]
    selected_papers = state["selected_papers"]

    if not question:
        return {
            "error": "The research question is empty."
        }

    if not evidence:
        return {
            "error": "No evidence is available for analysis."
        }

    llm = get_llm()

    evidence_context = "\n\n".join(
        [
            (
                f"Paper ID: {item['paper_id']}\n"
                f"Claim: {item['claim']}\n"
                f"Evidence: {item['text']}\n"
                f"Source: {item['source']}"
            )
            for item in evidence
        ]
    )

    selected_paper_summary = _build_selected_paper_summary(
        selected_papers
    )

    prompt = f"""
You are analyzing evidence extracted from selected scientific papers.

Research question:
{question}

All selected papers:
{selected_paper_summary}

Relevant evidence extracted from those papers:
{evidence_context}

Rules:

1. Exactly {len(selected_papers)} paper(s) were selected.
2. Use ONLY the supplied evidence and selected-paper list.
3. Do not introduce outside facts.
4. Never say that a selected paper was not provided or not supplied.
5. If a selected paper has no relevant evidence, say that its supplied
   abstract did not provide relevant evidence for this question.
6. Do not invent evidence for a paper with no relevant evidence.
7. Explain which selected papers contributed evidence.
8. Identify agreement or differences only when supported.
9. Identify limitations visible from the supplied abstracts/evidence.
10. Do not claim to have read the full papers.
11. Do not invent study results.
12. Distinguish evidence from interpretation.
"""

    try:
        response = llm.invoke(prompt)

    except Exception as exc:
        return {
            "error": (
                f"Analysis generation failed: {exc}"
            )
        }

    token_usage = _extract_token_usage(
        response
    )

    analysis = _extract_message_text(
        response
    )

    if not analysis:
        return {
            "error": "The model returned an empty analysis.",
            "token_usage": token_usage,
        }

    return {
        "analysis": analysis,
        "token_usage": token_usage,
        "error": None,
    }


# ---------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------

def generate_answer(
    state: AgentState,
) -> dict[str, Any]:
    question = state["question"].strip()
    analysis = state["analysis"].strip()
    citations = state["citations"]
    selected_papers = state["selected_papers"]

    if not question:
        return {
            "error": "The research question is empty."
        }

    if not analysis:
        return {
            "error": "No analysis is available."
        }

    llm = get_llm()

    citation_context = "\n\n".join(
        [
            (
                f"Paper ID: {item['paper_id']}\n"
                f"Evidence: {item['evidence_text']}"
            )
            for item in citations
        ]
    )

    selected_paper_summary = _build_selected_paper_summary(
        selected_papers
    )

    prompt = f"""
You are answering a research question using an analysis generated from
selected scientific paper evidence.

Research question:
{question}

All selected papers:
{selected_paper_summary}

Analysis:
{analysis}

Supporting evidence:
{citation_context}

Rules:

1. Exactly {len(selected_papers)} paper(s) were selected.
2. Use only the supplied analysis, selected-paper list, and evidence.
3. Do not add outside knowledge.
4. Never say that a selected paper was not provided or not supplied.
5. If evidence came from only some selected papers, state that clearly.
6. If a selected paper has no relevant evidence, say that its supplied
   abstract did not provide relevant evidence for this question.
7. Do not invent evidence for a paper with no relevant evidence.
8. Do not claim information from the full papers.
9. Give a clear, focused answer to the research question.
10. Keep factual claims tied to the supplied evidence.
11. Distinguish what the evidence shows from interpretation.
12. If the evidence is insufficient, explicitly say so.
"""

    try:
        response = llm.invoke(prompt)

    except Exception as exc:
        return {
            "error": (
                f"Answer generation failed: {exc}"
            )
        }

    token_usage = _extract_token_usage(
        response
    )

    answer = _extract_message_text(
        response
    )

    if not answer:
        return {
            "error": "The model returned an empty answer.",
            "token_usage": token_usage,
        }

    return {
        "answer": answer,
        "token_usage": token_usage,
        "error": None,
    }