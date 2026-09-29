import streamlit as st

from agent.graph import build_agent_graph
from research.database import ResearchDatabase
from research.embeddings import PaperEmbedder
from research.retrieval import ResearchRetriever


st.set_page_config(
    page_title="Embed for Health",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached backend resources
# ---------------------------------------------------------------------

@st.cache_resource
def get_embedder() -> PaperEmbedder:
    return PaperEmbedder()


@st.cache_resource
def get_retriever() -> ResearchRetriever:
    return ResearchRetriever()


@st.cache_resource
def get_agent_graph():
    return build_agent_graph()


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def initialize_session_state() -> None:
    defaults = {
        "search_results": [],
        "search_query": "",
        "search_input": "",
        "selected_paper_ids": [],
        "selection_message": "",
        "discovery_loaded": False,
        "agent_question": "",
        "agent_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def database_is_ready() -> bool:
    try:
        return ResearchDatabase().table_exists()
    except Exception:
        return False


# ---------------------------------------------------------------------
# Research search
# ---------------------------------------------------------------------

def search_papers(query: str) -> list[dict[str, object]]:
    embedder = get_embedder()
    retriever = get_retriever()

    query_vector = embedder.encode_query(query)

    return retriever.search(
        vector=query_vector,
        top_k=10,
    )


def load_discovery_papers() -> None:
    if st.session_state.discovery_loaded:
        return

    topics = [
        "Endometriosis",
        "PCOS",
        "Fertility",
        "Pregnancy",
        "Breast cancer",
        "Menopause",
        "Maternal health",
        "Women's cardiovascular health",
        "Women's metabolic health",
        "Women's mental health",
    ]

    with st.spinner("Loading research discovery..."):
        try:
            embedder = get_embedder()
            retriever = get_retriever()

            vectors = [
                embedder.encode_query(topic)
                for topic in topics
            ]

            results = retriever.discover(
                query_vectors=vectors,
                total=10,
                per_query=5,
            )

        except Exception as exc:
            st.error(f"Could not load research: {exc}")
            return

    st.session_state.search_results = results
    st.session_state.search_query = ""
    st.session_state.search_input = ""
    st.session_state.selected_paper_ids = []
    st.session_state.selection_message = ""
    st.session_state.agent_result = None
    st.session_state.discovery_loaded = True


# ---------------------------------------------------------------------
# Paper selection
# ---------------------------------------------------------------------

def handle_paper_selection(paper_id: str) -> None:
    widget_key = f"select_{paper_id}"
    selected = st.session_state.get(widget_key, False)
    selected_ids = st.session_state.selected_paper_ids

    if selected:
        if paper_id in selected_ids:
            return

        if len(selected_ids) >= 2:
            st.session_state[widget_key] = False
            st.session_state.selection_message = (
                "You can select at most 2 papers."
            )
            return

        selected_ids.append(paper_id)
        st.session_state.agent_result = None

    elif paper_id in selected_ids:
        selected_ids.remove(paper_id)
        st.session_state.agent_result = None


# ---------------------------------------------------------------------
# Paper card
# ---------------------------------------------------------------------

def render_paper(
    paper: dict[str, object],
    index: int,
    show_similarity: bool,
) -> None:
    paper_id = str(paper.get("id", ""))
    title = str(paper.get("title", "Untitled paper"))
    url = paper.get("url")

    if url:
        st.markdown(f"#### [{index}. {title}]({url})")
    else:
        st.markdown(f"#### {index}. {title}")

    similarity = paper.get("cosine_similarity")

    if show_similarity and isinstance(similarity, (int, float)):
        similarity_text = f"{float(similarity) * 100:.1f}%"
    else:
        similarity_text = "N/A"

    st.caption(
        f"Semantic similarity: {similarity_text}"
    )

    metadata = {
        "Date": paper.get("pub_date") or "N/A",
        "Source": paper.get("source") or "N/A",
        "PMID": paper.get("pmid") or "N/A",
        "DOI": paper.get("doi") or "N/A",
    }

    columns = st.columns(4)

    for column, (label, value) in zip(
        columns,
        metadata.items(),
    ):
        with column:
            st.markdown(
                f"**{label}:** {value}"
            )

    abstract = str(
        paper.get("abstract", "") or ""
    )

    if abstract:
        preview_length = 500

        if len(abstract) > preview_length:
            preview = (
                abstract[:preview_length].rstrip()
                + "..."
            )
        else:
            preview = abstract

        st.write(preview)

        if len(abstract) > preview_length:
            with st.expander("Read abstract"):
                st.write(abstract)

    else:
        st.caption("No abstract available.")

    st.checkbox(
        "Select paper",
        key=f"select_{paper_id}",
        on_change=handle_paper_selection,
        args=(paper_id,),
    )

    st.divider()


# ---------------------------------------------------------------------
# Search actions
# ---------------------------------------------------------------------

def perform_search(query: str) -> None:
    query = query.strip()

    if not query:
        st.warning("Enter a research topic or question.")
        return

    with st.spinner("Searching research..."):
        try:
            results = search_papers(query)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

    st.session_state.search_results = results
    st.session_state.search_query = query
    st.session_state.search_input = query
    st.session_state.selected_paper_ids = []
    st.session_state.selection_message = ""
    st.session_state.agent_result = None
    st.session_state.discovery_loaded = True

    for paper in results:
        paper_id = paper.get("id")
        if paper_id:
            st.session_state[f"select_{paper_id}"] = False


def select_topic(topic: str) -> None:
    st.session_state.search_input = topic
    perform_search(topic)


# ---------------------------------------------------------------------
# Selection summary
# ---------------------------------------------------------------------

def render_selection_bar() -> None:
    selected_ids = st.session_state.selected_paper_ids

    if not selected_ids:
        return

    st.markdown("---")

    st.subheader(
        f"{len(selected_ids)} paper"
        f"{'s' if len(selected_ids) != 1 else ''} selected"
    )

    st.caption(
        "Select up to 2 papers to use with the AI Research Agent."
    )

    results_by_id = {
        str(paper.get("id")): paper
        for paper in st.session_state.search_results
    }

    columns = st.columns(len(selected_ids))

    for column, paper_id in zip(columns, selected_ids):
        paper = results_by_id.get(paper_id)

        if not paper:
            continue

        with column:
            st.write(
                paper.get("title", "Untitled paper")
            )

    st.info(
        "The AI Research Agent will use only these selected papers."
    )


# ---------------------------------------------------------------------
# Research tab
# ---------------------------------------------------------------------

def render_research_tab() -> None:
    load_discovery_papers()

    st.title("Embed for Health")

    st.caption(
        "Discover and analyze research in women's health."
    )

    st.markdown("### Research discovery")

    search_query = st.text_input(
        "Search research",
        placeholder=(
            "Search conditions, biomarkers, treatments, "
            "clinical findings..."
        ),
        label_visibility="collapsed",
        key="search_input",
    )

    columns = st.columns([1, 5])

    with columns[0]:
        if st.button(
            "Search",
            type="primary",
            use_container_width=True,
        ):
            perform_search(search_query)

    st.markdown("##### Explore topics")

    topic_columns = st.columns(5)

    for column, topic in zip(
        topic_columns,
        [
            "Endometriosis",
            "PCOS",
            "Fertility",
            "Pregnancy",
            "Breast cancer",
        ],
    ):
        with column:
            st.button(
                topic,
                use_container_width=True,
                on_click=select_topic,
                args=(topic,),
            )

    st.markdown("---")

    results = st.session_state.search_results

    if st.session_state.search_query:
        st.subheader(
            f"Research results for "
            f"“{st.session_state.search_query}”"
        )
        show_similarity = True
    else:
        st.subheader("Research discovery")
        st.caption(
            "A diverse starting set from the "
            "available research database."
        )
        show_similarity = True

    if not results:
        st.info("No papers found for this search.")
        return

    for index, paper in enumerate(results, start=1):
        render_paper(
            paper=paper,
            index=index,
            show_similarity=show_similarity,
        )

    if st.session_state.selection_message:
        st.warning(
            st.session_state.selection_message
        )
        st.session_state.selection_message = ""

    render_selection_bar()


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------

def run_agent(
    question: str,
    selected_ids: list[str],
) -> dict[str, object]:
    retriever = get_retriever()

    selected_papers = retriever.get_selected_papers(
        selected_ids
    )

    state = {
        "question": question,
        "selected_papers": selected_papers,
        "evidence": [],
        "citations": [],
        "analysis": "",
        "answer": "",
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        },
        "error": None,
    }

    return get_agent_graph().invoke(state)


def render_agent_result(
    result: dict[str, object],
    selected_papers: list[dict[str, object]],
) -> None:
    if result.get("error"):
        st.error(str(result["error"]))
        return

    papers_by_id = {
        str(paper.get("id")): paper
        for paper in selected_papers
    }

    answer = result.get("answer")
    if answer:
        st.markdown("### Answer")
        st.write(str(answer))

    evidence = result.get("evidence", [])
    if evidence:
        st.markdown("### Evidence")
        for item in evidence:
            paper_id = str(item.get("paper_id", "Unknown"))
            paper = papers_by_id.get(paper_id, {})
            title = paper.get("title", "Selected paper")

            st.markdown(f"**{title}**")
            st.caption(
                f"Paper ID: `{paper_id}` · Source: abstract"
            )
            st.write(f"**Claim:** {item.get('claim', '')}")
            st.write(
                f"**Evidence from abstract:** {item.get('text', '')}"
            )
            st.divider()

    citations = result.get("citations", [])
    if citations:
        st.markdown("### Citations")
        st.caption(
            "Citation metadata is taken from the selected paper record, "
            "not generated by the LLM."
        )

        for item in citations:
            paper_id = str(item.get("paper_id", "Unknown"))
            paper = papers_by_id.get(paper_id, {})

            st.markdown(
                f"**{paper.get('title', 'Selected paper')}**"
            )
            st.caption(f"Paper ID: `{paper_id}`")

            metadata = []
            for label, key in (
                ("Date", "pub_date"),
                ("Source", "source"),
                ("PMID", "pmid"),
                ("DOI", "doi"),
            ):
                value = paper.get(key)
                if value:
                    metadata.append(f"{label}: {value}")

            if metadata:
                st.write(" · ".join(str(value) for value in metadata))

            url = paper.get("url")
            if url:
                st.markdown(f"[Open paper]({url})")

            evidence_text = item.get("evidence_text")
            if evidence_text:
                st.write(f"**Cited evidence:** {evidence_text}")

            st.divider()

    usage = result.get("token_usage")
    if isinstance(usage, dict):
        st.markdown("### Token usage")
        columns = st.columns(4)
        values = [
            ("Input", usage.get("input_tokens", 0)),
            ("Output", usage.get("output_tokens", 0)),
            ("Reasoning", usage.get("reasoning_tokens", 0)),
            ("Total", usage.get("total_tokens", 0)),
        ]
        for column, (label, value) in zip(columns, values):
            with column:
                st.metric(label, value)


def render_agent_tab() -> None:
    st.title("AI Research Agent")

    selected_ids = st.session_state.selected_paper_ids

    if not selected_ids:
        st.info(
            "Select 1–2 papers from Research Discovery "
            "to use the AI Research Agent."
        )
        st.write(
            "The agent will answer questions using "
            "only the papers you selected."
        )
        return

    st.subheader(
        f"{len(selected_ids)} selected paper"
        f"{'s' if len(selected_ids) != 1 else ''}"
    )

    try:
        selected_papers = (
            get_retriever().get_selected_papers(
                selected_ids
            )
        )
    except Exception as exc:
        st.error(
            f"Could not load selected papers: {exc}"
        )
        return

    for paper in selected_papers:
        st.write(
            f"• {paper.get('title', 'Untitled paper')}"
        )

    st.markdown("---")

    st.caption(
        "The agent analyzes the selected papers using "
        "their stored titles, abstracts, and metadata."
    )

    st.caption(
        "It does not retrieve or analyze full paper text."
    )

    st.markdown("### Research question")

    question = st.text_area(
        "Research question",
        placeholder=(
            "Ask a question about the selected papers..."
        ),
        key="agent_question",
        height=120,
        label_visibility="collapsed",
    )

    if st.button(
        "Analyze selected papers",
        type="primary",
        use_container_width=True,
    ):
        question = question.strip()

        if not question:
            st.warning(
                "Enter a research question."
            )
            return

        with st.spinner(
            "Running research agent..."
        ):
            try:
                st.session_state.agent_result = run_agent(
                    question=question,
                    selected_ids=selected_ids,
                )
            except Exception as exc:
                st.error(
                    f"Agent execution failed: {exc}"
                )
                return

    result = st.session_state.agent_result

    if result:
        st.markdown("---")
        render_agent_result(
            result,
            selected_papers,
        )


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

def main() -> None:
    initialize_session_state()

    if not database_is_ready():
        st.error(
            "Research database is not available."
        )
        st.write(
            "Run the ingestion pipeline from the project root:"
        )
        st.code(
            "python -m scripts.ingest",
            language="powershell",
        )
        st.stop()

    research_tab, agent_tab = st.tabs(
        [
            "Research",
            "AI Research Agent",
        ]
    )

    with research_tab:
        render_research_tab()

    with agent_tab:
        render_agent_tab()


if __name__ == "__main__":
    main()