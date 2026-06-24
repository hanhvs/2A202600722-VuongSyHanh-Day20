"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

SYSTEM = "You are a Researcher. Summarize the sources into concise notes with [n] citations."


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher"):
            sources = self.search.search(state.request.query, state.request.max_sources)
            state.sources = sources
            context = "\n".join(
                f"[{i + 1}] {s.title} ({s.url}): {s.snippet}" for i, s in enumerate(sources)
            )
            resp = self.llm.complete(
                SYSTEM, f"Query: {state.request.query}\nSources:\n{context}"
            )
            state.research_notes = resp.content
            record_result(state, AgentName.RESEARCHER, resp, {"num_sources": len(sources)})
        return state
