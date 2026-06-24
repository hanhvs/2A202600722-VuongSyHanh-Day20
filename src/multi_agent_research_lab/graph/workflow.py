"""Multi-agent orchestration and single-agent baseline.

ponytail: orchestration is a plain supervisor-driven loop, not a LangGraph
StateGraph. Same routing + stop condition with far less to debug. If you need a
visual graph or checkpointing, langgraph is installed and the loop below maps
1:1 onto a StateGraph with a conditional edge from `supervisor`.
"""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        llm = LLMClient(self.settings)
        search = SearchClient(self.settings)
        self.supervisor = SupervisorAgent(self.settings.max_iterations)
        self.workers = {
            "researcher": ResearcherAgent(llm, search),
            "analyst": AnalystAgent(llm),
            "writer": WriterAgent(llm),
        }

    def build(self) -> object:
        """Return the worker registry (the 'graph' the loop walks)."""

        return self.workers

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("multi_agent_workflow"):
            while True:
                self.supervisor.run(state)
                route = state.route_history[-1]
                if route == "done":
                    break
                worker = self.workers.get(route)
                if worker is None:
                    state.errors.append(f"unknown route: {route}")
                    break
                try:
                    worker.run(state)
                except Exception as exc:  # fail soft: record and let supervisor stop
                    state.errors.append(f"{route} failed: {exc}")
        return state


def run_multi_agent(query: str, settings: Settings | None = None) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow(settings).run(state)


def run_single_agent(query: str, settings: Settings | None = None) -> ResearchState:
    """Single-agent baseline: one LLM call does research+analysis+writing."""

    from multi_agent_research_lab.agents.base import record_result

    state = ResearchState(request=ResearchQuery(query=query))
    with trace_span("single_agent_baseline"):
        state.record_route("baseline")
        resp = LLMClient(settings or get_settings()).complete(
            "You are a research assistant. Research, analyze, and write a cited answer in one pass.",
            f"Query: {state.request.query}\nAudience: {state.request.audience}",
        )
        state.final_answer = resp.content
        record_result(state, AgentName.WRITER, resp, {"mode": "single_agent"})
    return state
