"""Behavior tests for the implemented agents/workflow (offline mode, no API key)."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

OFFLINE = Settings(OPENAI_API_KEY=None)


def test_supervisor_routes_researcher_first() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    assert SupervisorAgent().decide(state) == "researcher"


def test_supervisor_stops_at_max_iterations() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 99
    assert SupervisorAgent(max_iterations=6).decide(state) == "done"


def test_multi_agent_workflow_produces_cited_answer() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = MultiAgentWorkflow(OFFLINE).run(state)
    assert result.final_answer
    assert result.sources
    assert result.route_history[-1] == "done"
    # researcher, analyst, writer each ran once
    assert [r.agent.value for r in result.agent_results] == ["researcher", "analyst", "writer"]
    assert "[1]" in result.final_answer  # at least one citation carried through
    assert not result.errors
