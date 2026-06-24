"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client.

    Returns deterministic local sources by default. Wire a real provider
    (Tavily, Bing, SerpAPI, internal docs) in ``_search_provider``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        # ponytail: local mock so the lab runs offline; swap to a real provider
        # when TAVILY_API_KEY (or similar) is configured.
        return [
            SourceDocument(
                title=f"Source {i + 1}: {query[:50]}",
                url=f"https://example.org/research/{i + 1}",
                snippet=f"Finding {i + 1} relevant to '{query}'. Evidence point {i + 1}.",
                metadata={"rank": i + 1, "provider": "mock"},
            )
            for i in range(max_results)
        ]
