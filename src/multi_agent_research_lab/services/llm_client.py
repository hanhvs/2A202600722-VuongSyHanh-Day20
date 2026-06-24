"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.config import Settings, get_settings

# Rough gpt-4o-mini pricing (USD per token). Tune if you switch models.
_INPUT_RATE = 0.15 / 1_000_000
_OUTPUT_RATE = 0.60 / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(input_tokens * _INPUT_RATE + output_tokens * _OUTPUT_RATE, 6)


class LLMClient:
    """Provider-agnostic LLM client.

    Uses OpenAI when ``OPENAI_API_KEY`` is set; otherwise falls back to a
    deterministic offline stub so the lab runs with no key or network.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if self.settings.openai_api_key:
            return self._complete_openai(system_prompt, user_prompt)
        return self._complete_offline(system_prompt, user_prompt)

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from openai import APIError, OpenAI
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

        client = OpenAI(api_key=self.settings.openai_api_key)

        def _transient(exc: BaseException) -> bool:
            # Retry rate limits / 5xx, but not quota/auth (permanent → fail fast).
            code = getattr(exc, "code", None)
            if code in {"insufficient_quota", "invalid_api_key"}:
                return False
            return isinstance(exc, APIError)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=8),
            retry=retry_if_exception(_transient),
        )
        def _call():  # type: ignore[no-untyped-def]
            return client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=self.settings.timeout_seconds,
            )

        resp = _call()
        usage = resp.usage
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0
        content = resp.choices[0].message.content or ""
        return LLMResponse(content, inp, out, _estimate_cost(inp, out))

    def _complete_offline(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # ponytail: deterministic offline stub; real provider is _complete_openai
        # above, enabled by setting OPENAI_API_KEY. Lets the full pipeline + benchmark
        # run without keys/network.
        role = system_prompt.strip().splitlines()[0] if system_prompt.strip() else "assistant"
        content = f"[offline:{role}]\n{user_prompt.strip()}"
        inp = len(system_prompt.split()) + len(user_prompt.split())
        out = len(content.split())
        return LLMResponse(content, inp, out, _estimate_cost(inp, out))
