"""
app/core/llm.py

WHY a central LLM factory:
  - Every graph node (planner, reflection, report generator) needs an LLM client.
  - Centralising instantiation means switching providers requires changing one
    env var, not hunting across multiple files.

MODEL CHOICES:
  - OpenAI: gpt-4o — best balance of reasoning quality and speed.
  - Anthropic: claude-3-5-sonnet-20241022 — stronger at long structured documents.
  - Ollama: any locally-served model (default: llama3.2). Free, no API key needed.
    Requires Ollama running at OLLAMA_BASE_URL (default: http://localhost:11434).
  - temperature=0 for all nodes: research tasks require deterministic output.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.core.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    if settings.active_llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.openai_api_key)

    if settings.active_llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=settings.anthropic_api_key,
        )

    # ollama — runs locally, no API key required
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
