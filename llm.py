"""LLM factory. Provider is chosen with LLM_PROVIDER (gemini | openai)."""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from core.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'openai'.")
