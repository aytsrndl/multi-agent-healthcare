import os

from src.llm.base import LLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.ollama_client import OllamaClient


def create_llm_client() -> LLMClient:
    """
    Create the configured LLM backend.
    """

    provider = os.getenv(
        "LLM_PROVIDER",
        "openai",
    ).strip().lower()

    if provider == "openai":
        return OpenAIClient()

    if provider == "ollama":
        return OllamaClient()

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'"
    )