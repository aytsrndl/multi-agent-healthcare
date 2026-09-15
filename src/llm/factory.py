import os

from src.llm.base import LLMClient
from src.llm.openai_client import OpenAIClient
from src.llm.ollama_client import OllamaClient
from src.llm.arc_client import ARCClient
from src.llm.hf_client import HuggingFaceClient


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

    if provider == "arc":
        return ARCClient()

    if provider == "hf":
        return HuggingFaceClient()

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'"
    )