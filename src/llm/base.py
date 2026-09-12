from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """
    Common interface for every LLM backend.

    Agent code should depend on this class rather than
    directly depending on OpenAI, Ollama, ARC, etc.
    """

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
    ) -> T:
        """
        Generate a structured response matching
        the supplied Pydantic schema.
        """
        raise NotImplementedError