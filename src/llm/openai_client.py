import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src.llm.base import LLMClient


T = TypeVar("T", bound=BaseModel)


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL")

        if not self.model:
            raise ValueError(
                "OPENAI_MODEL environment variable is not set."
            )

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = OpenAI()

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
    ) -> T:

        response = self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=output_schema,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise ValueError(
                f"{self.model} did not return a valid "
                f"{output_schema.__name__}."
            )

        return parsed