import os
from typing import TypeVar

from ollama import Client
from pydantic import BaseModel

from src.llm.base import LLMClient


T = TypeVar("T", bound=BaseModel)


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL")
        self.host = host or os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )

        if not self.model:
            raise ValueError(
                "OLLAMA_MODEL environment variable is not set."
            )

        self.client = Client(
            host=self.host,
        )

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
    ) -> T:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=output_schema.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        content = response["message"]["content"]

        try:
            parsed = output_schema.model_validate_json(
                content
            )

        except Exception as error:
            raise ValueError(
                f"{self.model} did not return a valid "
                f"{output_schema.__name__}.\n"
                f"Raw response:\n{content}"
            ) from error

        return parsed