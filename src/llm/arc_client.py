import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.llm.base import LLMClient


T = TypeVar("T", bound=BaseModel)


class ARCClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
    ):
        self.model = model or os.getenv("ARC_MODEL")

        if not self.model:
            raise ValueError(
                "ARC_MODEL environment variable is not set."
            )

        api_key = os.getenv("ARC_API_KEY")

        if not api_key:
            raise ValueError(
                "ARC_API_KEY environment variable is not set."
            )

        base_url = os.getenv(
            "ARC_BASE_URL",
            "https://llm-api.arc.vt.edu/api/v1",
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        Remove Markdown JSON fences if a model returns them
        despite being instructed to return raw JSON.
        """
        text = text.strip()

        if text.startswith("```json"):
            text = text[len("```json"):]

        elif text.startswith("```"):
            text = text[len("```"):]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
    ) -> T:

        schema = json.dumps(
            output_schema.model_json_schema(),
            indent=2,
        )

        structured_prompt = f"""
{prompt}

STRUCTURED OUTPUT REQUIREMENT:

Return ONLY valid JSON.

Your response must conform exactly to this JSON schema:

{schema}

Do not include:
- Markdown
- code fences
- commentary before the JSON
- commentary after the JSON

Return only the JSON object.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": structured_prompt,
                }
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                f"{self.model} returned an empty response."
            )

        content = self._strip_code_fences(content)

        try:
            parsed = output_schema.model_validate_json(
                content
            )

        except ValidationError as error:
            raise ValueError(
                f"{self.model} returned JSON that did not "
                f"match {output_schema.__name__}.\n\n"
                f"Raw response:\n{content}"
            ) from error

        except ValueError as error:
            raise ValueError(
                f"{self.model} did not return valid JSON.\n\n"
                f"Raw response:\n{content}"
            ) from error

        return parsed