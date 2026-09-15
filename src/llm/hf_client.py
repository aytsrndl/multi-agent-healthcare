import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.llm.base import LLMClient


T = TypeVar("T", bound=BaseModel)


class HuggingFaceClient(LLMClient):
    def __init__(
        self,
        model: str | None = None,
    ):
        self.model = model or os.getenv("HF_MODEL")

        if not self.model:
            raise ValueError(
                "HF_MODEL environment variable is not set."
            )

        api_key = os.getenv("HF_API_KEY")

        if not api_key:
            raise ValueError(
                "HF_API_KEY environment variable is not set."
            )

        base_url = os.getenv("HF_BASE_URL")

        if not base_url:
            raise ValueError(
                "HF_BASE_URL environment variable is not set."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()

        if text.startswith("```json"):
            text = text[len("```json"):]

        elif text.startswith("```"):
            text = text[len("```"):]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    @staticmethod
    def _extract_matching_json(
        text: str,
        output_schema: type[T],
    ) -> T:
        """
        Find a JSON object in the model response that matches
        the requested Pydantic schema.

        Some models may echo the provided JSON schema before
        returning the actual structured answer.
        """

        decoder = json.JSONDecoder()

        # Search from the end because the model's actual answer
        # normally appears after any echoed schema/instructions.
        object_starts = [
            index
            for index, character in enumerate(text)
            if character == "{"
        ]

        for start in reversed(object_starts):
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue

            try:
                return output_schema.model_validate(obj)
            except ValidationError:
                continue

        raise ValueError(
            f"No JSON object matching "
            f"{output_schema.__name__} was found."
        )

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

        cleaned = self._strip_code_fences(content)

        # Normal path: the model returned exactly what we asked for.
        try:
            return output_schema.model_validate_json(cleaned)

        except (ValidationError, ValueError):
            pass

        # Recovery path: the model added extra text or echoed
        # the schema before returning the actual JSON object.
        try:
            return self._extract_matching_json(
                content,
                output_schema,
            )

        except ValueError as error:
            raise ValueError(
                f"{self.model} did not return a valid "
                f"{output_schema.__name__}.\n\n"
                f"Raw response:\n{content}"
            ) from error