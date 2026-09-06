"""
OpenAI provider. Implements the same BaseLLMProvider contract as
AnthropicProvider, so LLM_PROVIDER=openai is a pure config switch with no
changes needed anywhere else in the app (orchestrator, services, routes
all depend on the abstract interface, not on a specific vendor SDK).

Uses OpenAI's native structured-output mode (response_format with a JSON
schema) rather than prompt-only JSON instructions, since it's more
reliable at schema conformance than trusting the model to self-format --
this reduces (but does not replace) the need for the hallucination guard,
which still verifies numeric values against source text regardless of how
well-formed the JSON is.
"""

import json

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.core.exceptions import LLMProviderError
from src.llm.base_provider import BaseLLMProvider, SchemaT

logger = get_logger(__name__)

# Approximate per-million-token pricing for observability only (update if
# the configured model's pricing changes; not used for actual billing).
_INPUT_COST_PER_MTOK = 2.5
_OUTPUT_COST_PER_MTOK = 10.0


class OpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, dict]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": False,
                    },
                },
            )
        except Exception as exc:
            raise LLMProviderError(f"OpenAI API call failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        parsed = self._parse_response(text, schema)

        usage = response.usage
        cost = (
            usage.prompt_tokens / 1_000_000 * _INPUT_COST_PER_MTOK
            + usage.completion_tokens / 1_000_000 * _OUTPUT_COST_PER_MTOK
        )
        logger.info(
            "openai_provider.generate_structured",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=round(cost, 6),
        )
        return parsed, {
            "cost_usd": cost,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

    @staticmethod
    def _parse_response(text: str, schema: type[SchemaT]) -> SchemaT:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"LLM did not return valid JSON: {exc}") from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMProviderError(f"LLM output failed schema validation: {exc}") from exc