"""
Anthropic Claude provider. Prompts the model to emit a single JSON object
and validates it against the target Pydantic schema before returning it,
so callers never see a malformed or partially-structured response.
"""

import json

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.core.exceptions import LLMProviderError
from src.llm.base_provider import BaseLLMProvider, SchemaT

logger = get_logger(__name__)

# Approximate per-million-token pricing, used for cost observability only
# (not billing-accurate; update if the configured model's pricing changes).
_INPUT_COST_PER_MTOK = 3.0
_OUTPUT_COST_PER_MTOK = 15.0


class AnthropicProvider(BaseLLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY is not configured")
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, dict]:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LLMProviderError(f"Anthropic API call failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        parsed = self._parse_response(text, schema)

        usage = response.usage
        cost = (
            usage.input_tokens / 1_000_000 * _INPUT_COST_PER_MTOK
            + usage.output_tokens / 1_000_000 * _OUTPUT_COST_PER_MTOK
        )
        logger.info(
            "anthropic_provider.generate_structured",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=round(cost, 6),
        )
        return parsed, {
            "cost_usd": cost,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
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