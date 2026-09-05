"""
LLM provider abstraction. Every provider (hosted API, local fine-tuned
model) implements the same contract, which is what lets `LLM_PROVIDER` be
a one-line config switch rather than a code change throughout the app.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, dict]:
        """Generate a response and validate it against `schema`.

        Returns:
            (parsed_object, usage_metadata) where usage_metadata contains at
            least `cost_usd`, `input_tokens`, and `output_tokens`.

        Raises:
            LLMProviderError: if the call fails or the response cannot be
                parsed into a valid instance of `schema`.
        """