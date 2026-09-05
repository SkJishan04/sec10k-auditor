"""
Local inference provider backed by a QLoRA/DPO fine-tuned Llama-3 8B
adapter (see training/train_qlora_dpo.py). Heavy ML dependencies are
imported lazily so the rest of the application (API, retrieval, tests)
can run without requiring a GPU or the 'training' extras installed.
"""

import json

from pydantic import ValidationError

from src.config.settings import get_settings
from src.core.exceptions import LLMProviderError
from src.llm.base_provider import BaseLLMProvider, SchemaT


class LocalQLoRAProvider(BaseLLMProvider):
    def __init__(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMProviderError(
                "local_qlora provider requires the 'training' extras: "
                "pip install -e '.[training]'"
            ) from exc

        settings = get_settings()
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(settings.local_base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.local_base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        try:
            self._model = PeftModel.from_pretrained(base_model, settings.local_adapter_path)
        except Exception as exc:
            raise LLMProviderError(
                f"Could not load LoRA adapter from {settings.local_adapter_path}: {exc}"
            ) from exc
        self._model.eval()

    def generate_structured(self, prompt: str, schema: type[SchemaT]) -> tuple[SchemaT, dict]:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=2000, do_sample=False, temperature=None, top_p=None
            )

        input_len = inputs["input_ids"].shape[1]
        decoded = self._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        try:
            data = json.loads(decoded.strip())
            parsed = schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMProviderError(f"Local model produced invalid structured output: {exc}") from exc

        output_tokens = output_ids.shape[1] - input_len
        return parsed, {"cost_usd": 0.0, "input_tokens": input_len, "output_tokens": output_tokens}