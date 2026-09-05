"""
Prompt construction for the structured extraction step.

The prompt embeds the target Pydantic schema directly (via
`model_json_schema()`) so the model is instructed against the exact same
contract that will later validate its output -- there is no separate,
hand-maintained description of the JSON shape to drift out of sync.
"""

import json

from src.core.schemas import FinancialRiskReport, RetrievedChunk

_SYSTEM_INSTRUCTIONS = """You are a financial auditing assistant analyzing a SEC 10-K filing.

Using ONLY the filing excerpts provided, extract numeric financial metrics and identify \
risk findings such as off-balance-sheet liabilities, aggressive revenue recognition, \
related-party transactions, contingent liabilities, and going-concern issues.

STRICT RULES:
- Every numeric metric MUST include the exact chunk_id and a short excerpt (<=600 chars) \
copied verbatim from the source text that contains that number.
- Do NOT invent a number that does not literally appear in the excerpts below.
- If you are not confident a number is stated in the text, omit it rather than guess.
- Respond with ONLY a single JSON object matching the schema below. No prose, no markdown \
code fences, no explanation outside the JSON."""


def build_extraction_prompt(filing_id: int, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[chunk_id={c.chunk_id} page={c.page_number}]\n{c.text}" for c in chunks
    )
    schema = json.dumps(FinancialRiskReport.model_json_schema())

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"filing_id: {filing_id}\n\n"
        f"JSON SCHEMA:\n{schema}\n\n"
        f"FILING EXCERPTS:\n{context}\n"
    )