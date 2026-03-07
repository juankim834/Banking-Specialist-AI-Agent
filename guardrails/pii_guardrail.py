import re
from agents import GuardrailFunctionOutput, RunContextWrapper, input_guardrail, TResponseInputItem
from pydantic import BaseModel

PII_PATTERNS = [
    r"\b\d{16}\b", # Credit/Debit card numbers
    r"\b\d{3}-\d{2}-\d{4}\b", # SSN
    r"\b[A-Z]{2}\d{6}[A-Z]\b", # Passport-style IDs
    r"\b\d{9,18}\b", # Bank account numbers
]

class PIIDetectionOutput(BaseModel):
    pii_detected: bool
    reason: str

@input_guardrail
async def pii_guardrail(
    ctx: RunContextWrapper, agent, input: list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    text = str(input)
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return GuardrailFunctionOutput(
                output_info=PIIDetectionOutput(
                    pii_detected=True,
                    reason=f"Potential PII detected matching pattern: {pattern}"
                ),
                tripwire_triggered=True,
            )
    return GuardrailFunctionOutput(
        output_info=PIIDetectionOutput(pii_detected=False, reason="No PII detected"),
        tripwire_triggered=False,
    )
