from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    CHECK_BALANCE = "check_balance"
    TRANSACTION_HISTORY = "transaction_history"
    CARD_BLOCK = "card_block"
    TXN_DISPUTE = "txn_dispute"
    REFUND_STATUS = "refund_status"
    UPI_FAILURE = "upi_failure"
    STATEMENT_REQUEST = "statement_request"
    BENEFICIARY_ADD = "beneficiary_add"
    FD_RATES = "fd_rates"
    KYC_UPDATE = "kyc_update"
    LOAN_EMI = "loan_emi"
    OUT_OF_SCOPE = "out_of_scope"
    UNSAFE = "unsafe"


IntentKind = Literal["tool", "info", "redirect", "refuse"]

INTENT_KIND: dict[Intent, IntentKind] = {
    Intent.CHECK_BALANCE: "tool",
    Intent.TRANSACTION_HISTORY: "tool",
    Intent.CARD_BLOCK: "tool",
    Intent.TXN_DISPUTE: "tool",
    Intent.REFUND_STATUS: "tool",
    Intent.UPI_FAILURE: "tool",
    Intent.STATEMENT_REQUEST: "tool",
    Intent.BENEFICIARY_ADD: "tool",
    Intent.FD_RATES: "tool",
    Intent.KYC_UPDATE: "info",
    Intent.LOAN_EMI: "info",
    Intent.OUT_OF_SCOPE: "redirect",
    Intent.UNSAFE: "refuse",
}

TOOLS: dict[str, dict[str, Any]] = {
    "check_balance": {
        "name": "check_balance",
        "description": "Get the current balance of the customer's account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_type": {"type": "string", "enum": ["savings", "current"]},
            },
            "required": [],
        },
    },
    "get_transactions": {
        "name": "get_transactions",
        "description": "Fetch recent transactions, optionally within a date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "from_date": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}"},
                "to_date": {"type": "string", "pattern": r"\d{4}-\d{2}-\d{2}"},
            },
            "required": ["limit"],
        },
    },
    "block_card": {
        "name": "block_card",
        "description": "Block a debit or credit card immediately.",
        "parameters": {
            "type": "object",
            "properties": {
                "card_last4": {"type": "string", "pattern": r"\d{4}"},
                "reason": {"type": "string", "enum": ["lost", "stolen", "damaged", "fraud"]},
            },
            "required": ["card_last4", "reason"],
        },
    },
    "raise_dispute": {
        "name": "raise_dispute",
        "description": "Raise a dispute against a transaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "txn_id": {"type": "string"},
                "reason": {
                    "type": "string",
                    "enum": ["unauthorized", "duplicate", "failed_but_debited", "wrong_amount"],
                },
            },
            "required": ["txn_id", "reason"],
        },
    },
    "get_refund_status": {
        "name": "get_refund_status",
        "description": "Check the refund status for a failed or disputed transaction.",
        "parameters": {
            "type": "object",
            "properties": {"txn_id": {"type": "string"}},
            "required": ["txn_id"],
        },
    },
    "request_statement": {
        "name": "request_statement",
        "description": "Send an account statement to the customer's registered email.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "enum": ["last_month", "last_3_months", "last_6_months"]},
            },
            "required": ["period"],
        },
    },
    "get_fd_rates": {
        "name": "get_fd_rates",
        "description": "Get current fixed deposit interest rates.",
        "parameters": {
            "type": "object",
            "properties": {"tenure_months": {"type": "integer", "minimum": 3, "maximum": 120}},
            "required": [],
        },
    },
    "add_beneficiary": {
        "name": "add_beneficiary",
        "description": "Add a new payee for fund transfers.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "account_number": {"type": "string", "pattern": r"\d{9,18}"},
                "ifsc": {"type": "string", "pattern": r"[A-Z]{4}0[A-Z0-9]{6}"},
            },
            "required": ["name", "account_number", "ifsc"],
        },
    },
}


def validate_tool_call(name: str, arguments: dict[str, Any]) -> list[str]:
    spec = TOOLS.get(name)
    if spec is None:
        return [f"unknown tool: {name}"]
    errors: list[str] = []
    params = spec["parameters"]
    props = params["properties"]
    for key in params["required"]:
        if key not in arguments:
            errors.append(f"{name}: missing required '{key}'")
    for key, value in arguments.items():
        if key not in props:
            errors.append(f"{name}: unknown param '{key}'")
            continue
        errors.extend(_check_value(name, key, value, props[key]))
    return errors


def _check_value(tool: str, key: str, value: Any, schema: dict[str, Any]) -> list[str]:
    t = schema["type"]
    if t == "string":
        if not isinstance(value, str):
            return [f"{tool}.{key}: expected string"]
        if "enum" in schema and value not in schema["enum"]:
            return [f"{tool}.{key}: '{value}' not in enum"]
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            return [f"{tool}.{key}: '{value}' fails pattern"]
    elif t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{tool}.{key}: expected integer"]
        if "minimum" in schema and value < schema["minimum"]:
            return [f"{tool}.{key}: below minimum"]
        if "maximum" in schema and value > schema["maximum"]:
            return [f"{tool}.{key}: above maximum"]
    return []


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ToolCall:
        errors = validate_tool_call(self.name, self.arguments)
        if errors:
            raise ValueError("; ".join(errors))
        return self


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None


class Persona(BaseModel):
    age: int = Field(ge=18, le=80)
    city: str
    formality: Literal["casual", "neutral", "formal"]
    mood: Literal["calm", "confused", "frustrated", "angry"]
    hinglish_ratio: float = Field(ge=0.0, le=1.0)


class Sample(BaseModel):
    id: str
    intent: Intent
    persona: Persona
    messages: list[Message]
    multi_turn: bool = False
    noisy: bool = False
    source: str = "synthetic"

    # system prompt is attached at render time, not stored per sample
    @model_validator(mode="after")
    def _check(self) -> Sample:
        if not self.messages or self.messages[0].role != "user":
            raise ValueError("conversation must start with a user message")
        if self.messages[-1].role != "assistant":
            raise ValueError("conversation must end with an assistant message")
        for m in self.messages:
            if m.tool_calls and m.role != "assistant":
                raise ValueError("tool_calls only allowed on assistant messages")
        return self


class DPOSample(BaseModel):
    id: str
    intent: Intent
    prompt: list[Message]
    chosen: Message
    rejected: Message
    rejection_reason: Literal[
        "hallucination", "wrong_tool", "bad_args", "unsafe", "over_refusal", "wrong_language"
    ]