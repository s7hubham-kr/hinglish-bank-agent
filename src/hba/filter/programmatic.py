from __future__ import annotations

import re
from dataclasses import dataclass, field

from hba.schemas import INTENT_KIND, Intent, Sample, validate_tool_call

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[a-zA-Z]")
WORD = re.compile(r"[^\s]+")

HINDI_MARKERS = {
    "hai", "hain", "ho", "hua", "kar", "karo", "karna", "kya", "kyu", "kyun",
    "mera", "meri", "mujhe", "aap", "aapka", "apna", "nahi", "nahin", "haan",
    "bhai", "yaar", "abhi", "gaya", "gayi", "raha", "rahi", "diya", "kiya",
    "kaise", "kaisa", "kitna", "kitni", "batao", "bata", "chahiye", "wala",
    "se", "ka", "ki", "ke", "ko", "me", "mein", "par", "bhi", "toh", "phir",
}

MIN_USER_TOKENS = 2
MAX_USER_TOKENS = 80
MIN_ASSISTANT_TOKENS = 2
MAX_ASSISTANT_TOKENS = 200
CMI_MIN = 0.15
CMI_MAX = 0.9


@dataclass
class FilterResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return WORD.findall(text)


def code_mixing_index(text: str) -> float:
    toks = [t.lower() for t in _tokens(text) if LATIN.search(t)]
    if not toks:
        return 0.0
    hindi = sum(1 for t in toks if t.strip(".,!?") in HINDI_MARKERS)
    return hindi / len(toks)


def _user_turns(sample: Sample) -> list[str]:
    return [m.content for m in sample.messages if m.role == "user"]


def _assistant_turns(sample: Sample) -> list[str]:
    return [m.content for m in sample.messages if m.role == "assistant"]


def _has_tool_call(sample: Sample) -> bool:
    return any(m.tool_calls for m in sample.messages)


def check(sample: Sample) -> FilterResult:
    reasons: list[str] = []

    for m in sample.messages:
        for call in m.tool_calls or []:
            errs = validate_tool_call(call.name, call.arguments)
            reasons.extend(errs)

    users = _user_turns(sample)
    assistants = _assistant_turns(sample)

    if not users:
        reasons.append("no user turn")
    if not assistants:
        reasons.append("no assistant turn")

    for t in users:
        if DEVANAGARI.search(t):
            reasons.append("devanagari in user turn")
            break

    for t in users:
        n = len(_tokens(t))
        if n < MIN_USER_TOKENS:
            reasons.append("user turn too short")
        elif n > MAX_USER_TOKENS:
            reasons.append("user turn too long")

    for t in assistants:
        n = len(_tokens(t))
        if n == 0:
            continue
        if n < MIN_ASSISTANT_TOKENS:
            reasons.append("assistant turn too short")
        elif n > MAX_ASSISTANT_TOKENS:
            reasons.append("assistant turn too long")

    joined = " ".join(users)
    cmi = code_mixing_index(joined)
    if joined.strip() and (cmi < CMI_MIN or cmi > CMI_MAX):
        reasons.append(f"cmi out of band ({cmi:.2f})")

    kind = INTENT_KIND[sample.intent]
    has_call = _has_tool_call(sample)
    if kind == "tool" and not has_call:
        reasons.append("tool intent without tool call")
    if kind in ("refuse", "redirect") and has_call:
        reasons.append(f"{kind} intent should not call a tool")

    return FilterResult(ok=not reasons, reasons=reasons)