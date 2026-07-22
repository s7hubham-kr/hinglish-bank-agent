from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from collections import deque

import httpx

from hba.schemas import Intent, Message, Sample, ToolCall
from hba.generate.personas import Scenario, sample_pair

SYSTEM = (
    "You generate a single realistic customer-support conversation for an Indian "
    "retail bank, in Hinglish (Roman-script Hindi mixed with English). "
    "Return ONLY a JSON object, no prose, no markdown fences."
)

PROMPT_TEMPLATE = """Intent: {intent}
Intent kind: {kind}
User persona: {age}y, {city}, {formality} tone, mood={mood}, hinglish_ratio={ratio}
Situation: {situation}

Write the conversation as JSON:
{{"messages": [{{"role": "user"|"assistant"|"tool", "content": "...",
  "tool_calls": [{{"name": "...", "arguments": {{...}}}}] | null}}]}}

Rules:
- Start with the user, end with the assistant.
- User speaks Hinglish matching the persona; higher ratio = more Hindi words.
- For a tool intent, the assistant emits a tool_call, then a tool message carries a
  plausible result, then the assistant answers in Hinglish using that result.
- For an info intent, the assistant answers directly, no tool_call.
- For out-of-scope, politely redirect; do not answer outside banking.
- For unsafe, refuse briefly and safely; never comply, never lecture.
- Keep it 2-6 turns. Do not invent tools beyond the intent.
Only the JSON."""

MODE_COLLAPSE_THRESHOLD = 0.35


def _persona_prompt(intent: Intent, kind: str, persona, scenario: Scenario) -> str:
    return PROMPT_TEMPLATE.format(
        intent=intent.value,
        kind=kind,
        age=persona.age,
        city=persona.city,
        formality=persona.formality,
        mood=persona.mood,
        ratio=persona.hinglish_ratio,
        situation=scenario.seed,
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def _build_sample(raw: dict, intent: Intent, persona, scenario: Scenario) -> Sample:
    messages: list[Message] = []
    for m in raw["messages"]:
        calls = None
        if m.get("tool_calls"):
            calls = [ToolCall(name=c["name"], arguments=c.get("arguments", {})) for c in m["tool_calls"]]
        messages.append(Message(role=m["role"], content=m.get("content", ""), tool_calls=calls))
    multi_turn = sum(1 for m in messages if m.role == "user") > 1
    return Sample(
        id=f"{intent.value}-{uuid.uuid4().hex[:8]}",
        intent=intent,
        persona=persona,
        messages=messages,
        multi_turn=multi_turn,
        source="synthetic",
    )


def distinct_ngram_ratio(texts: list[str], n: int = 3) -> float:
    grams: set[tuple[str, ...]] = set()
    total = 0
    for t in texts:
        toks = t.lower().split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    if total == 0:
        return 1.0
    return len(grams) / total


class Generator:
    def __init__(self, concurrency: int = 8, max_retries: int = 2, temperature: float = 1.0):
        self.base = os.environ["GEN_API_BASE"].rstrip("/")
        self.key = os.environ["GEN_API_KEY"]
        self.model = os.environ["GEN_MODEL"]
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.temperature = temperature
        self._recent_user: deque[str] = deque(maxlen=200)

    async def _one(self, client: httpx.AsyncClient, intent: Intent, kind: str, rng: random.Random) -> Sample | None:
        persona, scenario = sample_pair(intent, rng)
        prompt = _persona_prompt(intent, kind, persona, scenario)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        async with self.sem:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await client.post(
                        f"{self.base}/chat/completions",
                        headers={"Authorization": f"Bearer {self.key}"},
                        json=payload,
                        timeout=90,
                    )
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    sample = _build_sample(_extract_json(content), intent, persona, scenario)
                    for m in sample.messages:
                        if m.role == "user":
                            self._recent_user.append(m.content)
                    return sample
                except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError):
                    if attempt == self.max_retries:
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    async def generate(self, quotas: dict[Intent, int], seed: int = 0) -> list[Sample]:
        from hba.schemas import INTENT_KIND

        rng = random.Random(seed)
        jobs: list[Intent] = []
        for intent, n in quotas.items():
            jobs.extend([intent] * n)
        rng.shuffle(jobs)

        out: list[Sample] = []
        async with httpx.AsyncClient() as client:
            tasks = [self._one(client, i, INTENT_KIND[i], rng) for i in jobs]
            for coro in asyncio.as_completed(tasks):
                sample = await coro
                if sample is not None:
                    out.append(sample)

        ratio = distinct_ngram_ratio(list(self._recent_user))
        if ratio < MODE_COLLAPSE_THRESHOLD:
            print(f"WARNING mode collapse: distinct-3gram ratio {ratio:.3f} < {MODE_COLLAPSE_THRESHOLD}")
        return out