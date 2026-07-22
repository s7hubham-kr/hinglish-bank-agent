from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

import httpx

from hba.schemas import INTENT_KIND, Sample

RUBRIC_SYSTEM = (
    "You are a strict data-quality judge for a Hinglish banking assistant dataset. "
    "Score one conversation on four axes, each 1-5. Reply with ONLY a JSON object, "
    "no prose, no markdown, no chain of thought."
)

RUBRIC_PROMPT = """Intent: {intent} (kind: {kind})

Conversation:
{convo}

Score each axis 1-5:
- language: user speaks natural Roman-script Hinglish (not pure English, not Devanagari, sensible code-mixing)
- correctness: banking-accurate; for a tool intent the assistant calls the RIGHT tool with valid args; for info it answers correctly; numbers/claims are plausible
- safety: unsafe requests are refused, out-of-scope is redirected, and in-scope requests are NOT over-refused
- naturalness: fits the persona and mood, reads like a real customer chat, not templated

Return exactly:
{{"language": int, "correctness": int, "safety": int, "naturalness": int, "reason": "one short clause"}}"""

AXES = ("language", "correctness", "safety", "naturalness")


@dataclass
class Verdict:
    sample_id: str
    scores: dict[str, int]
    reason: str
    passed: bool

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


def _render_convo(sample: Sample) -> str:
    lines: list[str] = []
    for m in sample.messages:
        if m.tool_calls:
            calls = "; ".join(
                f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in m.tool_calls
            )
            lines.append(f"{m.role}: [tool_call] {calls}")
        else:
            lines.append(f"{m.role}: {m.content}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge output")
    return json.loads(text[start : end + 1])


def _decide(scores: dict[str, int], axis_floor: int, mean_floor: float) -> bool:
    if any(scores.get(a, 0) < axis_floor for a in AXES):
        return False
    return (sum(scores[a] for a in AXES) / len(AXES)) >= mean_floor


class Judge:
    def __init__(
        self,
        concurrency: int = 8,
        max_retries: int = 2,
        axis_floor: int = 3,
        mean_floor: float = 3.5,
    ):
        self.base = os.environ["JUDGE_API_BASE"].rstrip("/")
        self.key = os.environ["JUDGE_API_KEY"]
        self.model = os.environ["JUDGE_MODEL"]
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.axis_floor = axis_floor
        self.mean_floor = mean_floor

    async def _one(self, client: httpx.AsyncClient, sample: Sample) -> Verdict | None:
        kind = INTENT_KIND[sample.intent]
        prompt = RUBRIC_PROMPT.format(
            intent=sample.intent.value, kind=kind, convo=_render_convo(sample)
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": RUBRIC_SYSTEM},
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
                    raw = _extract_json(content)
                    scores = {a: int(raw[a]) for a in AXES}
                    passed = _decide(scores, self.axis_floor, self.mean_floor)
                    return Verdict(sample.id, scores, str(raw.get("reason", "")), passed)
                except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError):
                    if attempt == self.max_retries:
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    async def judge(self, samples: list[Sample]) -> list[Verdict]:
        out: list[Verdict] = []
        async with httpx.AsyncClient() as client:
            tasks = [self._one(client, s) for s in samples]
            for coro in asyncio.as_completed(tasks):
                v = await coro
                if v is not None:
                    out.append(v)
        return out


def agreement(verdicts: list[Verdict], human_labels: dict[str, bool]) -> dict[str, float]:
    paired = [(v, human_labels[v.sample_id]) for v in verdicts if v.sample_id in human_labels]
    if not paired:
        return {"n": 0, "agreement": 0.0}
    agree = sum(1 for v, h in paired if v.passed == h)
    judge_pass_human_fail = sum(1 for v, h in paired if v.passed and not h)
    judge_fail_human_pass = sum(1 for v, h in paired if not v.passed and h)
    n = len(paired)
    return {
        "n": n,
        "agreement": agree / n,
        "too_lenient": judge_pass_human_fail / n,
        "too_strict": judge_fail_human_pass / n,
    }