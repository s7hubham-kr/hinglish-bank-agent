from __future__ import annotations

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from hba.schemas import Sample

_WS = re.compile(r"\s+")


def _dedup_text(sample: Sample) -> str:
    parts: list[str] = []
    for m in sample.messages:
        if m.role == "tool":
            continue
        if m.content:
            parts.append(m.content)
        for call in m.tool_calls or []:
            args = ",".join(f"{k}={v}" for k, v in sorted(call.arguments.items()))
            parts.append(f"{call.name}({args})")
    text = " ".join(parts).lower()
    return _WS.sub(" ", text).strip()


def _norm_key(sample: Sample) -> str:
    return hashlib.sha1(_dedup_text(sample).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5) -> set[str]:
    text = _WS.sub(" ", text)
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh


@dataclass
class DedupStats:
    total: int
    exact_removed: int
    near_removed: int
    kept: int
    removed_by_intent: dict[str, int] = field(default_factory=dict)


def dedup(
    samples: list[Sample],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> tuple[list[Sample], DedupStats]:
    total = len(samples)

    seen: set[str] = set()
    exact_kept: list[Sample] = []
    for s in samples:
        key = _norm_key(s)
        if key in seen:
            continue
        seen.add(key)
        exact_kept.append(s)
    exact_removed = total - len(exact_kept)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Sample] = []
    near_removed = 0
    for i, s in enumerate(exact_kept):
        text = _dedup_text(s)
        mh = _minhash(text, num_perm)
        if lsh.query(mh):
            near_removed += 1
            continue
        lsh.insert(str(i), mh)
        kept.append(s)

    stats = DedupStats(
        total=total,
        exact_removed=exact_removed,
        near_removed=near_removed,
        kept=len(kept),
    )
    return kept, stats