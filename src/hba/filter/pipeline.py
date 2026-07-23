from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from hba.augment.noise import apply_noise
from hba.filter import dedup as dedup_mod
from hba.filter import programmatic
from hba.schemas import Intent

OVERALL_REJECT_GATE = 0.10
SLICE_REJECT_GATE = 0.20


@dataclass
class StageCounts:
    per_intent_in: dict[str, int] = field(default_factory=dict)
    per_intent_rejected: dict[str, int] = field(default_factory=dict)

    def rate(self, intent: str) -> float:
        n = self.per_intent_in.get(intent, 0)
        if n == 0:
            return 0.0
        return self.per_intent_rejected.get(intent, 0) / n


@dataclass
class PipelineReport:
    generated: int
    after_programmatic: int
    after_dedup: int
    judged: int
    passed: int
    programmatic: StageCounts
    judge: StageCounts
    exact_removed: int
    near_removed: int
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def overall_reject_rate(self) -> float:
        if self.generated == 0:
            return 0.0
        return 1.0 - (self.passed / self.generated)

    @property
    def gate_passed(self) -> bool:
        return self.overall_reject_rate < OVERALL_REJECT_GATE and not self.slices_to_regenerate

    @property
    def slices_to_regenerate(self) -> list[str]:
        out: list[str] = []
        for intent in {**self.programmatic.per_intent_in, **self.judge.per_intent_in}:
            prog = self.programmatic.rate(intent)
            jud = self.judge.rate(intent)

            gen_in = self.programmatic.per_intent_in.get(intent, 0)
            if gen_in == 0:
                continue
            prog_rej = self.programmatic.per_intent_rejected.get(intent, 0)
            survivors = gen_in - prog_rej
            jud_rej = self.judge.per_intent_rejected.get(intent, 0)
            total_rej = prog_rej + jud_rej
            if total_rej / gen_in > SLICE_REJECT_GATE:
                out.append(intent)
        return sorted(out)


def run_filters(
    samples: list[Sample],
    verdicts: list[Verdict] | None = None,
    noise_fraction: float = 0.3,
    dedup_threshold: float = 0.8,
    seed: int = 0,
) -> tuple[list[Sample], PipelineReport]:
    generated = len(samples)

    prog = StageCounts()
    for s in samples:
        prog.per_intent_in[s.intent.value] = prog.per_intent_in.get(s.intent.value, 0) + 1

    reasons: dict[str, int] = defaultdict(int)
    prog_kept: list[Sample] = []
    for s in samples:
        result = programmatic.check(s)
        if result.ok:
            prog_kept.append(s)
        else:
            prog.per_intent_rejected[s.intent.value] = prog.per_intent_rejected.get(s.intent.value, 0) + 1
            for r in result.reasons:
                key = r.split("(")[0].strip()
                reasons[key] += 1

    deduped, dstats = dedup_mod.dedup(prog_kept, threshold=dedup_threshold)

    noised = apply_noise(deduped, fraction=noise_fraction, seed=seed)

    judge_counts = StageCounts()
    for s in noised:
        judge_counts.per_intent_in[s.intent.value] = judge_counts.per_intent_in.get(s.intent.value, 0) + 1

    if verdicts is None:
        passed_samples = noised
        judged = 0
    else:
        verdict_by_id = {v.sample_id: v for v in verdicts}
        passed_samples = []
        for s in noised:
            v = verdict_by_id.get(s.id)
            if v is None or v.passed:
                passed_samples.append(s)
            else:
                judge_counts.per_intent_rejected[s.intent.value] = (
                    judge_counts.per_intent_rejected.get(s.intent.value, 0) + 1
                )
        judged = sum(1 for s in noised if s.id in verdict_by_id)

    report = PipelineReport(
        generated=generated,
        after_programmatic=len(prog_kept),
        after_dedup=len(deduped),
        judged=judged,
        passed=len(passed_samples),
        programmatic=prog,
        judge=judge_counts,
        exact_removed=dstats.exact_removed,
        near_removed=dstats.near_removed,
        reject_reasons=dict(reasons),
    )
    return passed_samples, report


def format_report(report: PipelineReport) -> str:
    lines = [
        f"generated:          {report.generated}",
        f"after programmatic: {report.after_programmatic}",
        f"after dedup:        {report.after_dedup} (exact -{report.exact_removed}, near -{report.near_removed})",
        f"judged:             {report.judged}",
        f"passed:             {report.passed}",
        f"overall reject:     {report.overall_reject_rate:.1%}  (gate <{OVERALL_REJECT_GATE:.0%})",
        f"gate passed:        {report.gate_passed}",
    ]
    if report.slices_to_regenerate:
        lines.append(f"regenerate slices:  {', '.join(report.slices_to_regenerate)}")
    if report.reject_reasons:
        top = sorted(report.reject_reasons.items(), key=lambda x: -x[1])[:5]
        lines.append("top reject reasons: " + ", ".join(f"{k}={v}" for k, v in top))
    return "\n".join(lines)