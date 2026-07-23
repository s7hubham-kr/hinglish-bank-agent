"""Phase 1 dataset build: generate -> filter -> judge -> write sft.jsonl."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from hba.filter.judge import Judge
from hba.filter.pipeline import format_report, run_filters
from hba.generate.generator import Generator
from hba.schemas import Intent, Sample

DEFAULT_TOTAL = 15000

# tool intents carry the training signal, so they get the bulk of the budget.
# out_of_scope and unsafe need enough volume to hold the boundary without
# teaching the model to over-refuse.
INTENT_WEIGHTS: dict[Intent, float] = {
    Intent.CHECK_BALANCE: 0.10,
    Intent.TRANSACTION_HISTORY: 0.10,
    Intent.CARD_BLOCK: 0.09,
    Intent.TXN_DISPUTE: 0.09,
    Intent.REFUND_STATUS: 0.08,
    Intent.UPI_FAILURE: 0.08,
    Intent.STATEMENT_REQUEST: 0.07,
    Intent.BENEFICIARY_ADD: 0.07,
    Intent.FD_RATES: 0.06,
    Intent.KYC_UPDATE: 0.07,
    Intent.LOAN_EMI: 0.07,
    Intent.OUT_OF_SCOPE: 0.06,
    Intent.UNSAFE: 0.06,
}

REQUIRED_GEN_VARS = ("GEN_API_BASE", "GEN_API_KEY", "GEN_MODEL")
REQUIRED_JUDGE_VARS = ("JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL")


def build_quotas(total: int) -> dict[Intent, int]:
    quotas = {intent: max(1, round(total * w)) for intent, w in INTENT_WEIGHTS.items()}
    drift = total - sum(quotas.values())
    if drift:
        largest = max(quotas, key=lambda i: quotas[i])
        quotas[largest] += drift
    return quotas


def check_env(names: tuple[str, ...], context: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"missing env vars for {context}: {', '.join(missing)}")


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n")


def print_quotas(quotas: dict[Intent, int]) -> None:
    for intent, n in sorted(quotas.items(), key=lambda kv: -kv[1]):
        print(f"  {intent.value:<22} {n}")
    print(f"  {'total':<22} {sum(quotas.values())}")


async def run(args: argparse.Namespace) -> int:
    total = args.limit if args.limit else args.total
    quotas = build_quotas(total)

    print(f"quotas (total {total}):")
    print_quotas(quotas)

    if args.dry_run:
        print("\ndry run: no API calls made")
        return 0

    check_env(REQUIRED_GEN_VARS, "generation")
    if not args.no_judge:
        check_env(REQUIRED_JUDGE_VARS, "judging")

    gen = Generator(concurrency=args.concurrency, temperature=args.temperature)
    samples = await gen.generate(quotas, seed=args.seed)
    print(f"\ngenerated {len(samples)} of {total} requested")
    if not samples:
        sys.exit("generation produced nothing; check credentials and model name")

    # judge needs post-noise text, so filter once without verdicts, judge the
    # survivors, then re-filter with verdicts to get one coherent report.
    staged, _ = run_filters(
        samples,
        verdicts=None,
        noise_fraction=args.noise_fraction,
        dedup_threshold=args.dedup_threshold,
        seed=args.seed,
    )

    verdicts = None
    if not args.no_judge:
        print(f"judging {len(staged)} samples")
        judge = Judge(concurrency=args.concurrency)
        verdicts = await judge.judge(staged)
        print(f"got {len(verdicts)} verdicts")

    kept, report = run_filters(
        samples,
        verdicts=verdicts,
        noise_fraction=args.noise_fraction,
        dedup_threshold=args.dedup_threshold,
        seed=args.seed,
    )

    print()
    print(format_report(report))

    out = Path(args.out)
    write_jsonl(kept, out)
    print(f"\nwrote {len(kept)} samples to {out}")

    if verdicts is not None:
        vpath = out.with_name(out.stem + "_verdicts.jsonl")
        with vpath.open("w", encoding="utf-8") as f:
            for v in verdicts:
                f.write(
                    json.dumps(
                        {
                            "sample_id": v.sample_id,
                            "scores": v.scores,
                            "reason": v.reason,
                            "passed": v.passed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {len(verdicts)} verdicts to {vpath}")

    if not report.gate_passed:
        print("\nGATE FAILED - do not proceed to training")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Hinglish banking SFT dataset.")
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="samples to generate")
    p.add_argument("--limit", type=int, help="override total, for pilot runs")
    p.add_argument("--out", default="data/sft.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--noise-fraction", type=float, default=0.3)
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument("--dry-run", action="store_true", help="print quotas, make no API calls")
    p.add_argument("--no-judge", action="store_true", help="skip the paid judge pass")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))