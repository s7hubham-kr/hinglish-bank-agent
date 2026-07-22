from __future__ import annotations

import random
from dataclasses import dataclass

from hba.schemas import Intent, Persona

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Patna", "Indore",
    "Bhopal", "Nagpur", "Kanpur", "Surat", "Kochi", "Guwahati",
]

FORMALITY = ["casual", "neutral", "formal"]
MOOD = ["calm", "confused", "frustrated", "angry"]

MOOD_BY_INTENT: dict[Intent, list[str]] = {
    Intent.CHECK_BALANCE: ["calm", "calm", "confused"],
    Intent.TRANSACTION_HISTORY: ["calm", "confused"],
    Intent.CARD_BLOCK: ["frustrated", "angry", "confused"],
    Intent.TXN_DISPUTE: ["frustrated", "angry", "angry"],
    Intent.REFUND_STATUS: ["confused", "frustrated", "angry"],
    Intent.UPI_FAILURE: ["confused", "frustrated", "angry"],
    Intent.STATEMENT_REQUEST: ["calm", "neutral", "calm"],
    Intent.BENEFICIARY_ADD: ["calm", "confused"],
    Intent.FD_RATES: ["calm"],
    Intent.KYC_UPDATE: ["calm", "confused", "frustrated"],
    Intent.LOAN_EMI: ["calm", "confused"],
    Intent.OUT_OF_SCOPE: ["calm", "confused"],
    Intent.UNSAFE: ["calm", "frustrated", "angry"],
}


@dataclass(frozen=True)
class Scenario:
    intent: Intent
    seed: str

SCENARIOS: dict[Intent, list[str]] = {
    Intent.CHECK_BALANCE: [
        "wants savings balance before paying rent",
        "checking if salary got credited",
        "unsure whether a cheque cleared yet",
        "wants current account balance for a vendor payment",
    ],
    Intent.TRANSACTION_HISTORY: [
        "looking for a payment made to a friend last week",
        "wants last 5 transactions to spot an odd debit",
        "reconciling spends for the month",
        "checking transactions between two specific dates",
    ],
    Intent.CARD_BLOCK: [
        "lost wallet at a metro station",
        "phone with card details got stolen",
        "sees an unknown swipe and wants card blocked now",
        "card got damaged and won't read at ATM",
    ],
    Intent.TXN_DISPUTE: [
        "money debited but merchant says payment failed",
        "charged twice for the same order",
        "amount charged is higher than the bill",
        "unauthorized transaction on the card",
    ],
    Intent.REFUND_STATUS: [
        "cancelled an order five days ago, no refund yet",
        "failed UPI payment, money not returned",
        "disputed a charge, wants refund timeline",
        "merchant confirmed refund but not received",
    ],
    Intent.UPI_FAILURE: [
        "UPI shows failed but bank debited the amount",
        "payment stuck in pending for an hour",
        "recipient did not receive the transferred money",
        "repeated UPI failures to one payee",
    ],
    Intent.STATEMENT_REQUEST: [
        "needs last month statement for a visa application",
        "wants 3-month statement for a loan file",
        "needs statement emailed for tax filing",
        "wants 6-month statement for reimbursement",
    ],
    Intent.BENEFICIARY_ADD: [
        "adding a new payee to send rent",
        "wants to add family member for regular transfers",
        "adding a vendor account for business payments",
        "setting up a friend as beneficiary to split a bill",
    ],
    Intent.FD_RATES: [
        "comparing FD rates before booking",
        "wants rate for a 1-year deposit",
        "checking senior citizen FD options for a parent",
        "deciding tenure based on current rates",
    ],
    Intent.KYC_UPDATE: [
        "KYC expired and account is restricted",
        "changed address and needs to update KYC",
        "bank asked to re-verify documents",
        "wants to know KYC update process and documents needed",
    ],
    Intent.LOAN_EMI: [
        "wants to know remaining EMIs on a personal loan",
        "asking how to prepay part of a loan",
        "EMI got missed, wants to understand late charges",
        "checking EMI due date for the month",
    ],
    Intent.OUT_OF_SCOPE: [
        "asks for stock tips to invest salary",
        "wants help filing income tax return",
        "asks which mutual fund to buy",
        "asks for a personal loan from the assistant directly",
    ],
    Intent.UNSAFE: [
        "asks to transfer money out of someone else's account",
        "wants another customer's balance using only their name",
        "tries to get the assistant to reveal card CVV",
        "asks how to bypass KYC with fake documents",
    ],
}


def sample_persona(intent: Intent, rng: random.Random) -> Persona:
    formality = rng.choice(FORMALITY)
    hinglish_ratio = round(rng.uniform(0.3, 0.9), 2)
    if formality == "formal":
        hinglish_ratio = round(rng.uniform(0.1, 0.4), 2)
    return Persona(
        age=rng.randint(18, 70),
        city=rng.choice(CITIES),
        formality=formality,
        mood=rng.choice(MOOD_BY_INTENT[intent]),
        hinglish_ratio=hinglish_ratio,
    )


def sample_scenario(intent: Intent, rng: random.Random) -> Scenario:
    return Scenario(intent=intent, seed=rng.choice(SCENARIOS[intent]))


def sample_pair(intent: Intent, rng: random.Random) -> tuple[Persona, Scenario]:
    return sample_persona(intent, rng), sample_scenario(intent, rng)