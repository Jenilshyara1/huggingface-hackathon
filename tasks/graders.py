# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task graders. Each grader takes the final DB state (and
optionally the action history) and returns a score strictly in (0.0, 1.0).

NO LLM JUDGES. Pure Python set/dict comparison. Same trajectory → same score,
every time. This is a hard hackathon requirement.

All grader outputs are squished into the open interval (0, 1) via a
monotone linear transform: `_squish(raw) = SCORE_FLOOR + raw * (SCORE_CEIL - SCORE_FLOOR)`.
This keeps the ordering intact (so per-step destructive-regression detection
still fires correctly) while guaranteeing no task ever reports exactly 0.0 or 1.0,
which the hackathon's deep validator requires.
"""

from typing import Iterable


# ---------- score range ----------

SCORE_FLOOR = 0.02   # minimum reportable score (strictly > 0)
SCORE_CEIL = 0.98    # maximum reportable score (strictly < 1)


def _squish(raw: float) -> float:
    """Monotone map [0.0, 1.0] → [SCORE_FLOOR, SCORE_CEIL] ⊂ (0, 1)."""
    raw = max(0.0, min(1.0, raw))
    return SCORE_FLOOR + raw * (SCORE_CEIL - SCORE_FLOOR)


# ---------- helpers ----------

def _f1(predicted: set, golden: set) -> float:
    """Standard precision/recall F1 over two sets."""
    if not predicted and not golden:
        return 1.0
    if not predicted or not golden:
        return 0.0
    tp = len(predicted & golden)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(golden)
    return 2 * precision * recall / (precision + recall)


def _weighted_f1(predicted: set, golden: set, wrong_weight: float = 2.0) -> float:
    """F1 variant where false positives cost `wrong_weight`× more than false negatives.

    Used for destructive-action tasks where doing the wrong thing is worse than
    doing nothing. Returns a raw score in [0.0, 1.0]; callers apply `_squish`.
    """
    tp = len(predicted & golden)
    fp = len(predicted - golden)
    fn = len(golden - predicted)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    # weighted cost model: each fp costs `wrong_weight`, each fn costs 1
    cost = wrong_weight * fp + fn
    max_cost = wrong_weight * (tp + fp) + (tp + fn)  # worst-case denominator
    if max_cost == 0:
        return 1.0
    score = 1.0 - (cost / max_cost)
    return max(0.0, min(1.0, score))


# ---------- task graders ----------

def grade_task1(db: dict) -> tuple[float, dict]:
    """Task 1: refund exactly ONE of TechCorp's duplicate March invoices.

    Full credit (1.0): exactly one of {inv_901, inv_902} refunded, others untouched.
    Partial (0.5):     both TechCorp March invoices refunded (over-refunded).
    Partial (0.3):     correct customer's invoice refunded but it was a different month.
    Zero (0.0):        wrong customer refunded, or nothing refunded.
    """
    inv = db["invoices"]
    tc_march = {"inv_901", "inv_902"}
    refunded = {iid for iid, rec in inv.items() if rec["status"] == "refunded"}
    other_refunded = refunded - tc_march
    tc_march_refunded = refunded & tc_march

    if other_refunded:
        # touched someone else's money — destructive failure
        return _squish(0.0), {"reason": "refunded wrong invoice", "refunded": list(refunded)}

    if len(tc_march_refunded) == 1:
        return _squish(1.0), {"reason": "correct single refund", "refunded": list(refunded)}
    if len(tc_march_refunded) == 2:
        return _squish(0.5), {"reason": "over-refunded (both duplicates)", "refunded": list(refunded)}
    return _squish(0.0), {"reason": "no refund issued", "refunded": list(refunded)}


def grade_task2(db: dict) -> tuple[float, dict]:
    """Task 2: apply 10% discount to exactly the set of ACTIVE customers whose
    lifetime spend STRICTLY exceeds $500.

    Golden set (6 customers):
      cus_001 ($900), cus_003 ($1200), cus_005 ($750),
      cus_007 ($2000), cus_012 ($1500), cus_018 ($850)

    Notable non-qualifiers (traps):
      - cus_010: exactly $500.00 ("exceeds" → strictly >, so excluded)
      - cus_008: $450 (near-boundary distractor)
      - cus_021 ($900 inactive), cus_022 ($700 inactive) — above threshold but inactive

    Uses plain F1 over the discounted-customer set against the golden set.
    Also verifies the discount amount is 10% (penalty if wrong percent).
    """
    golden = {"cus_001", "cus_003", "cus_005", "cus_007", "cus_012", "cus_018"}
    predicted = set(db["discounts"].keys())
    score = _f1(predicted, golden)

    # percent-correctness modifier: if any discount is not 10%, halve the score
    wrong_percents = [d for d in db["discounts"].values() if d.get("percent") != 10]
    if wrong_percents:
        score *= 0.5

    return _squish(score), {
        "golden": sorted(golden),
        "predicted": sorted(predicted),
        "wrong_percents": len(wrong_percents),
    }


def grade_task3(db: dict) -> tuple[float, dict]:
    """Task 3: cancel exactly the 4 billing subs whose CRM user is 'cancelled'
    but whose billing subscription was still 'active' at episode start.

    Golden set: {sub_202, sub_204, sub_207, sub_211}.

    Traps (all 3 were already 'cancelled' in billing at reset — the agent must
    NOT attempt to cancel them again): {sub_205, sub_209, sub_213}.
    These are stripped from the predicted set so a no-op on them doesn't
    inflate precision, but re-cancelling one is a runtime error via
    cancel_subscription's "already cancelled" guard, which still costs a
    step and an invalid-action penalty.

    Fourth trap: crm_o (olga@ex.com) is CRM-cancelled but has no matching
    billing subscription at all — a dead-end lookup. The grader doesn't need
    to special-case it because the agent can't cancel a sub that doesn't exist.

    Uses weighted F1 where false positives cost 2× more than false negatives.
    """
    golden = {"sub_202", "sub_204", "sub_207", "sub_211"}
    # find subs that are currently cancelled
    predicted = {sid for sid, rec in db["subscriptions"].items() if rec["status"] == "cancelled"}
    # subtract the pre-existing cancelled subs so they don't inflate either set
    already_cancelled_at_start = {"sub_205", "sub_209", "sub_213"}
    predicted = predicted - already_cancelled_at_start

    score = _weighted_f1(predicted, golden, wrong_weight=2.0)
    return _squish(score), {
        "golden": sorted(golden),
        "predicted": sorted(predicted),
        "traps_avoided": sorted(already_cancelled_at_start),
    }


# Registry: task_id → grader fn
GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
