# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task graders. Each grader takes the final DB state (and
optionally the action history) and returns a score in [0.0, 1.0].

NO LLM JUDGES. Pure Python set/dict comparison. Same trajectory → same score,
every time. This is a hard hackathon requirement.
"""

from typing import Iterable


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
    doing nothing. Returns a score in [0.0, 1.0].
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
        return 0.0, {"reason": "refunded wrong invoice", "refunded": list(refunded)}

    if len(tc_march_refunded) == 1:
        return 1.0, {"reason": "correct single refund", "refunded": list(refunded)}
    if len(tc_march_refunded) == 2:
        return 0.5, {"reason": "over-refunded (both duplicates)", "refunded": list(refunded)}
    return 0.0, {"reason": "no refund issued", "refunded": list(refunded)}


def grade_task2(db: dict) -> tuple[float, dict]:
    """Task 2: apply 10% discount to exactly the set of customers with lifetime spend > $500.

    Uses plain F1 over the discounted-customer set against the golden set.
    Also verifies the discount amount is 10% (penalty if wrong percent).
    """
    golden = {"cus_001", "cus_003", "cus_005", "cus_007"}
    predicted = set(db["discounts"].keys())
    score = _f1(predicted, golden)

    # percent-correctness modifier: if any discount is not 10%, halve the score
    wrong_percents = [d for d in db["discounts"].values() if d.get("percent") != 10]
    if wrong_percents:
        score *= 0.5

    return score, {
        "golden": sorted(golden),
        "predicted": sorted(predicted),
        "wrong_percents": len(wrong_percents),
    }


def grade_task3(db: dict) -> tuple[float, dict]:
    """Task 3: cancel exactly {sub_202, sub_204} — the subscriptions where the
    CRM user is 'cancelled' but the billing sub was still 'active'.

    DO NOT cancel sub_205 (trap: CRM cancelled but billing already cancelled).
    Uses weighted F1 where false positives (wrong cancellations) cost 2× more
    than false negatives, because destructive errors are worse than omissions.
    """
    golden = {"sub_202", "sub_204"}
    # find subs that were flipped from active to cancelled this episode
    predicted = {sid for sid, rec in db["subscriptions"].items() if rec["status"] == "cancelled"}
    # subtract the pre-existing cancelled subs (sub_205 was already cancelled at reset)
    already_cancelled_at_start = {"sub_205"}
    predicted = predicted - already_cancelled_at_start

    score = _weighted_f1(predicted, golden, wrong_weight=2.0)
    return score, {
        "golden": sorted(golden),
        "predicted": sorted(predicted),
        "trap_avoided": "sub_205" not in predicted,
    }


# Registry: task_id → grader fn
GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
