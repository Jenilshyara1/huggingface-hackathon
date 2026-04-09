# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task configuration — ticket text, seed DB factory, grader, budget.

Tasks are the hackathon unit of difficulty. The environment can `reset(task_id=...)`
into any of these; the inference script runs all three.
"""

from dataclasses import dataclass
from typing import Callable

try:
    from .database import make_task1_db, make_task2_db, make_task3_db
    from .graders import grade_task1, grade_task2, grade_task3
except (ModuleNotFoundError, ImportError):
    from tasks.database import make_task1_db, make_task2_db, make_task3_db
    from tasks.graders import grade_task1, grade_task2, grade_task3


@dataclass
class TaskConfig:
    task_id: str
    difficulty: str
    ticket: str
    max_steps: int
    seed_fn: Callable[[], dict]
    grader_fn: Callable[[dict], tuple[float, dict]]


TASKS: dict[str, TaskConfig] = {
    "task1": TaskConfig(
        task_id="task1",
        difficulty="easy",
        ticket=(
            "Customer 'TechCorp Inc' emailed saying they were double-charged "
            "for their March invoice. Please find the duplicate charge and "
            "refund exactly one of the two identical March invoices. Do not "
            "touch any other customer's invoices — note that a few other "
            "customers also have March charges, and some even have their own "
            "unrelated duplicates; you must refund ONLY TechCorp's duplicate. "
            "Call `submit` when done."
        ),
        max_steps=10,
        seed_fn=make_task1_db,
        grader_fn=grade_task1,
    ),
    "task2": TaskConfig(
        task_id="task2",
        difficulty="medium",
        ticket=(
            "Apply a 10% loyalty discount to every ACTIVE customer whose "
            "lifetime spend (sum of all their invoices) EXCEEDS $500 "
            "(strictly greater than — a customer with exactly $500 does NOT "
            "qualify). Start with `list_customers(status=\"active\")`, then "
            "inspect each active customer's invoices. Do NOT discount "
            "customers below or at the threshold, and do NOT discount "
            "inactive customers even if their spend is high. "
            "Call `submit` when done."
        ),
        max_steps=20,
        seed_fn=make_task2_db,
        grader_fn=grade_task2,
    ),
    "task3": TaskConfig(
        task_id="task3",
        difficulty="hard",
        ticket=(
            "Cross-reference our CRM with our Billing system. For any user "
            "whose CRM status is 'cancelled' but whose billing subscription "
            "is still 'active', cancel the billing subscription. Do NOT "
            "cancel subscriptions that are already cancelled (that counts "
            "as a wrong destructive action). Some cancelled CRM users may "
            "have no matching billing subscription at all — that's a "
            "dead-end, not a target; just move on. Call `submit` when done."
        ),
        max_steps=20,
        seed_fn=make_task3_db,
        grader_fn=grade_task3,
    ),
}

TASK_ORDER = ["task1", "task2", "task3"]
