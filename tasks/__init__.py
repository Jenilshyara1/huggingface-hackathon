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
            "touch any other customer's invoices. Call `submit` when done."
        ),
        max_steps=8,
        seed_fn=make_task1_db,
        grader_fn=grade_task1,
    ),
    "task2": TaskConfig(
        task_id="task2",
        difficulty="medium",
        ticket=(
            "Apply a 10% loyalty discount to every active customer whose "
            "lifetime spend (sum of all their invoices) exceeds $500. "
            "Use `list_customers`, then inspect each customer's invoices. "
            "Do NOT discount customers below the threshold. Call `submit` when done."
        ),
        max_steps=25,
        seed_fn=make_task2_db,
        grader_fn=grade_task2,
    ),
    "task3": TaskConfig(
        task_id="task3",
        difficulty="hard",
        ticket=(
            "Cross-reference our CRM with our Billing system. For any user "
            "whose CRM status is 'cancelled' but who still has an 'active' "
            "billing subscription, cancel the billing subscription. Do NOT "
            "cancel subscriptions that are already cancelled (that wastes "
            "a step and counts as a wrong destructive action). "
            "Call `submit` when done."
        ),
        max_steps=25,
        seed_fn=make_task3_db,
        grader_fn=grade_task3,
    ),
}

TASK_ORDER = ["task1", "task2", "task3"]
