"""The environment — implements reset() and step(). This is the heart of OpenEnv.

In the real submission this logic gets wrapped in the OpenEnv server scaffold
(FastAPI + Docker), but the core logic lives here unchanged.
"""
from models import Action, Observation, StepResult
from database import make_task1_db
from tools import TOOLS


TASK1_TICKET = (
    "Customer 'TechCorp Inc' emailed saying they were double-charged "
    "for their March invoice. Please refund the duplicate charge."
)


class FinOpsEnvironment:
    def __init__(self):
        self.db: dict = {}
        self.ticket: str = ""
        self.steps: int = 0
        self.max_steps: int = 10
        self.done: bool = False
        self.last_response: dict = {}

    def reset(self) -> Observation:
        """Start a fresh Task 1 episode."""
        self.db = make_task1_db()
        self.ticket = TASK1_TICKET
        self.steps = 0
        self.done = False
        self.last_response = {"msg": "episode started"}
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Execute one tool call, return (observation, reward, done)."""
        self.steps += 1
        reward = 0.0

        # dispatch the tool call
        if action.tool not in TOOLS:
            self.last_response = {"ok": False, "error": f"unknown tool '{action.tool}'"}
            reward -= 0.1  # small penalty for invalid action
        else:
            try:
                self.last_response = TOOLS[action.tool](self.db, **action.args)
            except TypeError as e:
                self.last_response = {"ok": False, "error": f"bad args: {e}"}
                reward -= 0.1

        # shaped reward: reaching terminal success gives +1
        if self._task1_success():
            reward += 1.0
            self.done = True

        # budget exhaustion
        if self.steps >= self.max_steps:
            self.done = True

        # tiny per-step cost to discourage wandering
        reward -= 0.01

        return StepResult(observation=self._observe(), reward=reward, done=self.done)

    def _task1_success(self) -> bool:
        """Task 1 grader: did the agent refund invoice inv_902 (the duplicate)
        AND leave inv_901 (the legitimate charge) alone?"""
        inv = self.db["invoices"]
        return (
            inv["inv_902"]["status"] == "refunded"
            and inv["inv_901"]["status"] == "paid"
        )

    def _observe(self) -> Observation:
        return Observation(
            ticket=self.ticket,
            last_response=self.last_response,
            available_tools=list(TOOLS.keys()),
            steps_taken=self.steps,
            max_steps=self.max_steps,
            done=self.done,
        )
