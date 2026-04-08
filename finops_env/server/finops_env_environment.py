# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FinOps Support Automation Environment Implementation.

An LLM agent receives a natural-language customer support ticket and must
resolve it by calling mock billing/CRM tools. Rewards are shaped per-step
(dense signal) and a deterministic grader produces the final task score.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FinopsAction, FinopsObservation
    from ..tools import TOOLS, TOOL_CATALOG
    from ..tasks import TASKS, TASK_ORDER
except (ModuleNotFoundError, ImportError):
    from models import FinopsAction, FinopsObservation
    from tools import TOOLS, TOOL_CATALOG
    from tasks import TASKS, TASK_ORDER


# --- reward shaping constants ------------------------------------------------
STEP_COST = -0.01                # tiny per-step penalty → encourages efficiency
INVALID_ACTION_PENALTY = -0.1    # unknown tool or bad args
DESTRUCTIVE_WRONG_PENALTY = -0.5 # destructive call that hurt the grader's score
SUBGOAL_BONUS = 0.1              # reserved for sub-goal checkpoints (currently unused)
TERMINAL_SCALE = 1.0             # final grader score is multiplied by this


class FinopsEnvironment(Environment):
    """OpenEnv implementation of the FinOps support environment.

    State machine per episode:
      1. reset(task_id?) seeds a fresh DB and picks the task's ticket + budget.
      2. step(action) dispatches the tool call, mutates the DB, returns a
         shaped reward, and terminates when `submit` is called or max_steps
         is reached.
      3. On termination, the task's grader is called on the final DB state
         and its score (scaled by TERMINAL_SCALE) is added to the reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        # per-episode mutable state
        self._db: dict = {}
        self._task_id: str = "task1"
        self._ticket: str = ""
        self._max_steps: int = 0
        self._steps: int = 0
        self._done: bool = False
        self._last_response: dict = {"message": "call reset() to begin"}
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> FinopsObservation:  # type: ignore[override]
        """Start a new episode. `task_id` cycles through task1→task2→task3
        if not specified, so repeated reset() calls sweep the full suite.
        """
        if task_id is None:
            # round-robin across tasks based on reset count
            task_id = TASK_ORDER[self._reset_count % len(TASK_ORDER)]

        if task_id not in TASKS:
            task_id = "task1"

        cfg = TASKS[task_id]
        self._task_id = task_id
        self._db = cfg.seed_fn()
        self._ticket = cfg.ticket
        self._max_steps = cfg.max_steps
        self._steps = 0
        self._done = False
        self._last_response = {"message": f"episode started for {task_id}"}
        self._cumulative_reward = 0.0

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return self._observe(reward=0.0)

    def step(self, action: FinopsAction) -> FinopsObservation:  # type: ignore[override]
        """Execute one tool call. Returns an observation with shaped reward."""
        if self._done:
            # already terminated — return a noop observation with 0 reward
            return self._observe(reward=0.0)

        self._steps += 1
        self._state.step_count += 1

        reward = STEP_COST

        # --- dispatch -----------------------------------------------------
        if action.tool == "submit":
            # agent declares task complete — grade now
            self._last_response = {"ok": True, "message": "submitted, grading..."}
            terminal = self._finish()
            reward += terminal
            return self._observe(reward=reward)

        if action.tool not in TOOLS:
            self._last_response = {
                "ok": False,
                "error": f"unknown tool '{action.tool}'",
                "hint": f"valid tools: {list(TOOLS.keys())}",
            }
            reward += INVALID_ACTION_PENALTY
            return self._maybe_timeout(reward)

        fn, is_destructive = TOOLS[action.tool]

        # snapshot score before a destructive call so we can detect regressions
        pre_score = None
        if is_destructive:
            try:
                pre_score, _ = TASKS[self._task_id].grader_fn(self._db)
            except Exception:
                pre_score = None

        try:
            self._last_response = fn(self._db, **action.args)
        except TypeError as e:
            self._last_response = {"ok": False, "error": f"bad args: {e}"}
            reward += INVALID_ACTION_PENALTY
            return self._maybe_timeout(reward)
        except Exception as e:
            self._last_response = {"ok": False, "error": f"tool error: {e}"}
            reward += INVALID_ACTION_PENALTY
            return self._maybe_timeout(reward)

        # destructive-wrong penalty: if a destructive call *decreased* the
        # grader score, punish it. This incentivises the LLM to avoid guessing.
        if is_destructive and pre_score is not None:
            try:
                post_score, _ = TASKS[self._task_id].grader_fn(self._db)
                if post_score < pre_score:
                    reward += DESTRUCTIVE_WRONG_PENALTY
            except Exception:
                pass

        return self._maybe_timeout(reward)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _maybe_timeout(self, reward: float) -> FinopsObservation:
        if self._steps >= self._max_steps:
            terminal = self._finish()
            reward += terminal
        return self._observe(reward=reward)

    def _finish(self) -> float:
        """Grade the final DB state and return the scaled terminal reward."""
        self._done = True
        try:
            score, info = TASKS[self._task_id].grader_fn(self._db)
        except Exception as e:
            score, info = 0.0, {"error": str(e)}
        self._last_response = {
            "ok": True,
            "graded": True,
            "task_id": self._task_id,
            "final_score": score,
            "grader_info": info,
        }
        return TERMINAL_SCALE * score

    def _observe(self, reward: float) -> FinopsObservation:
        self._cumulative_reward += reward
        return FinopsObservation(
            task_id=self._task_id,
            ticket=self._ticket,
            last_response=self._last_response,
            available_tools=TOOL_CATALOG,
            steps_taken=self._steps,
            max_steps=self._max_steps,
            done=self._done,
            reward=reward,
            metadata={
                "cumulative_reward": self._cumulative_reward,
                "task_id": self._task_id,
            },
        )

    @property
    def state(self) -> State:
        return self._state
