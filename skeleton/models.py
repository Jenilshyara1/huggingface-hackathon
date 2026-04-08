"""Data models for the FinOps environment.

These are the shapes of data flowing between agent and environment.
In the real OpenEnv scaffold these will be @dataclass (not raw dicts).
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """What the agent sees on each turn."""
    ticket: str                       # the customer support ticket text
    last_response: dict[str, Any]     # JSON returned by the last tool call
    available_tools: list[str]        # tool names the agent can use
    steps_taken: int                  # how many actions so far
    max_steps: int                    # budget
    done: bool                        # episode over?


@dataclass
class Action:
    """What the agent submits each turn."""
    tool: str                         # e.g. "search_customers"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """What step() returns."""
    observation: Observation
    reward: float
    done: bool
