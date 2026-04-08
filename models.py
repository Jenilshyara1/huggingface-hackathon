# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the FinOps Support Automation Environment.

The agent receives a natural-language customer support ticket and must resolve
it by calling tools in a mock billing/CRM backend (search customers, list
invoices, issue refunds, apply discounts, cancel subscriptions).
"""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FinopsAction(Action):
    """Action = a single tool call.

    The agent picks one tool from the catalog exposed in the observation and
    supplies its arguments as a JSON-compatible dict. The environment dispatches
    on `tool` and calls the matching Python function with `**args`.
    """

    tool: str = Field(..., description="Name of the tool to invoke")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool"
    )


class FinopsObservation(Observation):
    """Observation = everything the agent sees on its turn.

    The agent is given the ticket text, the JSON result of the previous tool
    call, the catalog of available tools, and budget info. `reward` and `done`
    are inherited from the base Observation class.
    """

    task_id: str = Field(default="", description="Which task is being run: task1/task2/task3")
    ticket: str = Field(default="", description="The customer support ticket to resolve")
    last_response: dict[str, Any] = Field(
        default_factory=dict, description="JSON returned by the previous tool call"
    )
    available_tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Catalog of callable tools: name, description, and arg schema",
    )
    steps_taken: int = Field(default=0, description="Number of actions executed so far")
    max_steps: int = Field(default=0, description="Hard cap on episode length")
