#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline inference script for the FinOps Support Automation OpenEnv.

Runs an LLM agent (via an OpenAI-compatible API) against all 3 tasks and
prints the structured [START]/[STEP]/[END] logs required by the hackathon.

Environment variables (REQUIRED by the hackathon spec):
    API_BASE_URL   — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — API key (HuggingFace / OpenAI / etc.)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Make modules importable when running from project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI  # noqa: E402

from models import FinopsAction  # noqa: E402
from server.finops_env_environment import FinopsEnvironment  # noqa: E402
from tasks import TASK_ORDER, TASKS  # noqa: E402


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN")
# Optional — only used if you switch to from_docker_image()
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

MAX_RETRIES_PER_STEP = 2           # LLM parse-failure retries before we count it as an invalid action
REQUEST_TIMEOUT_SECONDS = 60       # per-request timeout for safety


# ----------------------------------------------------------------------------
# prompting
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful FinOps support automation agent. You resolve customer support tickets by calling tools against a mock billing and CRM API.

CRITICAL RULES:
- On every turn, output ONE JSON object with exactly two keys: "tool" and "args".
- "tool" MUST be copied EXACTLY from the AVAILABLE TOOLS list. Do NOT invent new tool names. Tools like "sum_invoices", "calculate_total", "compute_spend" DO NOT EXIST. If you need to sum or compute something, do the arithmetic yourself from the data the tools return.
- "args" is an object of keyword arguments for the tool (use {} if none).
- Output ONLY the JSON object. No prose, no markdown fences, no explanation.
- Destructive tools (create_refund, apply_discount, cancel_subscription) are irreversible and penalized heavily when wrong. Only call them when you are SURE of the target.
- When you believe the task is complete, call {"tool": "submit", "args": {}} to end the episode.
- If the previous TOOL RESULT contains an error, do NOT repeat the same call. Try a different tool or different args.

Strategy: read the ticket → plan the minimum sequence of calls using ONLY the listed tools → execute → when the underlying task is achieved, call submit."""


def build_user_prompt(obs) -> str:
    """Render the current observation into a compact user message."""
    tools_block = "\n".join(
        f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['args'].items())}) — {t['description']}"
        for t in obs.available_tools
    )
    return (
        f"TICKET:\n{obs.ticket}\n\n"
        f"AVAILABLE TOOLS:\n{tools_block}\n\n"
        f"STEP {obs.steps_taken}/{obs.max_steps}\n"
        f"LAST TOOL RESPONSE:\n{json.dumps(obs.last_response, indent=2)}\n\n"
        f"Output the next action as a single JSON object: "
        f'{{"tool": "<tool_name>", "args": {{...}}}}'
    )


def parse_action(text: str) -> FinopsAction | None:
    """Extract a FinopsAction from the LLM's response text.

    Handles the common failure modes: surrounding whitespace, markdown fences,
    leading prose before the JSON object.
    """
    if not text:
        return None
    text = text.strip()

    # strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # find the first '{' and parse from there
    start = text.find("{")
    if start == -1:
        return None
    # try progressively shorter suffixes until one parses (handles trailing prose)
    for end in range(len(text), start, -1):
        try:
            obj = json.loads(text[start:end])
            if isinstance(obj, dict) and "tool" in obj:
                return FinopsAction(tool=str(obj["tool"]), args=obj.get("args") or {})
        except json.JSONDecodeError:
            continue
    return None


# ----------------------------------------------------------------------------
# agent loop
# ----------------------------------------------------------------------------

def run_task(env: FinopsEnvironment, client: OpenAI, task_id: str) -> dict[str, Any]:
    """Run one task to completion. Returns {task_id, score, steps, elapsed_sec}."""
    start_wall = time.time()
    print(f"[START] task={task_id} model={MODEL_NAME}", flush=True)

    obs = env.reset(task_id=task_id)
    transcript: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_steps = 0
    total_reward = 0.0
    final_score = 0.0

    while not obs.done and obs.steps_taken < obs.max_steps:
        user_msg = build_user_prompt(obs)
        transcript.append({"role": "user", "content": user_msg})

        # call the LLM, retry on parse failures
        action: FinopsAction | None = None
        llm_text = ""
        for attempt in range(MAX_RETRIES_PER_STEP):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=transcript,
                    temperature=0.0,
                    max_tokens=256,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                llm_text = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                llm_text = ""
                print(f"[STEP] task={task_id} step={total_steps + 1} error=llm_call_failed: {e}", flush=True)

            action = parse_action(llm_text)
            if action is not None:
                break
            # nudge the model on retry
            transcript.append({"role": "assistant", "content": llm_text})
            transcript.append(
                {
                    "role": "user",
                    "content": 'Your previous response was not valid JSON. Output ONLY a single JSON object like {"tool": "search_customers", "args": {"name": "TechCorp"}}.',
                }
            )

        if action is None:
            # give up on this step — treat as invalid action
            action = FinopsAction(tool="__invalid__", args={})
            print(f"[STEP] task={task_id} step={total_steps + 1} tool=__parse_failed__", flush=True)
        else:
            transcript.append({"role": "assistant", "content": llm_text})

        # execute the action in the env
        obs = env.step(action)
        total_steps += 1
        total_reward += float(obs.reward or 0.0)
        print(
            f"[STEP] task={task_id} step={total_steps} tool={action.tool} "
            f"reward={obs.reward:+.3f} done={obs.done}",
            flush=True,
        )

        # attach the tool result to the transcript so the LLM sees it next turn
        transcript.append(
            {
                "role": "user",
                "content": f"TOOL RESULT: {json.dumps(obs.last_response)}",
            }
        )

        if obs.done:
            # final grader result is exposed in last_response when the episode ends
            final_score = float(obs.last_response.get("final_score") or 0.0)
            break

    elapsed = time.time() - start_wall
    print(
        f"[END] task={task_id} score={final_score:.3f} "
        f"cumulative_reward={total_reward:+.3f} steps={total_steps} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "score": final_score,
        "cumulative_reward": total_reward,
        "steps": total_steps,
        "elapsed_sec": elapsed,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> int:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (API key) must be set via environment variable.", file=sys.stderr)
        return 2

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = FinopsEnvironment()

    all_results = []
    overall_start = time.time()
    for task_id in TASK_ORDER:
        result = run_task(env, client, task_id)
        all_results.append(result)

    overall_elapsed = time.time() - overall_start
    print("\n=== BASELINE SUMMARY ===", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Total runtime: {overall_elapsed:.1f}s", flush=True)
    for r in all_results:
        cfg = TASKS[r["task_id"]]
        print(
            f"  {r['task_id']} ({cfg.difficulty}): score={r['score']:.3f}  "
            f"steps={r['steps']}  elapsed={r['elapsed_sec']:.1f}s",
            flush=True,
        )
    mean = sum(r["score"] for r in all_results) / len(all_results)
    print(f"Mean score: {mean:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
