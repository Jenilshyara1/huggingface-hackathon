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
    API_BASE_URL     — LLM API endpoint
    MODEL_NAME       — Model identifier
    HF_TOKEN         — Your Hugging Face / API key
    LOCAL_IMAGE_NAME — Docker image name (if using from_docker_image())

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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")
# Optional — only used if you switch to from_docker_image()
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

BENCHMARK = "finops_env"
SUCCESS_SCORE_THRESHOLD = 0.5       # score >= this → success=true in [END]
MAX_RETRIES_PER_STEP = 2            # LLM parse-failure retries before treating as invalid action
REQUEST_TIMEOUT_SECONDS = 60        # per-request timeout


# ----------------------------------------------------------------------------
# structured logging (matches the hackathon-required stdout format exactly)
# ----------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


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


def build_user_prompt(obs, include_tools: bool = True) -> str:
    """Render the current observation into a compact user message.

    `include_tools` is True only on the first step — subsequent steps omit the
    static tool catalog to keep the transcript small and avoid TPM limits.
    """
    step_line = f"STEP {obs.steps_taken}/{obs.max_steps}"
    response_line = f"LAST TOOL RESPONSE:\n{json.dumps(obs.last_response)}"
    suffix = "Output the next action as a single JSON object: {\"tool\": \"<tool_name>\", \"args\": {...}}"

    if not include_tools:
        return f"{step_line}\n{response_line}\n\n{suffix}"

    tools_block = "\n".join(
        f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['args'].items())}) — {t['description']}"
        for t in obs.available_tools
    )
    return (
        f"TICKET:\n{obs.ticket}\n\n"
        f"AVAILABLE TOOLS:\n{tools_block}\n\n"
        f"{step_line}\n{response_line}\n\n{suffix}"
    )


def parse_action(text: str) -> FinopsAction | None:
    """Extract a FinopsAction from the LLM's response text."""
    if not text:
        return None
    text = text.strip()

    # strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        return None
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
    """Run one task to completion. Returns result dict."""
    log_start(task=task_id, model=MODEL_NAME)

    obs = env.reset(task_id=task_id)
    transcript: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_steps = 0
    step_rewards: list[float] = []
    final_score = 0.0
    last_error: str | None = None

    try:
        while not obs.done and obs.steps_taken < obs.max_steps:
            user_msg = build_user_prompt(obs, include_tools=(total_steps == 0))
            transcript.append({"role": "user", "content": user_msg})

            # call the LLM, retry on parse failures
            action: FinopsAction | None = None
            llm_text = ""
            last_error = None

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
                    last_error = None
                except Exception as e:
                    llm_text = ""
                    last_error = f"llm_call_failed:{e}"

                action = parse_action(llm_text)
                if action is not None:
                    break
                if last_error is None:
                    last_error = "parse_failed"
                transcript.append({"role": "assistant", "content": llm_text})
                transcript.append({
                    "role": "user",
                    "content": 'Your previous response was not valid JSON. Output ONLY a single JSON object like {"tool": "search_customers", "args": {"name": "TechCorp"}}.',
                })

            if action is None:
                action = FinopsAction(tool="__invalid__", args={})

            transcript.append({"role": "assistant", "content": llm_text})

            # execute in env
            obs = env.step(action)
            total_steps += 1
            reward = float(obs.reward or 0.0)
            step_rewards.append(reward)

            # extract error from tool response if any
            step_error = last_error
            if step_error is None and not obs.last_response.get("ok", True):
                step_error = obs.last_response.get("error")

            log_step(
                step=total_steps,
                action=action.tool,
                reward=reward,
                done=obs.done,
                error=step_error,
            )

            transcript.append({
                "role": "user",
                "content": f"TOOL RESULT: {json.dumps(obs.last_response)}",
            })

            if obs.done:
                final_score = float(obs.last_response.get("final_score") or 0.0)
                break

    finally:
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=total_steps, score=final_score, rewards=step_rewards)

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": total_steps,
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
            f"  {r['task_id']} ({cfg.difficulty}): score={r['score']:.2f}  steps={r['steps']}",
            flush=True,
        )
    mean = sum(r["score"] for r in all_results) / len(all_results)
    print(f"Mean score: {mean:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
