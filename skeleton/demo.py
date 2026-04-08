"""A hand-written 'agent' that solves Task 1 perfectly.

This stands in for an LLM so you can see the full loop work locally
WITHOUT an API key. In the real inference.py, these hardcoded actions
would be replaced by parsed LLM outputs.
"""
from environment import FinOpsEnvironment
from models import Action


def run_scripted_agent():
    env = FinOpsEnvironment()
    obs = env.reset()

    print(f"TICKET: {obs.ticket}\n")
    print(f"AVAILABLE TOOLS: {obs.available_tools}\n")

    # Step 1: find TechCorp
    actions = [
        Action(tool="search_customers", args={"name": "TechCorp"}),
        Action(tool="list_invoices", args={"customer_id": "cus_001"}),
        Action(tool="create_refund", args={"invoice_id": "inv_902"}),
    ]

    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        total_reward += result.reward
        print(f"STEP {i}: {action.tool}({action.args})")
        print(f"  → response: {result.observation.last_response}")
        print(f"  → reward: {result.reward:+.2f} (cumulative: {total_reward:+.2f})")
        print(f"  → done: {result.done}\n")
        if result.done:
            break

    print(f"FINAL SCORE: {total_reward:+.2f}")
    print(f"Task succeeded: {env._task1_success()}")


if __name__ == "__main__":
    run_scripted_agent()
