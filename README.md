---
title: FinOps Support Automation
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - agent
  - customer-support
  - tool-use
  - finops
---

# FinOps Support Automation — OpenEnv Environment

An OpenEnv environment where an LLM agent resolves real customer support
tickets by navigating a mock billing and CRM backend. Built for the
**Meta PyTorch × HuggingFace OpenEnv Hackathon (Round 1)**.

> **Tagline**: Agents train on the same tasks a Tier-2 billing support engineer does every day — search a customer, inspect their invoices, issue a targeted refund, apply a loyalty discount, cancel a subscription — with deterministic graders and dense reward shaping.

---

## Why this environment

Most existing OpenEnv environments simulate games (Atari, Sudoku, Wordle), simple REPLs, or generic coding tasks. Real enterprise LLM value — the workloads companies actually pay for today — lives in agents that resolve support tickets, navigate internal APIs, and handle billing operations with care.

**FinOps Support Automation fills that gap.** It models tier-2 billing/CRM support as a structured tool-use task: each episode begins with a natural-language customer ticket, and the agent must resolve it by calling a fixed catalog of tools over a mock in-memory database. No browser simulation, no HTML parsing, no LLM-judge non-determinism — pure text in, tool-calls out, deterministic state diffs for grading.

### Differentiation from prior work

The closest academic benchmark is Sierra AI's **τ-bench** (retail / airline customer support). This environment differentiates along three dimensions:

1. **Multi-system correlation.** Task 3 requires joining across two independent namespaces (CRM ↔ Billing) — no single-API lookup solves it.
2. **Dense per-step reward shaping.** τ-bench is terminal 0/1. FinOps emits a shaped reward on every step (step cost, invalid-action penalty, destructive-regression penalty, terminal grader score).
3. **Destructive-action asymmetry.** Wrong refunds and wrong cancellations are penalized twice as heavily as missed actions — an RL-friendly encoding of "don't touch what you don't understand."

---

## Environment specification

### Observation space (`FinopsObservation`)

| Field              | Type                     | Description                                                       |
|--------------------|--------------------------|-------------------------------------------------------------------|
| `task_id`          | `str`                    | `task1` / `task2` / `task3`                                       |
| `ticket`           | `str`                    | Natural-language customer support request                        |
| `last_response`    | `dict[str, Any]`         | JSON result of the previous tool call (or episode-start message) |
| `available_tools`  | `list[dict]`             | Tool catalog: name, description, argument schema                  |
| `steps_taken`      | `int`                    | Actions executed so far                                           |
| `max_steps`        | `int`                    | Episode budget                                                    |
| `done`             | `bool`                   | Episode-terminated flag (inherited from base `Observation`)       |
| `reward`           | `float`                  | Per-step shaped reward, can be negative (inherited)               |
| `metadata`         | `dict[str, Any]`         | Contains `cumulative_reward` and `task_id` (inherited)            |

### Action space (`FinopsAction`)

A single tool invocation:

```json
{"tool": "search_customers", "args": {"name": "TechCorp"}}
```

`tool` must be one of the 11 names in the catalog; `args` is a JSON object of keyword arguments.

### Tool catalog

| Tool                                   | Destructive | Purpose                                             |
|----------------------------------------|:---:|-----------------------------------------------------|
| `search_customers(name)`               |     | Fuzzy search customers by name substring            |
| `get_customer(customer_id)`            |     | Fetch one customer by id                            |
| `list_customers(status="active")`      |     | List customers filtered by status                   |
| `list_invoices(customer_id)`           |     | All invoices for a given customer                   |
| `get_invoice(invoice_id)`              |     | Fetch one invoice by id                             |
| `list_crm_users(status)`               |     | List CRM-side users by status (Task 3)              |
| `get_subscription_by_email(email)`     |     | Look up billing subscription by email (Task 3)      |
| `create_refund(invoice_id)`            |  ✔  | Refund an invoice — mutates DB                      |
| `apply_discount(customer_id, percent)` |  ✔  | Apply a % discount — mutates DB                     |
| `cancel_subscription(subscription_id)` |  ✔  | Cancel a billing subscription — mutates DB          |
| `submit()`                             |     | Signal task complete; ends the episode              |

### Reward shaping

Per-step reward is the sum of:

| Event                                              | Reward           |
|---------------------------------------------------|------------------|
| Per-step time cost                                 | `-0.01`          |
| Invalid tool name / bad args                       | `-0.10`          |
| Destructive call that *decreased* the grader score | `-0.50`          |
| Terminal grader score on `submit` or timeout       | `+[0.0, 1.0]`    |

The destructive-regression penalty is computed by re-grading the DB before and after each mutating call — if the score drops, the agent gets the `-0.5` penalty in addition to the step cost. This creates a dense, informative training signal and actively discourages destructive guessing.

### State (`State`)

Standard OpenEnv `State` with `episode_id` and `step_count`. Exposed at `GET /state`.

### Endpoints

Auto-generated by `openenv.core.env_server.http_server.create_app`:

- `POST /reset` — start a new episode (round-robins through task1 → task2 → task3 if no `task_id` specified)
- `POST /step` — execute one `FinopsAction`
- `GET /state` — current `State`
- `GET /schema` — machine-readable action/observation schemas
- `WS /ws` — persistent WebSocket session for efficient multi-step episodes

---

## The 3 tasks

### Task 1 — Easy: Relational Lookup & Refund

> *"Customer 'TechCorp Inc' emailed saying they were double-charged for their March invoice. Please find the duplicate charge and refund exactly one of the two identical March invoices. Do not touch any other customer's invoices."*

**Seed DB**: 3 customers, 4 invoices. TechCorp has two identical $500 March charges (`inv_901`, `inv_902`). Acme's and Globex's invoices are red herrings.

**Optimal policy (~3 steps)**:
1. `search_customers(name="TechCorp")` → `cus_001`
2. `list_invoices(customer_id="cus_001")` → sees two identical March charges
3. `create_refund(invoice_id="inv_902")` → refund one of them
4. `submit` → end

**Grader**:
- `1.0` — exactly one of `{inv_901, inv_902}` refunded, others untouched
- `0.5` — both TechCorp March duplicates refunded (over-refunded)
- `0.0` — wrong customer refunded or no refund at all

**Budget**: `max_steps = 8`

---

### Task 2 — Medium: Aggregation & Bulk Action

> *"Apply a 10% loyalty discount to every active customer whose lifetime spend (sum of all their invoices) exceeds $500. Do NOT discount customers below the threshold."*

**Seed DB**: 8 active customers, 12 invoices. Four customers qualify (`cus_001` at $900, `cus_003` at $1200, `cus_005` at $750, `cus_007` at $2000). Four fall below ($250, $300, $100, $450) — note `cus_008` at $450 is deliberately close to the threshold as a distractor.

**Optimal policy (~10 steps)**:
1. `list_customers(status="active")` → all 8 customers
2. For each customer: `list_invoices(customer_id=...)`, sum amounts
3. For each qualifier: `apply_discount(customer_id=..., percent=10)`
4. `submit`

**Grader**: F1 score over the set of discounted `customer_id`s vs. the golden set `{cus_001, cus_003, cus_005, cus_007}`. If any applied discount is not 10%, the score is halved.

**Budget**: `max_steps = 25`

---

### Task 3 — Hard: Multi-System Correlation

> *"Cross-reference our CRM with our Billing system. For any user whose CRM status is 'cancelled' but who still has an 'active' billing subscription, cancel the billing subscription. Do NOT cancel subscriptions that are already cancelled."*

**Seed DB**: Two disjoint namespaces.
- CRM: 6 users. Three have `status=cancelled` (`crm_b`, `crm_d`, `crm_e`).
- Billing: 6 subscriptions keyed by email. Five are `active`, one (`sub_205`, eve@) is **already cancelled**.

**The trap**: `crm_e` (eve@) is cancelled in CRM, but her billing sub `sub_205` is already cancelled. Re-cancelling it (or treating her as a target) is penalized as a wrong destructive action.

**Correct target set**: `{sub_202 (bob), sub_204 (dave)}`.

**Optimal policy (~8 steps)**:
1. `list_crm_users(status="cancelled")` → `[crm_b, crm_d, crm_e]`
2. For each: `get_subscription_by_email(email=...)` → find `sub_202`, `sub_204`, `sub_205`
3. Cancel only those with current status `active` → `cancel_subscription(sub_202)`, `cancel_subscription(sub_204)`
4. `submit`

**Grader**: Weighted F1 over the set of newly-cancelled subscription IDs. False positives (wrong cancellations) cost **2×** more than false negatives (missed cancellations). Pre-existing cancellations (`sub_205`) are filtered out of the predicted set so they don't inflate precision.

**Budget**: `max_steps = 25`

---

## Baseline scores

Run with the provided `inference.py` using an OpenAI-compatible client. The full 3-task suite completes in well under the 20-minute / 2-vCPU / 8-GB hackathon runtime budget.

| Model                              | Provider            | Task 1 | Task 2 | Task 3 | Mean | Runtime |
|------------------------------------|---------------------|-------:|-------:|-------:|-----:|--------:|
| `llama-3.3-70b-versatile`          | Groq                | **1.000** | **1.000** | **1.000** | **1.000** | 338 s |
| *scripted optimal reference*       | deterministic agent | 1.000 | 1.000 | 1.000 | 1.000 | — |

### What the scores mean

- **Task 1 (easy) — 1.000**: Llama 3.3 70B solved the duplicate-refund task in 4 steps (`search_customers` → `list_invoices` → `create_refund` → `submit`), matching the optimal policy. Expected baseline for a capable instruction-following model.
- **Task 2 (medium) — 1.000**: The model correctly aggregated invoice totals for all 8 active customers, identified all 4 qualifying customers (including `cus_008` at $450 — correctly excluded as a below-threshold distractor), and applied exactly 10% discounts with zero wrong discounts. Solved in 14 steps.
- **Task 3 (hard) — 1.000**: The model correctly cross-referenced CRM ↔ Billing, identified all three CRM-cancelled users, inspected each of their billing subscriptions, cancelled only the two that were still active (`sub_202`, `sub_204`), and **avoided the trap** (`sub_205`, already cancelled). This is the strongest signal that the destructive-action asymmetry penalty is training meaningful caution.

The scripted-reference row is not an LLM score — it's generated by invoking the same `step()` code path a perfect agent would take, establishing the achievable upper bound and confirming all three graders return `1.0` on correct trajectories.

---

## Quickstart

### Run locally (no Docker)

```bash
# from project root
uv sync          # creates .venv and installs all dependencies

# start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the baseline inference script

```bash
# From project root — defaults to Groq + llama-3.3-70b-versatile
export API_BASE_URL="https://api.groq.com/openai/v1"  # any OpenAI-compatible endpoint
export MODEL_NAME="llama-3.3-70b-versatile"            # or gpt-4o-mini, Qwen2.5-7B-Instruct, etc.
export HF_TOKEN="gsk_..."                              # your API key (Groq/OpenAI/HF/etc.)

python3 inference.py
```

The script runs all 3 tasks end-to-end and emits structured `[START]` / `[STEP]` / `[END]` logs in the format required by the hackathon spec. Total runtime stays well under the 20-minute / 2-vCPU / 8-GB budget because each episode is bounded by `max_steps` (8 / 25 / 25) and the in-memory DB means tool calls are essentially free.

### Run the container

```bash
# from project root
docker build -t finops-env:latest .
docker run --rm -p 8000:8000 finops-env:latest
# curl -X POST http://localhost:8000/reset
```

### Validate spec compliance

```bash
openenv validate .
# [OK] finops_env: Ready for multi-mode deployment
```

---

## File layout

```
.
├── openenv.yaml             # Manifest (spec_version, runtime, port)
├── pyproject.toml           # Package metadata + dependencies
├── uv.lock                  # Pinned dependency graph
├── Dockerfile               # Container build for HF Spaces deployment
├── README.md                # (this file)
├── inference.py             # Baseline LLM agent loop (hackathon-required)
├── validator.bash           # Runs openenv validate against a live server
├── .env.example             # Template for required environment variables
├── __init__.py
├── models.py                # FinopsAction, FinopsObservation (Pydantic)
├── client.py                # EnvClient subclass for remote agents
├── tools.py                 # 11 tool functions + TOOL_CATALOG
├── server/
│   ├── app.py               # FastAPI app via openenv.core.http_server
│   ├── finops_env_environment.py  # FinopsEnvironment — reset/step/state
│   └── __init__.py
├── tasks/
│   ├── __init__.py          # TaskConfig dataclasses wiring tickets → seeds → graders
│   ├── database.py          # Per-task seed DBs (tasks 1–3)
│   └── graders.py           # Deterministic task graders (F1, weighted F1)
├── assets/                  # Static assets (graphs, diagrams)
└── outputs/                 # Inference run outputs
```

---

## License

Submitted under the same BSD-style license as the OpenEnv project templates.
