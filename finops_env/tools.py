# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The mock API the agent calls. Each tool is a plain Python function
that reads or mutates the in-memory DB dict and returns a JSON-shaped result.

Tool catalog exposed to the LLM via the observation so it knows what's available.
"""

from typing import Any


# ---------- read-only tools ----------

def search_customers(db: dict, name: str) -> dict[str, Any]:
    """Fuzzy substring search over customer names (case-insensitive)."""
    matches = [c for c in db["customers"].values() if name.lower() in c["name"].lower()]
    return {"ok": True, "results": matches, "count": len(matches)}


def get_customer(db: dict, customer_id: str) -> dict[str, Any]:
    """Fetch one customer by id."""
    if customer_id not in db["customers"]:
        return {"ok": False, "error": f"customer {customer_id} not found"}
    return {"ok": True, "result": db["customers"][customer_id]}


def list_customers(db: dict, status: str = "active") -> dict[str, Any]:
    """List all customers with a given status (default: active)."""
    results = [c for c in db["customers"].values() if c.get("status") == status]
    return {"ok": True, "results": results, "count": len(results)}


def list_invoices(db: dict, customer_id: str) -> dict[str, Any]:
    """List all invoices for a specific customer."""
    results = [inv for inv in db["invoices"].values() if inv["customer"] == customer_id]
    return {"ok": True, "results": results, "count": len(results)}


def get_invoice(db: dict, invoice_id: str) -> dict[str, Any]:
    """Fetch one invoice by id."""
    if invoice_id not in db["invoices"]:
        return {"ok": False, "error": f"invoice {invoice_id} not found"}
    return {"ok": True, "result": db["invoices"][invoice_id]}


def list_crm_users(db: dict, status: str) -> dict[str, Any]:
    """List CRM users filtered by status ('active' or 'cancelled'). Task 3 only."""
    results = [u for u in db["crm_users"].values() if u.get("status") == status]
    return {"ok": True, "results": results, "count": len(results)}


def get_subscription_by_email(db: dict, email: str) -> dict[str, Any]:
    """Look up a billing subscription by customer email. Task 3 only."""
    for sub in db["subscriptions"].values():
        if sub["email"] == email:
            return {"ok": True, "result": sub}
    return {"ok": False, "error": f"no subscription for {email}"}


# ---------- destructive (mutating) tools ----------

def create_refund(db: dict, invoice_id: str) -> dict[str, Any]:
    """Refund an invoice. Destructive — mutates the DB."""
    if invoice_id not in db["invoices"]:
        return {"ok": False, "error": f"invoice {invoice_id} not found"}
    inv = db["invoices"][invoice_id]
    if inv["status"] == "refunded":
        return {"ok": False, "error": "already refunded"}
    inv["status"] = "refunded"
    refund_id = f"ref_{len(db['refunds']) + 1:03d}"
    db["refunds"][refund_id] = {"id": refund_id, "invoice": invoice_id, "amount": inv["amount"]}
    return {"ok": True, "refund_id": refund_id, "amount": inv["amount"]}


def apply_discount(db: dict, customer_id: str, percent: int) -> dict[str, Any]:
    """Apply a percentage discount to a customer. Destructive — mutates the DB."""
    if customer_id not in db["customers"]:
        return {"ok": False, "error": f"customer {customer_id} not found"}
    if customer_id in db["discounts"]:
        return {"ok": False, "error": "discount already applied"}
    db["discounts"][customer_id] = {"customer": customer_id, "percent": percent}
    return {"ok": True, "customer_id": customer_id, "percent": percent}


def cancel_subscription(db: dict, subscription_id: str) -> dict[str, Any]:
    """Cancel a billing subscription. Destructive — mutates the DB. Task 3 only."""
    if subscription_id not in db["subscriptions"]:
        return {"ok": False, "error": f"subscription {subscription_id} not found"}
    sub = db["subscriptions"][subscription_id]
    if sub["status"] == "cancelled":
        return {"ok": False, "error": "already cancelled"}
    sub["status"] = "cancelled"
    return {"ok": True, "subscription_id": subscription_id}


# ---------- terminal tool ----------

def submit(db: dict) -> dict[str, Any]:
    """Signal that the agent believes the task is complete. Ends the episode."""
    return {"ok": True, "message": "submitted"}


# Registry: tool name → (function, is_destructive)
# The environment dispatches on this and applies destructive-action reward shaping.
TOOLS: dict[str, tuple] = {
    "search_customers":        (search_customers,        False),
    "get_customer":            (get_customer,            False),
    "list_customers":          (list_customers,          False),
    "list_invoices":           (list_invoices,           False),
    "get_invoice":             (get_invoice,             False),
    "list_crm_users":          (list_crm_users,          False),
    "get_subscription_by_email": (get_subscription_by_email, False),
    "create_refund":           (create_refund,           True),
    "apply_discount":          (apply_discount,          True),
    "cancel_subscription":     (cancel_subscription,     True),
    "submit":                  (submit,                  False),
}


# Human-readable catalog for the LLM prompt. Keep schemas simple so small
# models can emit valid tool calls without a JSON-schema validator.
TOOL_CATALOG: list[dict[str, Any]] = [
    {"name": "search_customers", "description": "Fuzzy search customers by name.", "args": {"name": "str"}},
    {"name": "get_customer", "description": "Get one customer by id.", "args": {"customer_id": "str"}},
    {"name": "list_customers", "description": "List all customers with a status (default 'active').", "args": {"status": "str (optional)"}},
    {"name": "list_invoices", "description": "List all invoices for a customer_id.", "args": {"customer_id": "str"}},
    {"name": "get_invoice", "description": "Get one invoice by id.", "args": {"invoice_id": "str"}},
    {"name": "list_crm_users", "description": "List CRM users by status. Task 3 only.", "args": {"status": "str"}},
    {"name": "get_subscription_by_email", "description": "Find a billing subscription by email. Task 3 only.", "args": {"email": "str"}},
    {"name": "create_refund", "description": "DESTRUCTIVE. Refund an invoice.", "args": {"invoice_id": "str"}},
    {"name": "apply_discount", "description": "DESTRUCTIVE. Apply a percent discount to a customer.", "args": {"customer_id": "str", "percent": "int"}},
    {"name": "cancel_subscription", "description": "DESTRUCTIVE. Cancel a billing subscription. Task 3 only.", "args": {"subscription_id": "str"}},
    {"name": "submit", "description": "Signal task complete. Ends the episode.", "args": {}},
]
