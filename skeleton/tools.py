"""The 'API' the agent uses. Each tool is a plain Python function
that reads or writes the in-memory DB dict and returns a JSON-shaped result.

In the real env we'll have ~10 tools. Here we show 3 so you can see the shape.
"""
from typing import Any


def search_customers(db: dict, name: str) -> dict[str, Any]:
    """Find customers whose name contains `name` (case-insensitive)."""
    matches = [c for c in db["customers"].values() if name.lower() in c["name"].lower()]
    return {"ok": True, "results": matches, "count": len(matches)}


def list_invoices(db: dict, customer_id: str) -> dict[str, Any]:
    """List all invoices for a customer."""
    invoices = [inv for inv in db["invoices"].values() if inv["customer"] == customer_id]
    return {"ok": True, "results": invoices, "count": len(invoices)}


def create_refund(db: dict, invoice_id: str) -> dict[str, Any]:
    """Issue a refund for an invoice. Mutates the DB."""
    if invoice_id not in db["invoices"]:
        return {"ok": False, "error": f"invoice {invoice_id} not found"}
    inv = db["invoices"][invoice_id]
    if inv["status"] == "refunded":
        return {"ok": False, "error": "already refunded"}
    inv["status"] = "refunded"
    refund_id = f"ref_{len(db['refunds']) + 1:03d}"
    db["refunds"][refund_id] = {"id": refund_id, "invoice": invoice_id, "amount": inv["amount"]}
    return {"ok": True, "refund_id": refund_id, "amount": inv["amount"]}


# Registry: maps tool name → function. The Environment dispatches on this.
TOOLS = {
    "search_customers": search_customers,
    "list_invoices": list_invoices,
    "create_refund": create_refund,
}
