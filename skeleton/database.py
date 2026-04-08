"""The 'backend' — just a Python dict pretending to be a database.

Each task gets its own seeded DB via make_task1_db(), make_task2_db(), etc.
"""
from copy import deepcopy


def make_task1_db() -> dict:
    """Task 1 seed: TechCorp has two identical March invoices (a duplicate charge)."""
    return {
        "customers": {
            "cus_001": {"id": "cus_001", "name": "TechCorp Inc", "email": "ap@techcorp.com"},
            "cus_002": {"id": "cus_002", "name": "Acme Co", "email": "billing@acme.com"},
            "cus_003": {"id": "cus_003", "name": "Globex", "email": "finance@globex.com"},
        },
        "invoices": {
            "inv_901": {"id": "inv_901", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},
            "inv_902": {"id": "inv_902", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},  # the duplicate
            "inv_903": {"id": "inv_903", "customer": "cus_002", "amount": 250, "month": "March", "status": "paid"},
            "inv_904": {"id": "inv_904", "customer": "cus_003", "amount": 800, "month": "February", "status": "paid"},
        },
        "refunds": {},
    }


def snapshot(db: dict) -> dict:
    """Deep-copy the DB so graders can compare before/after safely."""
    return deepcopy(db)
