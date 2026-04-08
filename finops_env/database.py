# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mock backend state. Each task gets a freshly-seeded dict via make_taskN_db().

Keeping these seeds deliberately small (<15 records) so that:
  1. Episodes stay short enough for a cheap LLM baseline to finish under 20 min.
  2. Graders can do exact dict comparison without fuzziness.
  3. A human reviewer can eyeball the "correct" answer by reading the seed.
"""

from copy import deepcopy


def make_task1_db() -> dict:
    """Task 1 seed — TechCorp has two identical March invoices (a duplicate charge).

    Correct action: refund inv_902 (or inv_901, either duplicate) while leaving
    the other March charge, Acme's invoice, and Globex's February invoice alone.
    """
    return {
        "customers": {
            "cus_001": {"id": "cus_001", "name": "TechCorp Inc", "email": "ap@techcorp.com", "status": "active"},
            "cus_002": {"id": "cus_002", "name": "Acme Co", "email": "billing@acme.com", "status": "active"},
            "cus_003": {"id": "cus_003", "name": "Globex", "email": "finance@globex.com", "status": "active"},
        },
        "invoices": {
            "inv_901": {"id": "inv_901", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},
            "inv_902": {"id": "inv_902", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},
            "inv_903": {"id": "inv_903", "customer": "cus_002", "amount": 250, "month": "March", "status": "paid"},
            "inv_904": {"id": "inv_904", "customer": "cus_003", "amount": 800, "month": "February", "status": "paid"},
        },
        "refunds": {},
        "discounts": {},
        "crm_users": {},
        "subscriptions": {},
    }


def make_task2_db() -> dict:
    """Task 2 seed — 8 active customers, 4 with lifetime spend > $500, 4 below.

    Correct action: apply a 10% discount to exactly {cus_001, cus_003, cus_005, cus_007}.
    """
    return {
        "customers": {
            "cus_001": {"id": "cus_001", "name": "TechCorp Inc", "email": "ap@techcorp.com", "status": "active"},
            "cus_002": {"id": "cus_002", "name": "Acme Co", "email": "billing@acme.com", "status": "active"},
            "cus_003": {"id": "cus_003", "name": "Globex", "email": "finance@globex.com", "status": "active"},
            "cus_004": {"id": "cus_004", "name": "Initech", "email": "ops@initech.com", "status": "active"},
            "cus_005": {"id": "cus_005", "name": "Hooli", "email": "pay@hooli.com", "status": "active"},
            "cus_006": {"id": "cus_006", "name": "Pied Piper", "email": "fin@pp.com", "status": "active"},
            "cus_007": {"id": "cus_007", "name": "Massive Dynamic", "email": "ar@md.com", "status": "active"},
            "cus_008": {"id": "cus_008", "name": "Stark Industries", "email": "billing@stark.com", "status": "active"},
        },
        "invoices": {
            # cus_001 — lifetime $900 (QUALIFIES)
            "inv_101": {"id": "inv_101", "customer": "cus_001", "amount": 500, "month": "January", "status": "paid"},
            "inv_102": {"id": "inv_102", "customer": "cus_001", "amount": 400, "month": "February", "status": "paid"},
            # cus_002 — lifetime $250 (below)
            "inv_103": {"id": "inv_103", "customer": "cus_002", "amount": 250, "month": "January", "status": "paid"},
            # cus_003 — lifetime $1200 (QUALIFIES)
            "inv_104": {"id": "inv_104", "customer": "cus_003", "amount": 600, "month": "January", "status": "paid"},
            "inv_105": {"id": "inv_105", "customer": "cus_003", "amount": 600, "month": "February", "status": "paid"},
            # cus_004 — lifetime $300 (below)
            "inv_106": {"id": "inv_106", "customer": "cus_004", "amount": 300, "month": "January", "status": "paid"},
            # cus_005 — lifetime $750 (QUALIFIES)
            "inv_107": {"id": "inv_107", "customer": "cus_005", "amount": 250, "month": "January", "status": "paid"},
            "inv_108": {"id": "inv_108", "customer": "cus_005", "amount": 250, "month": "February", "status": "paid"},
            "inv_109": {"id": "inv_109", "customer": "cus_005", "amount": 250, "month": "March", "status": "paid"},
            # cus_006 — lifetime $100 (below)
            "inv_110": {"id": "inv_110", "customer": "cus_006", "amount": 100, "month": "January", "status": "paid"},
            # cus_007 — lifetime $2000 (QUALIFIES)
            "inv_111": {"id": "inv_111", "customer": "cus_007", "amount": 2000, "month": "January", "status": "paid"},
            # cus_008 — lifetime $450 (below — just barely)
            "inv_112": {"id": "inv_112", "customer": "cus_008", "amount": 450, "month": "January", "status": "paid"},
        },
        "refunds": {},
        "discounts": {},
        "crm_users": {},
        "subscriptions": {},
    }


def make_task3_db() -> dict:
    """Task 3 seed — CRM ↔ Billing multi-system correlation.

    6 users total across two namespaces. Find users whose CRM status is
    'cancelled' AND who still have an 'active' billing subscription, then
    cancel their billing subscription.

    Correct target set: {sub_202, sub_204}.
    TRAP: cus_e is cancelled in CRM but subscription sub_205 is already
    cancelled in billing — do NOT re-cancel it (wastes a step and should
    be treated as wrong).
    """
    return {
        "customers": {},
        "invoices": {},
        "refunds": {},
        "discounts": {},
        "crm_users": {
            "crm_a": {"id": "crm_a", "email": "alice@ex.com", "status": "active"},
            "crm_b": {"id": "crm_b", "email": "bob@ex.com", "status": "cancelled"},    # target
            "crm_c": {"id": "crm_c", "email": "carol@ex.com", "status": "active"},
            "crm_d": {"id": "crm_d", "email": "dave@ex.com", "status": "cancelled"},   # target
            "crm_e": {"id": "crm_e", "email": "eve@ex.com", "status": "cancelled"},    # TRAP
            "crm_f": {"id": "crm_f", "email": "frank@ex.com", "status": "active"},
        },
        "subscriptions": {
            "sub_201": {"id": "sub_201", "email": "alice@ex.com", "status": "active"},
            "sub_202": {"id": "sub_202", "email": "bob@ex.com", "status": "active"},       # must cancel
            "sub_203": {"id": "sub_203", "email": "carol@ex.com", "status": "active"},
            "sub_204": {"id": "sub_204", "email": "dave@ex.com", "status": "active"},      # must cancel
            "sub_205": {"id": "sub_205", "email": "eve@ex.com", "status": "cancelled"},    # trap: already cancelled
            "sub_206": {"id": "sub_206", "email": "frank@ex.com", "status": "active"},
        },
    }


def snapshot(db: dict) -> dict:
    """Deep-copy the DB. Used by graders to capture before/after state."""
    return deepcopy(db)
