# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mock backend state. Each task gets a freshly-seeded dict via make_taskN_db().

Seed design principles:
  1. Large enough that a model can't just pattern-match its way to correctness —
     the agent has to actually read, filter, and compute.
  2. Each task contains at least one trap designed to make destructive-guessing
     agents fail the grader (wrong-customer refund, boundary-value exclusion,
     pre-cancelled subscription, email-mismatch dead-end).
  3. Graders remain deterministic — no fuzziness, no LLM judges.
"""

from copy import deepcopy


# ----------------------------------------------------------------------------
# Task 1 — Easy: Duplicate-charge refund with wrong-customer distractor traps
# ----------------------------------------------------------------------------

def make_task1_db() -> dict:
    """Task 1 seed — 15 customers, 18 invoices. Three customers have duplicate
    March charges; only TechCorp's is the correct target.

    Correct action: refund exactly one of {inv_901, inv_902} (TechCorp's March
    duplicates). Any refund of the trap duplicates (Initech's inv_905/inv_906
    or Stark's inv_910/inv_911) is a wrong-customer destructive error and
    scores zero.
    """
    return {
        "customers": {
            "cus_001": {"id": "cus_001", "name": "TechCorp Inc", "email": "ap@techcorp.com", "status": "active"},
            "cus_002": {"id": "cus_002", "name": "Acme Co", "email": "billing@acme.com", "status": "active"},
            "cus_003": {"id": "cus_003", "name": "Globex", "email": "finance@globex.com", "status": "active"},
            "cus_004": {"id": "cus_004", "name": "Initech", "email": "ap@initech.com", "status": "active"},
            "cus_005": {"id": "cus_005", "name": "Hooli", "email": "pay@hooli.com", "status": "active"},
            "cus_006": {"id": "cus_006", "name": "Pied Piper", "email": "fin@pp.com", "status": "active"},
            "cus_007": {"id": "cus_007", "name": "Massive Dynamic", "email": "ar@md.com", "status": "active"},
            "cus_008": {"id": "cus_008", "name": "Stark Industries", "email": "billing@stark.com", "status": "active"},
            "cus_009": {"id": "cus_009", "name": "Umbrella Corp", "email": "ap@umbrella.com", "status": "active"},
            "cus_010": {"id": "cus_010", "name": "Wayne Enterprises", "email": "finance@wayne.com", "status": "active"},
            "cus_011": {"id": "cus_011", "name": "OsCorp", "email": "billing@oscorp.com", "status": "active"},
            "cus_012": {"id": "cus_012", "name": "LexCorp", "email": "pay@lexcorp.com", "status": "active"},
            "cus_013": {"id": "cus_013", "name": "Dunder Mifflin", "email": "ap@dundermifflin.com", "status": "active"},
            "cus_014": {"id": "cus_014", "name": "Cyberdyne", "email": "finance@cyberdyne.com", "status": "active"},
            "cus_015": {"id": "cus_015", "name": "Soylent Corp", "email": "ap@soylent.com", "status": "active"},
        },
        "invoices": {
            # TechCorp — the correct target. Two identical March charges (duplicate).
            "inv_901": {"id": "inv_901", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},
            "inv_902": {"id": "inv_902", "customer": "cus_001", "amount": 500, "month": "March", "status": "paid"},
            # Acme — normal single March charge
            "inv_903": {"id": "inv_903", "customer": "cus_002", "amount": 250, "month": "March", "status": "paid"},
            # Globex — normal February
            "inv_904": {"id": "inv_904", "customer": "cus_003", "amount": 800, "month": "February", "status": "paid"},
            # Initech — TRAP: also has two identical March charges (wrong customer)
            "inv_905": {"id": "inv_905", "customer": "cus_004", "amount": 300, "month": "March", "status": "paid"},
            "inv_906": {"id": "inv_906", "customer": "cus_004", "amount": 300, "month": "March", "status": "paid"},
            # Hooli — normal April
            "inv_907": {"id": "inv_907", "customer": "cus_005", "amount": 600, "month": "April", "status": "paid"},
            # Pied Piper — normal January
            "inv_908": {"id": "inv_908", "customer": "cus_006", "amount": 150, "month": "January", "status": "paid"},
            # Massive Dynamic — normal single March (NOT a duplicate)
            "inv_909": {"id": "inv_909", "customer": "cus_007", "amount": 450, "month": "March", "status": "paid"},
            # Stark Industries — TRAP: another set of identical March duplicates
            "inv_910": {"id": "inv_910", "customer": "cus_008", "amount": 150, "month": "March", "status": "paid"},
            "inv_911": {"id": "inv_911", "customer": "cus_008", "amount": 150, "month": "March", "status": "paid"},
            # Umbrella Corp — normal February
            "inv_912": {"id": "inv_912", "customer": "cus_009", "amount": 700, "month": "February", "status": "paid"},
            # Wayne Enterprises — normal March
            "inv_913": {"id": "inv_913", "customer": "cus_010", "amount": 900, "month": "March", "status": "paid"},
            # OsCorp — normal March
            "inv_914": {"id": "inv_914", "customer": "cus_011", "amount": 350, "month": "March", "status": "paid"},
            # LexCorp — normal January
            "inv_915": {"id": "inv_915", "customer": "cus_012", "amount": 1200, "month": "January", "status": "paid"},
            # Dunder Mifflin — normal February
            "inv_916": {"id": "inv_916", "customer": "cus_013", "amount": 550, "month": "February", "status": "paid"},
            # Cyberdyne — normal March
            "inv_917": {"id": "inv_917", "customer": "cus_014", "amount": 420, "month": "March", "status": "paid"},
            # Soylent — normal April
            "inv_918": {"id": "inv_918", "customer": "cus_015", "amount": 680, "month": "April", "status": "paid"},
        },
        "refunds": {},
        "discounts": {},
        "crm_users": {},
        "subscriptions": {},
    }


# ----------------------------------------------------------------------------
# Task 2 — Medium: Boundary-exclusion and inactive-customer traps
# ----------------------------------------------------------------------------

def make_task2_db() -> dict:
    """Task 2 seed — 8 active customers, 2 inactive, 6 qualifiers.

    Correct action: apply a 10% discount to exactly
    {cus_001, cus_003, cus_005, cus_007, cus_012, cus_018}.

    Traps:
      - cus_010 (Wayne Enterprises): lifetime spend EXACTLY $500.00 — excluded
        because "exceeds $500" means strictly greater than.
      - cus_008 (Stark): $450 — near-boundary distractor below threshold.
      - cus_021 (Soylent inactive): $900 — above threshold but inactive.
      - cus_022 (Weyland-Yutani inactive): $700 — above threshold but inactive.
    """
    return {
        "customers": {
            "cus_001": {"id": "cus_001", "name": "TechCorp Inc", "email": "ap@techcorp.com", "status": "active"},
            "cus_003": {"id": "cus_003", "name": "Globex", "email": "finance@globex.com", "status": "active"},
            "cus_005": {"id": "cus_005", "name": "Hooli", "email": "pay@hooli.com", "status": "active"},
            "cus_007": {"id": "cus_007", "name": "Massive Dynamic", "email": "ar@md.com", "status": "active"},
            "cus_008": {"id": "cus_008", "name": "Stark Industries", "email": "billing@stark.com", "status": "active"},
            "cus_010": {"id": "cus_010", "name": "Wayne Enterprises", "email": "finance@wayne.com", "status": "active"},
            "cus_012": {"id": "cus_012", "name": "LexCorp", "email": "pay@lexcorp.com", "status": "active"},
            "cus_018": {"id": "cus_018", "name": "Cyberdyne Systems", "email": "finance@cyberdyne.com", "status": "active"},
            # inactive — trap if agent doesn't filter by status
            "cus_021": {"id": "cus_021", "name": "Soylent Corp", "email": "ap@soylent.com", "status": "inactive"},
            "cus_022": {"id": "cus_022", "name": "Weyland-Yutani", "email": "ap@wy.com", "status": "inactive"},
        },
        "invoices": {
            # cus_001 — lifetime $900 (QUALIFIES)
            "inv_101": {"id": "inv_101", "customer": "cus_001", "amount": 500, "month": "January", "status": "paid"},
            "inv_102": {"id": "inv_102", "customer": "cus_001", "amount": 400, "month": "February", "status": "paid"},
            # cus_003 — lifetime $1200 (QUALIFIES)
            "inv_104": {"id": "inv_104", "customer": "cus_003", "amount": 600, "month": "January", "status": "paid"},
            "inv_105": {"id": "inv_105", "customer": "cus_003", "amount": 600, "month": "February", "status": "paid"},
            # cus_005 — lifetime $750 (QUALIFIES)
            "inv_107": {"id": "inv_107", "customer": "cus_005", "amount": 250, "month": "January", "status": "paid"},
            "inv_108": {"id": "inv_108", "customer": "cus_005", "amount": 250, "month": "February", "status": "paid"},
            "inv_109": {"id": "inv_109", "customer": "cus_005", "amount": 250, "month": "March", "status": "paid"},
            # cus_007 — lifetime $2000 (QUALIFIES)
            "inv_111": {"id": "inv_111", "customer": "cus_007", "amount": 2000, "month": "January", "status": "paid"},
            # cus_008 — lifetime $450 (near-boundary distractor)
            "inv_112": {"id": "inv_112", "customer": "cus_008", "amount": 450, "month": "January", "status": "paid"},
            # cus_010 — lifetime $500 EXACTLY (BOUNDARY TRAP)
            "inv_114": {"id": "inv_114", "customer": "cus_010", "amount": 300, "month": "January", "status": "paid"},
            "inv_115": {"id": "inv_115", "customer": "cus_010", "amount": 200, "month": "February", "status": "paid"},
            # cus_012 — lifetime $1500 (QUALIFIES)
            "inv_117": {"id": "inv_117", "customer": "cus_012", "amount": 800, "month": "January", "status": "paid"},
            "inv_118": {"id": "inv_118", "customer": "cus_012", "amount": 700, "month": "February", "status": "paid"},
            # cus_018 — lifetime $850 (QUALIFIES)
            "inv_124": {"id": "inv_124", "customer": "cus_018", "amount": 500, "month": "January", "status": "paid"},
            "inv_125": {"id": "inv_125", "customer": "cus_018", "amount": 350, "month": "February", "status": "paid"},
            # cus_021 — inactive, lifetime $900 (INACTIVE TRAP)
            "inv_128": {"id": "inv_128", "customer": "cus_021", "amount": 900, "month": "January", "status": "paid"},
            # cus_022 — inactive, lifetime $700 (INACTIVE TRAP)
            "inv_129": {"id": "inv_129", "customer": "cus_022", "amount": 700, "month": "January", "status": "paid"},
        },
        "refunds": {},
        "discounts": {},
        "crm_users": {},
        "subscriptions": {},
    }


# ----------------------------------------------------------------------------
# Task 3 — Hard: Multi-system correlation with multiple traps
# ----------------------------------------------------------------------------

def make_task3_db() -> dict:
    """Task 3 seed — 13 CRM users, 12 billing subscriptions, 4 targets, 3 traps.

    Correct target set: {sub_202, sub_204, sub_207, sub_211}.

    Traps (all must be avoided):
      - sub_205 (eve): CRM cancelled, billing already cancelled
      - sub_209 (iris): CRM cancelled, billing already cancelled
      - sub_213 (maya): CRM cancelled, billing already cancelled
      - crm_o (olga): CRM cancelled, but her email has NO matching subscription
        (dead-end lookup — should be a no-op, not a destructive action)
    """
    return {
        "customers": {},
        "invoices": {},
        "refunds": {},
        "discounts": {},
        "crm_users": {
            "crm_a": {"id": "crm_a", "email": "alice@ex.com", "status": "active"},
            "crm_b": {"id": "crm_b", "email": "bob@ex.com", "status": "cancelled"},      # TARGET → sub_202
            "crm_c": {"id": "crm_c", "email": "carol@ex.com", "status": "active"},
            "crm_d": {"id": "crm_d", "email": "dave@ex.com", "status": "cancelled"},     # TARGET → sub_204
            "crm_e": {"id": "crm_e", "email": "eve@ex.com", "status": "cancelled"},      # TRAP: sub_205 already cancelled
            "crm_f": {"id": "crm_f", "email": "frank@ex.com", "status": "active"},
            "crm_g": {"id": "crm_g", "email": "grace@ex.com", "status": "cancelled"},    # TARGET → sub_207
            "crm_h": {"id": "crm_h", "email": "henry@ex.com", "status": "active"},
            "crm_i": {"id": "crm_i", "email": "iris@ex.com", "status": "cancelled"},     # TRAP: sub_209 already cancelled
            "crm_j": {"id": "crm_j", "email": "jack@ex.com", "status": "active"},
            "crm_k": {"id": "crm_k", "email": "karen@ex.com", "status": "cancelled"},    # TARGET → sub_211
            "crm_m": {"id": "crm_m", "email": "maya@ex.com", "status": "cancelled"},     # TRAP: sub_213 already cancelled
            "crm_o": {"id": "crm_o", "email": "olga@ex.com", "status": "cancelled"},     # TRAP: no matching sub (dead-end)
        },
        "subscriptions": {
            "sub_201": {"id": "sub_201", "email": "alice@ex.com", "status": "active"},
            "sub_202": {"id": "sub_202", "email": "bob@ex.com", "status": "active"},        # must cancel
            "sub_203": {"id": "sub_203", "email": "carol@ex.com", "status": "active"},
            "sub_204": {"id": "sub_204", "email": "dave@ex.com", "status": "active"},       # must cancel
            "sub_205": {"id": "sub_205", "email": "eve@ex.com", "status": "cancelled"},     # trap: already cancelled
            "sub_206": {"id": "sub_206", "email": "frank@ex.com", "status": "active"},
            "sub_207": {"id": "sub_207", "email": "grace@ex.com", "status": "active"},      # must cancel
            "sub_208": {"id": "sub_208", "email": "henry@ex.com", "status": "active"},
            "sub_209": {"id": "sub_209", "email": "iris@ex.com", "status": "cancelled"},    # trap: already cancelled
            "sub_210": {"id": "sub_210", "email": "jack@ex.com", "status": "active"},
            "sub_211": {"id": "sub_211", "email": "karen@ex.com", "status": "active"},      # must cancel
            "sub_213": {"id": "sub_213", "email": "maya@ex.com", "status": "cancelled"},    # trap: already cancelled
            # Note: no sub for olga@ex.com (crm_o) — the dead-end trap
        },
    }


def snapshot(db: dict) -> dict:
    """Deep-copy the DB. Used by graders to capture before/after state."""
    return deepcopy(db)
