# -*- coding: utf-8 -*-
"""Loads the sample claims, receipt documents and settled-claim ledger.

In production these are HRIS, object storage and the claims ledger, each behind
the same interface. Here they are local JSON, which is why the demo runs with no
network at all.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .contracts import Claim, LedgerEntry, ReceiptRecord

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))


class ClaimRepository:
    """One place that knows where the sample data lives."""

    def __init__(self, path: str = DATA):
        self.path = path

    def _read(self, name: str):
        with open(os.path.join(self.path, name), encoding="utf-8") as fh:
            return json.load(fh)

    def claims(self) -> List[Claim]:
        return [Claim.model_validate(c) for c in self._read("claims.json")]

    def claim(self, claim_id: str) -> Claim:
        for c in self.claims():
            if c.claim_id == claim_id:
                return c
        raise KeyError("no such claim: %s" % claim_id)

    def receipts(self) -> Dict[str, ReceiptRecord]:
        records = [ReceiptRecord.model_validate(r) for r in self._read("receipts.json")]
        return {r.receipt_id: r for r in records}

    def ledger(self) -> List[LedgerEntry]:
        return [LedgerEntry.model_validate(e) for e in self._read("ledger.json")]
