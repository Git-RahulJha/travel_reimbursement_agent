"""Loads the sample claims, receipt documents and settled-claim ledger."""

#from __future__ import annotations

import json
import os
from typing import Dict, List

#from agent.contracts import LedgerEntry, ReceiptRecord
from .models.claim import Claim
from agent_azure_openai.models.receipts import RawReceipt, ParsedReceipt, Receipt


class ClaimRepository:
    """One place that knows where the sample data lives."""
    DATA = os.path.join(os.path.dirname(__file__), "..", "data")
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

    def raw_receipts(self) -> Dict[str, RawReceipt]:
                return [RawReceipt.model_validate(r) for r in self._read("receipts.json")]
                #return {r.receipt_id: r for r in records}

    def raw_receipt(self, receipt_id: str) -> Receipt:
            for receipt in self.raw_receipts():
                if receipt.receipt_id == receipt_id:
                    return receipt
    
            raise KeyError(f"No such receipt: {receipt_id}")
    
    def receipts(self) -> Dict[str, Receipt]:
            return [Receipt.model_validate(r) for r in self._read("receipts_structured.json")]
            #return {r.receipt_id: r for r in records}
    
    def receipt(self, receipt_id: str) -> Receipt:
        for receipt in self.receipts():
            if receipt.receipt_id == receipt_id:
                return receipt

        raise KeyError(f"No such receipt: {receipt_id}")

    #def ledger(self) -> List[LedgerEntry]:
    #    return [LedgerEntry.model_validate(e) for e in self._read("ledger.json")]
