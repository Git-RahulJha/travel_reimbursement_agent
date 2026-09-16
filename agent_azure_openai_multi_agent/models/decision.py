# models/decision.py

from typing import Optional
from pydantic import BaseModel


class LineDecision(BaseModel):
    line_id: str
    category: str
    claimed_amount: float
    approved_amount: float
    rejected_amount: float
    status: str
    reason: str
    receipt_present: bool

#status: APPROVED, PARTIALLY_APPROVED, REJECTED, MANUAL_REVIEW
class ClaimDecision(BaseModel):
    claim_id: str
    claimed_amount: float
    approved_amount: float
    rejected_amount: float
    status: str
    approval_authority: Optional[str] = None
    line_decisions: list[LineDecision]