# -*- coding: utf-8 -*-
"""Typed contracts for the travel reimbursement agent.

Everything that crosses a component boundary is one of these objects. Nothing
in the workflow passes a bare dict, which is what makes the graph state
inspectable and the output contract stable.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------- enumerations

class Category(str, Enum):
    FLIGHT = "FLIGHT"
    HOTEL = "HOTEL"
    MEALS = "MEALS"
    TAXI = "TAXI"
    INCIDENTAL = "INCIDENTAL"


class Outcome(str, Enum):
    APPROVE = "APPROVE"
    PARTIALLY_APPROVE = "PARTIALLY_APPROVE"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class Disposition(str, Enum):
    """What happened to a line. The three buckets, plus the clean case."""
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"
    REJECTED = "REJECTED"
    HELD = "HELD"          # cannot be assessed - neither approved nor rejected


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"          # a HIGH trigger can never be settled automatically


class TriggerCode(str, Enum):
    """EXC-001 manual review triggers."""
    EVIDENCE_UNREADABLE = "EVIDENCE_UNREADABLE"
    RECEIPT_CLAIM_MISMATCH = "RECEIPT_CLAIM_MISMATCH"
    NO_PRICING_CLAUSE = "NO_PRICING_CLAUSE"
    RECEIPT_MISSING = "RECEIPT_MISSING"
    ABOVE_AGENT_DELEGATION = "ABOVE_AGENT_DELEGATION"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    FOREIGN_CURRENCY = "FOREIGN_CURRENCY"
    SCHEMA_INVALID = "SCHEMA_INVALID"


# -------------------------------------------------------------------- inputs

class ClaimLine(BaseModel):
    line_id: str
    category: Category
    description: str
    amount: Decimal
    currency: str = "INR"
    date_from: str
    date_to: str
    nights: int = 0                       # HOTEL
    sectors: int = 0                      # FLIGHT
    travel_days: int = 1                  # MEALS, TAXI
    receipt_id: Optional[str] = None
    fare_class: Optional[str] = None      # FLIGHT
    room_type: Optional[str] = None       # HOTEL
    purpose: Optional[str] = None         # TAXI

    @property
    def is_priced_per_unit(self) -> int:
        return max(1, self.nights or self.sectors or self.travel_days or 1)


class Claim(BaseModel):
    claim_id: str
    employee_id: str
    employee_name: str
    trip_id: str
    trip_from: str
    trip_to: str
    submitted_on: str
    lines: List[ClaimLine]

    @property
    def claimed_total(self) -> Decimal:
        return sum((l.amount for l in self.lines), Decimal("0"))


class ReceiptRecord(BaseModel):
    """What the document store holds: an OCR rendering of a receipt image."""
    receipt_id: str
    ocr_text: str
    page_quality: float = Field(ge=0.0, le=1.0, default=0.95)


class LedgerEntry(BaseModel):
    """A previously settled claim line, for the DUP-001 match key."""
    employee_id: str
    receipt_id: str
    merchant: str
    date: str
    amount: Decimal
    settled_claim_id: str


# ----------------------------------------------------- model task outputs

class ReceiptExtraction(BaseModel):
    """Receipt Evidence Service output. Schema-validated before it is trusted."""
    receipt_id: str
    merchant: Optional[str] = None
    invoice_no: Optional[str] = None
    date: Optional[str] = None
    total: Optional[Decimal] = None
    currency: Optional[str] = None
    line_items: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def has_required_fields(self) -> bool:
        """RCP-001: a valid receipt carries all six fields."""
        return all([self.invoice_no, self.merchant, self.date,
                    self.total is not None, self.currency, self.line_items])


class ItemClassification(BaseModel):
    """What the model found on a receipt that policy excludes.

    It may only ever *reduce* an entitlement - there is no field here that can
    increase one.
    """
    excluded_amount: Decimal = Decimal("0")
    excluded_items: List[str] = Field(default_factory=list)
    clause: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class RuleBasis(str, Enum):
    """What the limit in a clause is charged against."""
    PER_SECTOR = "PER_SECTOR"
    PER_NIGHT = "PER_NIGHT"
    PER_TRAVEL_DAY = "PER_TRAVEL_DAY"
    PER_CLAIM = "PER_CLAIM"
    THRESHOLD = "THRESHOLD"
    NONE = "NONE"


class PolicyRule(BaseModel):
    """A rule read out of the policy document itself, not out of the code.

    `verified` is the important field: it is True only when `limit_amount`
    appears verbatim in the clause text it was read from. An unverified rule
    may not price anything - the line is held instead.
    """
    clause_id: str
    category: str
    limit_amount: Optional[Decimal] = None
    basis: RuleBasis = RuleBasis.NONE
    ineligible_terms: List[str] = Field(default_factory=list)
    excluded_items: List[str] = Field(default_factory=list)
    receipt_required_always: bool = False
    tiers: List[str] = Field(default_factory=list)      # APR-001 approver ladder
    source_phrase: str = ""
    verified: bool = False          # the limit was found verbatim in the clause
    tiers_verified: bool = False    # every tier ceiling and approver likewise

    @property
    def can_price(self) -> bool:
        return self.verified and self.limit_amount is not None


# ------------------------------------------------------------------ evidence

class RetrievedClause(BaseModel):
    clause_id: str
    title: str
    category: str
    clause_type: str                      # priced | eligibility | evidence | ...
    effective_from: str
    document: str
    version: str
    score: float = 0.0

    @property
    def is_authority_to_pay(self) -> bool:
        """A topically relevant clause is not authority to pay. Only a priced one is."""
        return self.clause_type == "priced"


# ------------------------------------------------------------------- outputs

class ReviewAction(str, Enum):
    """What a human reviewer may do with an escalated claim."""
    APPROVE = "APPROVE"       # accept the assessment as it stands
    AMEND = "AMEND"           # settle at an amount the reviewer sets
    REJECT = "REJECT"         # refuse the claim


class ReviewOutcome(BaseModel):
    """A human decision, recorded against the claim it settled."""
    reviewer: str
    action: ReviewAction
    approved_amount: Optional[Decimal] = None      # required for AMEND
    note: str = ""
    decided_at: str = ""


class Trigger(BaseModel):
    code: TriggerCode
    severity: Severity
    clause: str
    detail: str
    line_id: Optional[str] = None


class MissingDocument(BaseModel):
    line_id: str
    required: str
    clause: str


class LineAssessment(BaseModel):
    line_id: str
    category: Category
    claimed: Decimal
    approved: Decimal = Decimal("0")
    rejected: Decimal = Decimal("0")
    held: Decimal = Decimal("0")
    disposition: Disposition = Disposition.APPROVED
    clause: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0

    def balances(self) -> bool:
        return self.approved + self.rejected + self.held == self.claimed


class ApprovalDecision(BaseModel):
    tier: str
    within_agent_delegation: bool
    clause: str = "APR-001"


class Decision(BaseModel):
    """The output contract. Stable, machine-readable, and citable."""
    claim_id: str
    employee_id: str
    currency: str = "INR"
    decision: Outcome
    claimed_amount: Decimal
    approved_amount: Decimal
    rejected_amount: Decimal
    held_amount: Decimal
    lines: List[LineAssessment] = Field(default_factory=list)
    missing_documents: List[MissingDocument] = Field(default_factory=list)
    evidence: List[RetrievedClause] = Field(default_factory=list)
    approval: Optional[ApprovalDecision] = None
    review_triggers: List[Trigger] = Field(default_factory=list)
    manual_review: bool = False
    review: Optional[ReviewOutcome] = None
    confidence: float = 1.0
    explanation: str = ""
    audit: Dict[str, str] = Field(default_factory=dict)

    def balances(self) -> bool:
        """approved + rejected + held == claimed, on every claim."""
        return (self.approved_amount + self.rejected_amount + self.held_amount
                == self.claimed_amount)


class TraceStep(BaseModel):
    """One node execution, stamped with the correlation ID of its claim."""
    seq: int
    node: str
    summary: str
    elapsed_ms: int = 0
    correlation_id: str = ""
