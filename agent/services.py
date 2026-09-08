# -*- coding: utf-8 -*-
"""The deterministic services the workflow calls, and the audit store.

Each service is a declared contract: a purpose, typed inputs and outputs, a
retry policy, and a stated behaviour on failure. None of them ever answers
'approve anyway' - every failure resolves to escalation.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from .contracts import (ApprovalDecision, ClaimLine, Decision, LedgerEntry,
                        ReceiptExtraction, ReceiptRecord, TraceStep)
from .llm import ModelBackend


class ServiceUnavailable(RuntimeError):
    """A dependency that did not answer within its retry policy."""


@dataclass
class ToolContract:
    """The published contract for one external interaction.

    Purpose, inputs, outputs and failure handling are declared here rather than
    left implicit, so the interface can be reviewed without reading the caller.
    """
    name: str
    purpose: str
    inputs: str
    outputs: str
    timeout_ms: int
    retries: int
    on_failure: str
    idempotent: bool = True


CATALOGUE: Dict[str, ToolContract] = {}


def declare(contract: "ToolContract") -> "ToolContract":
    """Publish a contract enforced elsewhere, so the catalogue stays complete."""
    CATALOGUE[contract.name] = contract
    return contract


def tool(contract: ToolContract):
    """Publish a contract and enforce its retry policy on the method."""
    CATALOGUE[contract.name] = contract

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last: Optional[Exception] = None
            for attempt in range(contract.retries + 1):
                try:
                    return fn(*args, **kwargs)
                except ServiceUnavailable as exc:
                    last = exc
                    time.sleep(0)          # a real client would back off here
            raise ServiceUnavailable("%s failed after %d attempt(s): %s"
                                     % (contract.name, contract.retries + 1, last))
        wrapper.contract = contract
        return wrapper
    return decorator


# ------------------------------------------------------------------ evidence

class ReceiptEvidenceService:
    """Turns a receipt document into structured fields with a confidence.

    The only component besides retrieval that touches a model. It never decides
    anything - it reports what it could read and how sure it is.
    """

    def __init__(self, receipts: Dict[str, ReceiptRecord], backend: ModelBackend,
                 fail: bool = False):
        self._receipts = receipts
        self._backend = backend
        self._fail = fail

    @tool(ToolContract(
        name="docs.extract_receipt",
        purpose="Turn a receipt image or PDF into structured invoice fields with "
                "a calibrated confidence.",
        inputs="receipt_id",
        outputs="merchant, invoice no, date, total, currency, line items, confidence",
        timeout_ms=15000, retries=1,
        on_failure="MANUAL_REVIEW - never estimate an unreadable amount"))
    def extract(self, receipt_id: str) -> Optional[ReceiptExtraction]:
        if self._fail:
            raise ServiceUnavailable("document intelligence unavailable")
        record = self._receipts.get(receipt_id)
        if record is None:
            return None
        return self._backend.extract_receipt(record)


# ----------------------------------------------------------------- controls

class DuplicateCheckService:
    """DUP-001. Deterministic key match against the settled-claim ledger.

    No model is involved, and none could be: 'has this been paid before' is a
    question of exact matching, not of judgement.
    """

    KEY = ("employee_id", "receipt_id", "merchant", "date", "amount")

    def __init__(self, ledger: List[LedgerEntry], fail: bool = False):
        self._ledger = ledger
        self._fail = fail

    @staticmethod
    def _key(employee_id: str, receipt_id: str, merchant: str, date: str,
             amount: Decimal) -> str:
        return "|".join([employee_id, receipt_id, (merchant or "").strip().lower(),
                         date or "", str(amount)])

    @tool(ToolContract(
        name="controls.check_duplicate",
        purpose="Detect a claim line already submitted or settled within the "
                "DUP-001 lookback window.",
        inputs="employee id, claim line, receipt extraction",
        outputs="the matching settled ledger entry, or none",
        timeout_ms=1000, retries=2,
        on_failure="MANUAL_REVIEW - never assume a line is new"))
    def check(self, employee_id: str, line: ClaimLine,
              extraction: Optional[ReceiptExtraction]) -> Optional[LedgerEntry]:
        if self._fail:
            raise ServiceUnavailable("claims ledger unavailable")
        if extraction is None or not extraction.invoice_no:
            return None
        probe = self._key(employee_id, line.receipt_id or "",
                          extraction.merchant or "", extraction.date or "",
                          line.amount)
        for entry in self._ledger:
            candidate = self._key(entry.employee_id, entry.receipt_id,
                                  entry.merchant, entry.date, entry.amount)
            if candidate == probe:
                return entry
        return None


class ApprovalMatrixService:
    """APR-001 and AGT-001. A table read, not a judgement.

    The ladder and the delegation limit are supplied by the caller, having been
    read out of the policy document - nothing here is hardcoded.
    """

    def __init__(self, tiers=None, delegation: Optional[Decimal] = None,
                 fail: bool = False):
        self._tiers = tiers or []
        self._delegation = delegation
        self._fail = fail

    @tool(ToolContract(
        name="approval.lookup_matrix",
        purpose="Resolve the approver for a claim total under the delegation of "
                "authority read from APR-001, and whether it is inside the "
                "agent's own delegation under AGT-001.",
        inputs="claim total",
        outputs="approver tier, within_agent_delegation flag",
        timeout_ms=700, retries=2,
        on_failure="Escalate to the highest tier"))
    def resolve(self, claimed_total: Decimal) -> ApprovalDecision:
        if self._fail:
            raise ServiceUnavailable("delegation service unavailable")
        if not self._tiers or self._delegation is None:
            # the ladder could not be read from the document - escalate, never guess
            return ApprovalDecision(tier="FINANCE_CONTROLLER",
                                    within_agent_delegation=False)
        tier = self._tiers[-1][1]
        for ceiling, approver in self._tiers:
            if ceiling is None or claimed_total <= ceiling:
                tier = approver
                break
        return ApprovalDecision(tier=tier,
                                within_agent_delegation=claimed_total <= self._delegation)


# --------------------------------------------------------------------- audit

@dataclass
class AuditRecord:
    claim_id: str
    trace: List[TraceStep]
    decision: Decision
    policy_version: str
    model_backend: str
    workflow_version: str
    digest: str = ""


class AuditStore:
    """Every run is recorded with the evidence it used and a digest.

    The digest is over the decision only, so the same claim assessed twice
    produces the same digest - which is the reproducibility claim, testable.
    """

    WORKFLOW_VERSION = "1.0.0"

    def __init__(self) -> None:
        self._records: Dict[str, AuditRecord] = {}

    def write(self, decision: Decision, trace: List[TraceStep],
              policy_version: str, model_backend: str) -> AuditRecord:
        payload = decision.model_dump_json(exclude={"audit", "explanation"})
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        record = AuditRecord(
            claim_id=decision.claim_id, trace=trace, decision=decision,
            policy_version=policy_version, model_backend=model_backend,
            workflow_version=self.WORKFLOW_VERSION, digest=digest)
        self._records[decision.claim_id] = record
        return record

    def get(self, claim_id: str) -> Optional[AuditRecord]:
        return self._records.get(claim_id)


# Declared here, enforced in their own modules. policy.search lives in
# knowledge.py; rules.assess_claim lives in rules.py, which imports no model
# client and no service code by design.

declare(ToolContract(
    name="policy.search",
    purpose="Retrieve the clauses governing a claim line, filtered to the policy "
            "version in force on the travel dates, and read the limit each states.",
    inputs="category, line description, travel start date",
    outputs="clause id, type, effective date, similarity score, verified limit",
    timeout_ms=1500, retries=2,
    on_failure="MANUAL_REVIEW - never decide a line without policy evidence"))

declare(ToolContract(
    name="rules.assess_claim",
    purpose="Compute approved, rejected and held amounts per line from the "
            "retrieved rule, and derive the claim outcome.",
    inputs="claim lines, receipt extractions, retrieved rules, receipt threshold",
    outputs="per-line assessment with clause citation, claim totals, open triggers",
    timeout_ms=50, retries=0,
    on_failure="MANUAL_REVIEW - an arithmetic or contract violation never settles"))
