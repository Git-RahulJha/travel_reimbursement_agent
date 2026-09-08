# -*- coding: utf-8 -*-
"""The deterministic core. This module owns every rupee.

It imports no model client and opens no socket - deliberately, and that is a
property a test can assert. Nothing else in the system is permitted to produce
an approved amount.

**No policy figure is hardcoded here.** Every limit, threshold and eligibility
term arrives as a `PolicyRule` read out of the policy PDF by the retriever and
verified against the clause text. This module decides only how to apply them:
which gate fires first, how a cap multiplies, how the buckets add up. Edit the
policy document and the arithmetic changes; edit this file and only the
*procedure* changes.

An unverified rule prices nothing. There is no fallback constant to fall back to.

Three buckets, not two. A line is approved, rejected, or **held** - the amount
the agent declines to assess because the evidence or the policy will not support
a decision. On every line and every claim:

    approved + rejected + held == claimed
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .contracts import (ApprovalDecision, Category, Claim, ClaimLine, Decision,
                        Disposition, ItemClassification, LedgerEntry,
                        LineAssessment, MissingDocument, Outcome, PolicyRule,
                        ReceiptExtraction, RetrievedClause, RuleBasis, Severity,
                        Trigger, TriggerCode)

ZERO = Decimal("0")


class EngineParameters:
    """The only numbers this engine owns.

    These are engineering parameters, not policy: they are not stated in the
    travel policy and no clause could supply them. Everything that *is* policy
    comes from the document.
    """
    CONFIDENCE_FLOOR = 0.70        # raised to 0.80 on the on-premise build
    MISMATCH_TOLERANCE = Decimal("1")


def money(value: Decimal) -> str:
    """Format a policy figure the way the document writes it: 8,000 not 8E+3."""
    if value == value.to_integral_value():
        return "{:,.0f}".format(value)
    return "{:,.2f}".format(value)


def _first_token(value: str) -> str:
    return re.split(r"[^A-Za-z]+", (value or "").upper().strip())[0] if value else ""


def _is_ineligible(value: Optional[str], terms: List[str]) -> Optional[str]:
    """Does a claimed fare class / room type / purpose match a barred term?

    Matched on the leading word, so 'BUSINESS' matches 'BUSINESS_CLASS' while
    'ECONOMY' does not match 'PREMIUM-ECONOMY'.
    """
    head = _first_token(value)
    if not head:
        return None
    for term in terms:
        if _first_token(term) == head:
            return term
    return None


class RuleEngine:
    """Assesses one line at a time against the rule read from its clause."""

    def __init__(self, params: EngineParameters = EngineParameters()):
        self.params = params

    # -- entry point -------------------------------------------------------

    def assess_line(self, line: ClaimLine, extraction: Optional[ReceiptExtraction],
                    clause: Optional[RetrievedClause], rule: Optional[PolicyRule],
                    classification: Optional[ItemClassification],
                    receipt_threshold: Optional[Decimal]
                    ) -> Tuple[LineAssessment, List[Trigger]]:
        """Four gates, then the category rule. Any gate short-circuits."""
        gate = (self._gate_currency(line)
                or self._gate_receipt(line, extraction, rule, receipt_threshold)
                or self._gate_evidence(line, extraction)
                or self._gate_authority(line, clause, rule))
        if gate is not None:
            return gate
        return self._price(line, extraction, rule, classification)

    # -- gates -------------------------------------------------------------

    def _gate_currency(self, line):
        """GEN-002: this policy pays in INR and converts nothing."""
        if line.currency.upper() != "INR":
            return self._held(line, "GEN-002",
                              "claimed in %s; this policy reimburses INR only"
                              % line.currency,
                              TriggerCode.FOREIGN_CURRENCY, Severity.HIGH)
        return None

    def _gate_receipt(self, line, extraction, rule, threshold):
        """RCP-001, plus the always-required cases the category clause states."""
        if threshold is None:
            return self._held(line, "RCP-001",
                              "the receipt threshold could not be read from the "
                              "policy document",
                              TriggerCode.NO_PRICING_CLAUSE, Severity.HIGH)
        always = bool(rule and rule.receipt_required_always)
        if not (always or line.amount > threshold):
            return None
        if not line.receipt_id or extraction is None:
            return self._rejected(line, "RCP-001",
                                  "receipt missing above the INR %s threshold"
                                  % money(threshold),
                                  TriggerCode.RECEIPT_MISSING, Severity.MEDIUM)
        return None

    def _gate_evidence(self, line, extraction):
        """EVD-001: unreadable evidence is held; a mismatch is never absorbed."""
        if extraction is None:
            return None
        floor = self.params.CONFIDENCE_FLOOR
        if extraction.confidence < floor:
            return self._held(line, "EVD-001",
                              "extraction confidence %.2f below the %.2f floor"
                              % (extraction.confidence, floor),
                              TriggerCode.EVIDENCE_UNREADABLE, Severity.HIGH)
        if not extraction.has_required_fields:
            return self._rejected(line, "RCP-001",
                                  "receipt is missing a required field",
                                  TriggerCode.RECEIPT_MISSING, Severity.MEDIUM)
        if (extraction.total is not None
                and abs(extraction.total - line.amount) > self.params.MISMATCH_TOLERANCE):
            return self._held(line, "EVD-001",
                              "receipt reads INR %s against INR %s claimed"
                              % (extraction.total, line.amount),
                              TriggerCode.RECEIPT_CLAIM_MISMATCH, Severity.HIGH)
        return None

    def _gate_authority(self, line, clause, rule):
        """Retrieval is evidence, not authority. Only a verified priced rule pays."""
        if clause is None:
            return self._held(line, "EXC-001",
                              "no clause of this policy was retrieved for this line",
                              TriggerCode.NO_PRICING_CLAUSE, Severity.HIGH)
        if rule is None or not rule.can_price:
            detail = ("no clause of this policy prices this expense (nearest: %s)"
                      % clause.clause_id if rule is None else
                      "the limit in %s could not be verified against the clause text"
                      % clause.clause_id)
            return self._held(line, clause.clause_id, detail,
                              TriggerCode.NO_PRICING_CLAUSE, Severity.HIGH)
        return None

    # -- pricing, entirely from the rule the document supplied --------------

    def _units(self, line: ClaimLine, basis: RuleBasis) -> int:
        if basis is RuleBasis.PER_SECTOR:
            return max(1, line.sectors)
        if basis is RuleBasis.PER_NIGHT:
            return max(1, line.nights)
        if basis is RuleBasis.PER_TRAVEL_DAY:
            return max(1, line.travel_days)
        return 1

    def _price(self, line: ClaimLine, extraction, rule: PolicyRule,
               classification: Optional[ItemClassification]):
        confidence = extraction.confidence if extraction else 0.95
        clause_id = rule.clause_id

        barred = _is_ineligible(line.fare_class or line.purpose, rule.ineligible_terms)
        if barred and line.category in (Category.FLIGHT, Category.TAXI):
            return self._rejected(line, clause_id,
                                  "%s is not eligible under this clause"
                                  % barred.lower().replace("_", " "),
                                  None, None, confidence)

        cap = rule.limit_amount * self._units(line, rule.basis)

        excluded = (classification.excluded_amount
                    if classification and classification.excluded_amount else ZERO)
        allowable = min(line.amount - excluded, cap)
        approved = max(ZERO, allowable)

        if approved == line.amount:
            reason = "within policy"
        elif excluded > 0 and line.amount - excluded > cap:
            reason = "non-reimbursable items removed and limit applied"
        elif excluded > 0:
            reason = "non-reimbursable items removed"
        else:
            reason = "above the %s limit of INR %s" % (
                rule.basis.value.lower().replace("per_", "per ").replace("_", " "),
                money(rule.limit_amount))

        downgraded = _is_ineligible(line.room_type, rule.ineligible_terms)
        if downgraded and line.category is Category.HOTEL and approved < line.amount:
            reason = ("%s room is not eligible; paid at the standard entitlement"
                      % downgraded.lower().replace("_", " "))

        return self._settle(line, clause_id, approved, reason, confidence)

    # -- assessment builders ------------------------------------------------

    def _settle(self, line, clause_id, approved, reason, confidence):
        approved = max(ZERO, min(approved, line.amount))
        rejected = line.amount - approved
        disposition = (Disposition.APPROVED if rejected == ZERO
                       else Disposition.REDUCED if approved > ZERO
                       else Disposition.REJECTED)
        return LineAssessment(
            line_id=line.line_id, category=line.category, claimed=line.amount,
            approved=approved, rejected=rejected, held=ZERO,
            disposition=disposition, clause=clause_id, reason=reason,
            confidence=confidence), []

    def _rejected(self, line, clause_id, reason, code, severity, confidence=0.99):
        triggers = ([Trigger(code=code, severity=severity, clause=clause_id,
                             detail=reason, line_id=line.line_id)]
                    if code else [])
        return LineAssessment(
            line_id=line.line_id, category=line.category, claimed=line.amount,
            approved=ZERO, rejected=line.amount, held=ZERO,
            disposition=Disposition.REJECTED, clause=clause_id, reason=reason,
            confidence=confidence), triggers

    def _held(self, line, clause_id, reason, code, severity):
        return LineAssessment(
            line_id=line.line_id, category=line.category, claimed=line.amount,
            approved=ZERO, rejected=ZERO, held=line.amount,
            disposition=Disposition.HELD, clause=clause_id, reason=reason,
            confidence=0.35), [Trigger(code=code, severity=severity,
                                       clause=clause_id, detail=reason,
                                       line_id=line.line_id)]


# ------------------------------------------------------------------ decision

class DecisionEngine:
    """Aggregates line results into one outcome. Still no model, still no network."""

    def apply_duplicates(self, assessments: List[LineAssessment],
                         duplicates: Dict[str, LedgerEntry]
                         ) -> Tuple[List[LineAssessment], List[Trigger]]:
        """DUP-001: an exact key match is rejected outright, not escalated."""
        out: List[LineAssessment] = []
        for a in assessments:
            hit = duplicates.get(a.line_id)
            if hit is None:
                out.append(a)
                continue
            out.append(a.model_copy(update={
                "approved": ZERO, "rejected": a.claimed, "held": ZERO,
                "disposition": Disposition.REJECTED, "clause": "DUP-001",
                "reason": "already settled on claim %s" % hit.settled_claim_id,
                "confidence": 0.99}))
        return out, []

    def compose(self, claim: Claim, assessments: List[LineAssessment],
                triggers: List[Trigger], approval: ApprovalDecision,
                evidence: List[RetrievedClause]) -> Decision:
        approved = sum((a.approved for a in assessments), ZERO)
        rejected = sum((a.rejected for a in assessments), ZERO)
        held = sum((a.held for a in assessments), ZERO)
        claimed = claim.claimed_total

        if not approval.within_agent_delegation:
            triggers = triggers + [Trigger(
                code=TriggerCode.ABOVE_AGENT_DELEGATION, severity=Severity.HIGH,
                clause="AGT-001",
                detail="claim total INR %s exceeds the agent's delegation" % claimed)]

        escalate = any(t.severity is Severity.HIGH for t in triggers)
        if escalate:
            outcome = Outcome.MANUAL_REVIEW
        elif approved == ZERO and claimed > ZERO:
            outcome = Outcome.REJECT
        elif approved < claimed:
            outcome = Outcome.PARTIALLY_APPROVE
        else:
            outcome = Outcome.APPROVE

        confidence = min([a.confidence for a in assessments], default=1.0)
        confidence -= 0.05 * sum(1 for t in triggers if t.severity is Severity.MEDIUM)

        missing = [MissingDocument(line_id=t.line_id, required="TAX_INVOICE",
                                   clause=t.clause)
                   for t in triggers
                   if t.code is TriggerCode.RECEIPT_MISSING and t.line_id]

        return Decision(
            claim_id=claim.claim_id, employee_id=claim.employee_id,
            decision=outcome, claimed_amount=claimed, approved_amount=approved,
            rejected_amount=rejected, held_amount=held, lines=assessments,
            missing_documents=missing, evidence=evidence, approval=approval,
            review_triggers=triggers, manual_review=escalate,
            confidence=round(max(0.0, min(1.0, confidence)), 2))


class SafetyGate:
    """The last deterministic check before anything is called final.

    It cannot be talked round: it reads open triggers and the delegation, and
    nothing else. A model has no input at this node.
    """

    def evaluate(self, decision: Decision) -> Tuple[bool, str]:
        blocking = [t for t in decision.review_triggers
                    if t.severity is Severity.HIGH]
        if blocking:
            return False, "escalated: %s" % ", ".join(
                sorted({t.code.value for t in blocking}))
        if decision.approval and not decision.approval.within_agent_delegation:
            return False, "escalated: above the agent's delegation (AGT-001)"
        return True, "finalised within the agent's delegation"
