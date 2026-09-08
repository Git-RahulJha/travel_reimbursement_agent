# -*- coding: utf-8 -*-
"""The orchestration graph.

An explicit LangGraph state machine, not an autonomous planner. Eight nodes,
two conditional edges, four outcomes. The model is invoked inside two leaf
nodes and never directs the graph, so every path is enumerable and testable and
the trace has a fixed shape.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .contracts import (ApprovalDecision, Claim, Decision, Disposition,
                        ItemClassification, LedgerEntry, LineAssessment, Outcome,
                        ReceiptExtraction, ReceiptRecord, RetrievedClause,
                        PolicyRule, ReviewAction, ReviewOutcome, Severity,
                        TraceStep, Trigger, TriggerCode)
from .knowledge import PolicyKnowledgeBase, PolicyRuleResolver
from .llm import ModelBackend, ModelError, OfflineBackend
from .rules import DecisionEngine, RuleEngine, SafetyGate, money
from .services import (ApprovalMatrixService, AuditStore, DuplicateCheckService,
                       ReceiptEvidenceService, ServiceUnavailable)


class ClaimState(TypedDict, total=False):
    """Typed graph state. Every node reads and writes this and nothing else."""
    claim: Claim
    correlation_id: str          # assigned at intake, survives suspend and resume
    admissible: bool
    extractions: Dict[str, ReceiptExtraction]
    classifications: Dict[str, ItemClassification]
    clauses: Dict[str, RetrievedClause]
    rules: Dict[str, PolicyRule]
    assessments: List[LineAssessment]
    duplicates: Dict[str, LedgerEntry]
    approval: ApprovalDecision
    triggers: List[Trigger]
    result: Decision
    trace: List[TraceStep]
    finalised: bool
    gate_reason: str
    review: ReviewOutcome        # written by a human, on resume


class ReimbursementAgent:
    """Wires the components into a graph and runs claims through it."""

    def __init__(self, knowledge: PolicyKnowledgeBase,
                 receipts: Dict[str, ReceiptRecord],
                 ledger: List[LedgerEntry],
                 backend: Optional[ModelBackend] = None,
                 fail: Optional[str] = None):
        self.knowledge = knowledge
        self.backend = backend or OfflineBackend()
        self.evidence_service = ReceiptEvidenceService(
            receipts, self.backend, fail=(fail == "evidence"))
        self.duplicate_service = DuplicateCheckService(
            ledger, fail=(fail == "duplicate"))
        self.resolver = PolicyRuleResolver(knowledge, self.backend)
        self.approval_service = ApprovalMatrixService(
            tiers=self.resolver.approval_tiers(),
            delegation=self.resolver.agent_delegation(),
            fail=(fail == "approval"))
        self.rules = RuleEngine()
        self.decisions = DecisionEngine()
        self.gate = SafetyGate()
        self.audit = AuditStore()
        self._fail = fail
        self.checkpoints = MemorySaver()
        self._threads: Dict[str, Claim] = {}
        self.graph = self._build()

    # ------------------------------------------------------------- graph

    def _build(self):
        g = StateGraph(ClaimState)
        for name in ("intake", "evidence", "retrieval", "validation", "tools",
                     "decision", "safety_gate", "finalise", "manual_review",
                     "reject_intake", "audit"):
            g.add_node(name, getattr(self, "n_" + name))

        g.set_entry_point("intake")
        g.add_conditional_edges("intake", self._admissible,
                                {"proceed": "evidence", "reject": "reject_intake"})
        g.add_edge("evidence", "retrieval")
        g.add_edge("retrieval", "validation")
        g.add_edge("validation", "tools")
        g.add_edge("tools", "decision")
        g.add_edge("decision", "safety_gate")
        g.add_conditional_edges("safety_gate", self._route,
                                {"finalise": "finalise",
                                 "escalate": "manual_review"})
        g.add_edge("finalise", "audit")
        g.add_edge("manual_review", "audit")
        g.add_edge("reject_intake", "audit")
        g.add_edge("audit", END)
        # The human-in-the-loop boundary. The run suspends here with its state
        # checkpointed; nothing settles until a person calls resume().
        return g.compile(checkpointer=self.checkpoints,
                         interrupt_before=["manual_review"])

    @staticmethod
    def _admissible(state: ClaimState) -> str:
        return "proceed" if state.get("admissible") else "reject"

    @staticmethod
    def _route(state: ClaimState) -> str:
        return "finalise" if state.get("finalised") else "escalate"

    def mermaid(self) -> str:
        """The workflow diagram, generated from the graph that actually runs."""
        return self.graph.get_graph().draw_mermaid()

    # ------------------------------------------------------------- nodes

    def _step(self, state: ClaimState, node: str, summary: str, started: float,
              correlation_id: str = "") -> List[TraceStep]:
        """Append one trace step, stamped with the claim's correlation ID.

        The ID is assigned at intake and lives in checkpointed state, so it is
        the same before and after a suspension for human review - one identifier
        spans the whole journey of a claim, including the part a person did.
        """
        trace = list(state.get("trace", []))
        trace.append(TraceStep(
            seq=len(trace) + 1, node=node, summary=summary,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            correlation_id=correlation_id or state.get("correlation_id", "")))
        return trace

    def n_intake(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        claim = state["claim"]
        correlation_id = state.get("correlation_id") or uuid.uuid4().hex[:12]
        problems = []
        if not claim.lines:
            problems.append("claim carries no lines")
        if not claim.trip_id:
            problems.append("claim does not reference an approved trip")

        triggers = list(state.get("triggers", []))
        if problems:
            triggers.append(Trigger(code=TriggerCode.SCHEMA_INVALID,
                                    severity=Severity.HIGH, clause="GEN-003",
                                    detail="; ".join(problems)))
        summary = ("%s: %d line(s), %s, trip %s %s..%s  [corr %s]"
                   % (claim.claim_id, len(claim.lines), claim.employee_name,
                      claim.trip_id, claim.trip_from, claim.trip_to, correlation_id)
                   if not problems else "rejected at intake: " + "; ".join(problems))
        return {"admissible": not problems, "triggers": triggers,
                "correlation_id": correlation_id,
                "trace": self._step(state, "intake", summary, t0, correlation_id)}

    def n_evidence(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        claim, evidence = state["claim"], {}
        triggers = list(state.get("triggers", []))
        for line in claim.lines:
            if not line.receipt_id:
                continue
            try:
                found = self.evidence_service.extract(line.receipt_id)
            except (ServiceUnavailable, ModelError) as exc:
                triggers.append(Trigger(
                    code=TriggerCode.SERVICE_UNAVAILABLE, severity=Severity.HIGH,
                    clause="EXC-001", detail=str(exc), line_id=line.line_id))
                continue
            if found is not None:
                evidence[line.line_id] = found

        detail = ", ".join("%s:%.2f" % (k, v.confidence)
                           for k, v in evidence.items()) or "no receipts supplied"
        return {"extractions": evidence, "triggers": triggers,
                "trace": self._step(state, "evidence",
                                    "extraction confidence - " + detail, t0)}

    def n_retrieval(self, state: ClaimState) -> ClaimState:
        """Retrieve the governing clause for each line, and read its rule.

        This is the RAG step. The clause comes from the vector index; the limit
        comes from the clause text, verified against it. No figure in this
        system is a constant in the code.
        """
        t0 = time.perf_counter()
        claim = state["claim"]
        clauses: Dict[str, RetrievedClause] = {}
        rules: Dict[str, PolicyRule] = {}

        for line in claim.lines:
            query = "%s %s" % (line.category.value, line.description)
            clause, rule = self.resolver.for_line(
                line.category.value, query, as_of=claim.trip_from)
            if clause is not None:
                clauses[line.line_id] = clause
            if rule is not None:
                rules[line.line_id] = rule

        detail = "; ".join(
            "%s -> %s (%s%s)"
            % (lid, c.clause_id, c.clause_type,
               ", INR %s %s" % (money(rules[lid].limit_amount),
                                rules[lid].basis.value.lower())
               if lid in rules and rules[lid].can_price else ", prices nothing")
            for lid, c in clauses.items()) or "nothing retrieved"
        return {"clauses": clauses, "rules": rules,
                "trace": self._step(state, "retrieval", detail, t0)}

    def n_validation(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        claim = state["claim"]
        evidence = state.get("extractions", {})
        clauses = state.get("clauses", {})
        rules = state.get("rules", {})
        threshold = self.resolver.receipt_threshold()
        triggers = list(state.get("triggers", []))
        assessments, classifications = [], {}

        for line in claim.lines:
            extraction = evidence.get(line.line_id)
            clause = clauses.get(line.line_id)

            classification = None
            if extraction is not None and clause is not None and clause.is_authority_to_pay:
                try:
                    classification = self.backend.classify_items(
                        line, extraction, self.knowledge.text(clause.clause_id))
                    classifications[line.line_id] = classification
                except ModelError as exc:
                    triggers.append(Trigger(
                        code=TriggerCode.SERVICE_UNAVAILABLE,
                        severity=Severity.HIGH, clause="EXC-001",
                        detail=str(exc), line_id=line.line_id))

            assessment, line_triggers = self.rules.assess_line(
                line, extraction, clause, rules.get(line.line_id), classification,
                threshold)
            assessments.append(assessment)
            triggers.extend(line_triggers)

        approvable = sum((a.approved for a in assessments), Decimal("0"))
        return {"assessments": assessments, "classifications": classifications,
                "triggers": triggers,
                "trace": self._step(state, "validation",
                                    "assessed %d line(s), approvable INR %s"
                                    % (len(assessments), approvable), t0)}

    def n_tools(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        claim = state["claim"]
        evidence = state.get("extractions", {})
        triggers = list(state.get("triggers", []))
        duplicates: Dict[str, LedgerEntry] = {}

        try:
            for line in claim.lines:
                hit = self.duplicate_service.check(
                    claim.employee_id, line, evidence.get(line.line_id))
                if hit is not None:
                    duplicates[line.line_id] = hit
        except ServiceUnavailable as exc:
            triggers.append(Trigger(code=TriggerCode.SERVICE_UNAVAILABLE,
                                    severity=Severity.HIGH, clause="DUP-001",
                                    detail=str(exc)))

        try:
            approval = self.approval_service.resolve(claim.claimed_total)
        except ServiceUnavailable as exc:
            approval = ApprovalDecision(tier="FINANCE_CONTROLLER",
                                        within_agent_delegation=False)
            triggers.append(Trigger(code=TriggerCode.SERVICE_UNAVAILABLE,
                                    severity=Severity.HIGH, clause="APR-001",
                                    detail=str(exc)))

        return {"duplicates": duplicates, "approval": approval,
                "triggers": triggers,
                "trace": self._step(state, "tools",
                                    "%d duplicate(s); approver %s"
                                    % (len(duplicates), approval.tier), t0)}

    def n_decision(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        assessments, dup_triggers = self.decisions.apply_duplicates(
            state["assessments"], state.get("duplicates", {}))
        triggers = list(state.get("triggers", [])) + dup_triggers

        evidence_clauses = list({c.clause_id: c
                                 for c in state.get("clauses", {}).values()}.values())
        decision = self.decisions.compose(
            state["claim"], assessments, triggers, state["approval"],
            evidence_clauses)

        return {"assessments": assessments, "result": decision,
                "triggers": decision.review_triggers,
                "trace": self._step(
                    state, "decision",
                    "%s - claimed %s, approved %s, rejected %s, held %s"
                    % (decision.decision.value, decision.claimed_amount,
                       decision.approved_amount, decision.rejected_amount,
                       decision.held_amount), t0)}

    def n_safety_gate(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        finalised, reason = self.gate.evaluate(state["result"])
        return {"finalised": finalised, "gate_reason": reason,
                "trace": self._step(state, "safety_gate", reason, t0)}

    def n_finalise(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        decision = state["result"]
        decision.explanation = self.backend.explain(decision)
        return {"result": decision,
                "trace": self._step(state, "finalise",
                                    "decision final; explanation generated from "
                                    "the decision object", t0)}

    def n_manual_review(self, state: ClaimState) -> ClaimState:
        """Runs only after a human has acted.

        The graph is compiled with interrupt_before on this node, so execution
        suspends before it and the claim waits. On resume the reviewer's outcome
        is in state, and this node applies it and records who decided.
        """
        t0 = time.perf_counter()
        decision = state["result"]
        decision.manual_review = True
        review: Optional[ReviewOutcome] = state.get("review")

        if review is None:
            # resumed with no outcome supplied - stay escalated, settle nothing
            decision.decision = Outcome.MANUAL_REVIEW
            decision.explanation = self.backend.explain(decision)
            codes = sorted({t.code.value for t in decision.review_triggers})
            return {"result": decision,
                    "trace": self._step(state, "manual_review",
                                        "no reviewer outcome; claim remains open "
                                        "(%s)" % (", ".join(codes) or "none"), t0)}

        decision.review = review
        summary = self._apply_review(decision, review)
        decision.explanation = self.backend.explain(decision)
        return {"result": decision,
                "trace": self._step(state, "manual_review", summary, t0)}

    def _apply_review(self, decision: Decision, review: ReviewOutcome) -> str:
        """Settle the claim on the human's authority.

        The held bucket is what the agent declined to assess, so a human
        approval is precisely the act of releasing it.
        """
        if review.action is ReviewAction.REJECT:
            decision.decision = Outcome.REJECT
            decision.rejected_amount = decision.claimed_amount
            decision.approved_amount = decision.held_amount = Decimal("0")
            for line in decision.lines:
                line.approved, line.held = Decimal("0"), Decimal("0")
                line.rejected = line.claimed
                line.disposition = Disposition.REJECTED

        elif review.action is ReviewAction.AMEND:
            target = review.approved_amount
            if target is None:
                raise ValueError("AMEND requires approved_amount")
            target = max(Decimal("0"), min(target, decision.claimed_amount))
            # The reviewer sets a claim total; distribute it across the lines pro
            # rata so the line-level and claim-level figures cannot disagree.
            running = Decimal("0")
            for i, line in enumerate(decision.lines):
                if i == len(decision.lines) - 1:
                    approved = target - running          # absorb the rounding here
                else:
                    approved = (line.claimed * target / decision.claimed_amount
                                ).quantize(Decimal("0.01"))
                    running += approved
                line.approved = approved
                line.rejected = line.claimed - approved
                line.held = Decimal("0")
                line.disposition = (Disposition.APPROVED if line.rejected == 0
                                    else Disposition.REJECTED if approved == 0
                                    else Disposition.REDUCED)
                line.reason = "amended by reviewer (pro rata)"
            decision.approved_amount = target
            decision.held_amount = Decimal("0")
            decision.rejected_amount = decision.claimed_amount - target
            decision.decision = (Outcome.APPROVE if target == decision.claimed_amount
                                 else Outcome.REJECT if target == 0
                                 else Outcome.PARTIALLY_APPROVE)

        else:                                   # APPROVE the assessment as it stands
            released = decision.held_amount
            decision.approved_amount += released
            decision.held_amount = Decimal("0")
            for line in decision.lines:
                if line.held > 0:
                    line.approved, line.held = line.claimed - line.rejected, Decimal("0")
                    line.disposition = (Disposition.APPROVED if line.rejected == 0
                                        else Disposition.REDUCED)
                    line.reason = "released by reviewer: " + (review.note or "approved")
            decision.decision = (
                Outcome.APPROVE if decision.rejected_amount == 0
                else Outcome.REJECT if decision.approved_amount == 0
                else Outcome.PARTIALLY_APPROVE)

        return ("%s by %s - approved INR %s of INR %s%s"
                % (review.action.value, review.reviewer, decision.approved_amount,
                   decision.claimed_amount,
                   "; " + review.note if review.note else ""))

    def n_reject_intake(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        claim = state["claim"]
        decision = Decision(
            claim_id=claim.claim_id, employee_id=claim.employee_id,
            decision=Outcome.REJECT, claimed_amount=claim.claimed_total,
            approved_amount=claim.claimed_total * 0, rejected_amount=claim.claimed_total,
            held_amount=claim.claimed_total * 0,
            review_triggers=state.get("triggers", []), confidence=0.99)
        decision.explanation = ("This claim could not be accepted: "
                               + "; ".join(t.detail for t in decision.review_triggers))
        return {"result": decision,
                "trace": self._step(state, "reject_intake",
                                    "rejected before any model call", t0)}

    def n_audit(self, state: ClaimState) -> ClaimState:
        t0 = time.perf_counter()
        decision = state["result"]
        record = self.audit.write(decision, state.get("trace", []),
                                  "TRV-POL v1.0", self.backend.name)
        decision.audit = {
            "correlation_id": state.get("correlation_id", ""),
            "policy_version": "TRV-POL v1.0",
            "workflow_version": record.workflow_version,
            "model_backend": record.model_backend,
            "trace_digest": record.digest,
        }
        return {"result": decision,
                "trace": self._step(state, "audit",
                                    "trace written, digest %s" % record.digest, t0)}

    # --------------------------------------------------------------- run

    def submit(self, claim: Claim) -> Tuple[Decision, str]:
        """Run a claim until it completes, or until it needs a human.

        Returns (decision, status) where status is COMPLETE or PENDING_REVIEW.
        A PENDING_REVIEW claim is checkpointed mid-graph: nothing has settled,
        and it stays that way until resume() is called.
        """
        self._threads[claim.claim_id] = claim
        config = self._config(claim.claim_id)
        self.graph.invoke({"claim": claim, "trace": [], "triggers": []}, config)
        return self._snapshot(claim.claim_id)

    def resume(self, claim_id: str, outcome: ReviewOutcome) -> Decision:
        """Continue a suspended claim with a human's decision."""
        config = self._config(claim_id)
        snapshot = self.graph.get_state(config)
        if not snapshot.next:
            raise ValueError("%s is not awaiting review" % claim_id)
        if not outcome.decided_at:
            outcome.decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.graph.update_state(config, {"review": outcome})
        self.graph.invoke(None, config)
        return self._snapshot(claim_id)[0]

    def pending(self) -> List[Dict[str, object]]:
        """Every claim currently waiting on a person, and what it is waiting for."""
        waiting = []
        for claim_id in self._threads:
            snapshot = self.graph.get_state(self._config(claim_id))
            if not snapshot.next:
                continue
            decision = snapshot.values["result"]
            waiting.append({
                "claim_id": claim_id,
                "waiting_at": snapshot.next[0],
                "approver": decision.approval.tier if decision.approval else "FINANCE",
                "claimed": decision.claimed_amount,
                "assessed": decision.approved_amount,
                "held": decision.held_amount,
                "triggers": sorted({t.code.value for t in decision.review_triggers}),
            })
        return waiting

    def trace(self, claim_id: str) -> List[TraceStep]:
        return self.graph.get_state(self._config(claim_id)).values.get("trace", [])

    @staticmethod
    def _config(claim_id: str) -> dict:
        return {"configurable": {"thread_id": claim_id}}

    def _snapshot(self, claim_id: str) -> Tuple[Decision, str]:
        snapshot = self.graph.get_state(self._config(claim_id))
        status = "PENDING_REVIEW" if snapshot.next else "COMPLETE"
        return snapshot.values["result"], status

    # -- kept for batch use: run a claim and take whatever state it reaches

    def run(self, claim: Claim) -> Decision:
        return self.submit(claim)[0]

    def run_with_trace(self, claim: Claim) -> Tuple[Decision, List[TraceStep]]:
        decision, _ = self.submit(claim)
        return decision, self.trace(claim.claim_id)
