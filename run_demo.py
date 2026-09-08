# -*- coding: utf-8 -*-
"""Run the Travel Reimbursement Approval Agent over the sample claims.

    python run_demo.py                 all six claims, one summary line each
    python run_demo.py CLM-002         one claim, in full detail
    python run_demo.py --pending       claims suspended, waiting on a human
    python run_demo.py --review CLM-005 --action approve --reviewer "P. Menon"
                                       act as the reviewer and let the claim finish
    python run_demo.py --rules         show the rules read out of the policy PDF
    python run_demo.py --graph         print the workflow graph from LangGraph
    python run_demo.py --fail evidence inject a dependency outage and watch it escalate
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal

from agent import ClaimRepository, PolicyKnowledgeBase, ReimbursementAgent, build_backend
from agent.contracts import Decision, Outcome, ReviewAction, ReviewOutcome

RULE = "=" * 92
THIN = "-" * 92


def money(value: Decimal) -> str:
    return "{:>12,.2f}".format(value)


def plain(value: Decimal) -> str:
    return "{:,.0f}".format(value)


def build(fail=None, hosted=True) -> ReimbursementAgent:
    repo = ClaimRepository()
    kb = PolicyKnowledgeBase(prefer_hosted=hosted)
    return ReimbursementAgent(knowledge=kb, receipts=repo.receipts(),
                              ledger=repo.ledger(), backend=build_backend(hosted),
                              fail=fail)


def show_detail(decision: Decision, trace) -> None:
    print(RULE)
    print("%s   %s" % (decision.claim_id, decision.decision.value))
    print(RULE)

    print("\nTRACE")
    for step in trace:
        print("  [%d] %-14s %4d ms  %s" % (step.seq, step.node, step.elapsed_ms,
                                           step.summary))

    print("\nLINES")
    print("  %-6s %-11s %12s %12s %12s %12s  %-10s %s"
          % ("LINE", "CATEGORY", "CLAIMED", "APPROVED", "REJECTED", "HELD",
             "CLAUSE", "REASON"))
    print("  " + THIN)
    for l in decision.lines:
        print("  %-6s %-11s %s %s %s %s  %-10s %s"
              % (l.line_id, l.category.value, money(l.claimed), money(l.approved),
                 money(l.rejected), money(l.held), l.clause or "-", l.reason))
    print("  " + THIN)
    print("  %-18s %s %s %s %s"
          % ("TOTAL", money(decision.claimed_amount), money(decision.approved_amount),
             money(decision.rejected_amount), money(decision.held_amount)))
    lines_ok = (sum(l.approved for l in decision.lines) == decision.approved_amount
                and sum(l.rejected for l in decision.lines) == decision.rejected_amount
                and sum(l.held for l in decision.lines) == decision.held_amount)
    print("  balances (approved + rejected + held == claimed): %s"
          % ("yes" if decision.balances() else "NO"))
    print("  line totals reconcile to the claim totals:        %s"
          % ("yes" if lines_ok else "NO"))

    if decision.missing_documents:
        print("\nMISSING DOCUMENTS")
        for m in decision.missing_documents:
            print("  %-6s %-14s %s" % (m.line_id, m.required, m.clause))

    if decision.evidence:
        print("\nPOLICY EVIDENCE")
        for c in decision.evidence:
            print("  %-9s %-40s %-11s %s, effective %s"
                  % (c.clause_id, c.title, c.clause_type, c.document, c.effective_from))

    if decision.review_triggers:
        print("\nOPEN TRIGGERS")
        for t in decision.review_triggers:
            print("  %-8s %-26s %-9s %s"
                  % (t.severity.value, t.code.value, t.clause, t.detail))

    print("\nAPPROVAL")
    if decision.approval:
        print("  tier %s, within the agent's delegation: %s"
              % (decision.approval.tier, decision.approval.within_agent_delegation))
    print("  confidence %.2f   manual review: %s" % (decision.confidence,
                                                     decision.manual_review))

    print("\nEXPLANATION TO THE CLAIMANT")
    print("  " + decision.explanation)

    print("\nAUDIT")
    for k, v in decision.audit.items():
        print("  %-18s %s" % (k, v))
    print()


def show_summary(rows) -> None:
    print(RULE)
    print("%-9s %-18s %-15s %12s %12s %12s %12s  %s"
          % ("CLAIM", "DECISION", "STATUS", "CLAIMED", "APPROVED", "REJECTED",
             "HELD", "OPEN TRIGGERS"))
    print(RULE)
    for d, status in rows:
        codes = sorted({t.code.value for t in d.review_triggers})
        print("%-9s %-18s %-15s %s %s %s %s  %s"
              % (d.claim_id, d.decision.value, status, money(d.claimed_amount),
                 money(d.approved_amount), money(d.rejected_amount),
                 money(d.held_amount), ", ".join(codes) if codes else "none"))
    print(RULE)
    total = len(rows)
    balanced = sum(1 for d, _ in rows if d.balances())
    waiting = sum(1 for _, st in rows if st == "PENDING_REVIEW")
    print("%d claims  |  %d/%d balance  |  %d suspended awaiting a human  |  %s"
          % (total, balanced, total, waiting,
             ", ".join("%s x%d" % (o.value, sum(1 for d, _ in rows if d.decision is o))
                       for o in Outcome
                       if any(d.decision is o for d, _ in rows))))


def show_rules(agent) -> None:
    """What the run actually read out of the policy document."""
    resolver = agent.resolver
    for clause_id in ("FLT-001", "HTL-001", "MEL-001", "TXI-001", "INC-001",
                      "RCP-001", "AGT-001", "APR-001"):
        resolver.by_clause(clause_id)
    print(RULE)
    print("%-9s %12s  %-16s %-9s %-30s %s"
          % ("CLAUSE", "LIMIT", "BASIS", "VERIFIED", "READ FROM THE CLAUSE", "BARRED"))
    print(RULE)
    for rule in resolver.summary():
        print("%-9s %12s  %-16s %-9s %-30s %s"
              % (rule.clause_id,
                 plain(rule.limit_amount) if rule.limit_amount is not None else "-",
                 rule.basis.value.lower(),
                 "yes" if rule.verified else "NO",
                 (rule.source_phrase or "-")[:30],
                 ", ".join(rule.ineligible_terms[:3]) or "-"))
    print(RULE)
    print("approval ladder read from APR-001: %s"
          % "  ".join("%s -> %s" % ("above" if c is None else "up to " + plain(c), a)
                      for c, a in resolver.approval_tiers()))
    print("Not one of these figures is a constant in the code. Verified means the")
    print("number was found verbatim in the clause it was read from; an unverified")
    print("rule prices nothing and the line is held.")


def show_pending(agent) -> None:
    waiting = agent.pending()
    if not waiting:
        print("Nothing is waiting on a human.")
        return
    print(RULE)
    print("%-9s %-14s %-20s %12s %12s %12s  %s"
          % ("CLAIM", "SUSPENDED AT", "APPROVER", "CLAIMED", "ASSESSED", "HELD",
             "WAITING ON"))
    print(RULE)
    for row in waiting:
        print("%-9s %-14s %-20s %s %s %s  %s"
              % (row["claim_id"], row["waiting_at"], row["approver"],
                 money(row["claimed"]), money(row["assessed"]), money(row["held"]),
                 ", ".join(row["triggers"])))
    print(RULE)
    print("These claims are checkpointed mid-graph. Nothing has settled.")


def review(agent, repo, claim_id, action, reviewer, amount, note) -> None:
    before, status = agent.submit(repo.claim(claim_id))
    print(RULE)
    print("%s  submitted  ->  %s" % (claim_id, status))
    print(RULE)
    if status != "PENDING_REVIEW":
        print("This claim settled without a human; nothing to review.")
        return
    print("  decision so far : %s" % before.decision.value)
    print("  assessed        : %s of %s claimed, %s held"
          % (money(before.approved_amount).strip(),
             money(before.claimed_amount).strip(), money(before.held_amount).strip()))
    print("  waiting on      : %s"
          % ", ".join(sorted({t.code.value for t in before.review_triggers})))
    print("  approver        : %s"
          % (before.approval.tier if before.approval else "FINANCE"))

    outcome = ReviewOutcome(reviewer=reviewer, action=ReviewAction(action.upper()),
                            approved_amount=amount, note=note)
    after = agent.resume(claim_id, outcome)

    print("\n%s  resumed by %s  ->  %s" % (claim_id, reviewer, after.decision.value))
    print(RULE)
    show_detail(after, agent.trace(claim_id))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("claim_id", nargs="?", help="run a single claim in detail")
    ap.add_argument("--graph", action="store_true", help="print the LangGraph graph")
    ap.add_argument("--fail", choices=["evidence", "duplicate", "approval"],
                    help="inject a dependency outage")
    ap.add_argument("--offline", action="store_true",
                    help="force the offline embedding and backend, ignoring any key")
    ap.add_argument("--rules", action="store_true",
                    help="show the rules read out of the policy document")
    ap.add_argument("--pending", action="store_true",
                    help="list claims suspended awaiting a human")
    ap.add_argument("--review", metavar="CLAIM_ID",
                    help="act as the reviewer on a suspended claim")
    ap.add_argument("--action", choices=["approve", "amend", "reject"],
                    default="approve", help="the reviewer's decision")
    ap.add_argument("--amount", type=Decimal, help="settled amount, for --action amend")
    ap.add_argument("--reviewer", default="Finance Reviewer")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    agent = build(fail=args.fail, hosted=not args.offline)
    repo = ClaimRepository()

    kb = agent.knowledge
    print("policy corpus : %s" % os.path.basename(kb.corpus_path))
    print("                %d clauses -> %d chunks; priced: %s"
          % (kb.clause_count, kb.chunk_count, ", ".join(kb.priced_clauses)))
    print("embeddings    : %s" % kb.embedding_name)
    print("chat model    : %s" % agent.backend.name)
    if agent.backend.name == "offline-deterministic":
        print("                (no OPENAI_API_KEY found - set one, or create a "
               ".env file, to use OpenAI)")
    if args.fail:
        print("fault injected: %s service unavailable" % args.fail)
    print()

    if args.graph:
        print(agent.mermaid())
        return 0

    if args.rules:
        show_rules(agent)
        return 0

    if args.review:
        review(agent, repo, args.review, args.action, args.reviewer,
               args.amount, args.note)
        return 0

    if args.claim_id:
        decision, trace = agent.run_with_trace(repo.claim(args.claim_id))
        show_detail(decision, trace)
        return 0

    rows = [agent.submit(c) for c in repo.claims()]
    show_summary(rows)
    if args.pending:
        print()
        show_pending(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
