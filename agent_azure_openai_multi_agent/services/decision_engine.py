from agent_azure_openai_multi_agent.models.claim import Claim
from agent_azure_openai_multi_agent.models.receipts import ParsedReceipt
from agent_azure_openai_multi_agent.models.policy import PolicyRules
from agent_azure_openai_multi_agent.models.decision import ClaimDecision, LineDecision
from agent_azure_openai_multi_agent.services.policy_service import find_expense_rule


class DecisionEngine:

    def evaluate(self, claim: Claim, receipts: list[ParsedReceipt], 
                 policy: PolicyRules) -> ClaimDecision:
        print("Evaluating claim..")
        print("Claim:", claim)
        print("Receipts:", receipts)
        print("Policy:", policy)
        line_decisions = []

        for line in claim.lines:
            rule = find_expense_rule(policy,line.category)

            if rule is None:
                decision = LineDecision(
                    line_id=line.line_id,
                    category=line.category,
                    claimed_amount=line.amount,
                    approved_amount=0,
                    rejected_amount=line.amount,
                    status="MANUAL_REVIEW",
                    reason="No applicable policy rule found.",
                    receipt_present=False
                )

                line_decisions.append(decision)
                continue

            receipt = self._find_receipt(
                line.receipt_id,
                receipts
            )

            receipt_present = receipt is not None

            # Rule 1: category not eligible
            if not rule.eligible:

                line_decisions.append(
                    LineDecision(
                        line_id=line.line_id,
                        category=line.category,
                        claimed_amount=line.amount,
                        approved_amount=0,
                        rejected_amount=line.amount,
                        status="REJECTED",
                        reason="Expense category is not eligible.",
                        receipt_present=receipt_present
                    )
                )

                continue

            # Rule 2: receipt required
            if rule.receipt_required and not receipt_present:

                line_decisions.append(
                    LineDecision(
                        line_id=line.line_id,
                        category=line.category,
                        claimed_amount=line.amount,
                        approved_amount=0,
                        rejected_amount=line.amount,
                        status="MANUAL_REVIEW",
                        reason="Required receipt is missing.",
                        receipt_present=False
                    )
                )

                continue

            # Rule 3: calculate allowed amount
            approved_amount = self._calculate_allowed_amount(
                line.amount,
                line,
                rule
            )

            rejected_amount = max(
                line.amount - approved_amount,
                0
            )

            if rejected_amount == 0:
                status = "APPROVED"
            else:
                status = "PARTIALLY_APPROVED"

            line_decisions.append(
                LineDecision(
                    line_id=line.line_id,
                    category=line.category,
                    claimed_amount=line.amount,
                    approved_amount=approved_amount,
                    rejected_amount=rejected_amount,
                    status=status,
                    reason=self._build_reason(
                        line.amount,
                        approved_amount,
                        rule
                    ),
                    receipt_present=receipt_present
                )
            )

        return self._build_claim_decision(
            claim,
            line_decisions,
            policy
        )

    def _find_receipt(self, receipt_id: str | None,
                    receipts: list[ParsedReceipt]) -> ParsedReceipt | None:

        if not receipt_id:
            return None

        for receipt in receipts:
            if receipt.receipt_id == receipt_id:
                return receipt

        return None

    def _calculate_allowed_amount(self, claimed_amount: float, line, rule) -> float:

        if rule.max_amount is None:
            return claimed_amount

        unit = (rule.max_amount_unit or "").lower()

        if "per night" in unit:
            nights = line.nights or 1
            limit = rule.max_amount * nights
            return min(
                claimed_amount,
                limit
            )

        if "per travel day" in unit:
            days = line.travel_days or 1
            limit = rule.max_amount * days
            return min(
                claimed_amount,
                limit
            )

        if "per flight" in unit:
            sectors = line.sectors or 1
            limit = rule.max_amount * sectors
            return min(
                claimed_amount,
                limit
            )

        # Unknown unit → don't silently calculate
        return claimed_amount

    def _build_reason(self, claimed_amount: float, approved_amount: float, rule) -> str:

        if claimed_amount == approved_amount:
            return "Expense is within the applicable policy limit."

        return (
            f"Claimed amount exceeds the applicable policy limit. "
            f"Approved amount: INR {approved_amount:.2f}."
        )

    def _build_claim_decision(self, claim: Claim, line_decisions: list[LineDecision],
        policy: PolicyRules) -> ClaimDecision:

        claimed_amount = sum(d.claimed_amount for d in line_decisions)
        approved_amount = sum(d.approved_amount for d in line_decisions)
        rejected_amount = sum(d.rejected_amount for d in line_decisions)

        if any( d.status == "MANUAL_REVIEW" for d in line_decisions):
            status = "MANUAL_REVIEW"

        elif approved_amount == 0:
            status = "REJECTED"

        elif rejected_amount == 0:
            status = "APPROVED"

        else:
            status = "PARTIALLY_APPROVED"

        approval_authority = self._find_approval_authority(approved_amount, policy)

        return ClaimDecision(
            claim_id=claim.claim_id,
            claimed_amount=claimed_amount,
            approved_amount=approved_amount,
            rejected_amount=rejected_amount,
            status=status,
            approval_authority=approval_authority,
            line_decisions=line_decisions
        )

    def _find_approval_authority(self, amount: float, policy: PolicyRules) -> str | None:

        for rule in policy.approval_rules:
            minimum_ok = (rule.min_amount is None or amount >= rule.min_amount)
            maximum_ok = (rule.max_amount is None or amount <= rule.max_amount)

            if minimum_ok and maximum_ok:
                return rule.authority

        return None