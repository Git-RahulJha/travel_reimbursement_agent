from typing import TypedDict

from agent_azure_openai_multi_agent.models.claim import Claim
from agent_azure_openai_multi_agent.models.investigation import InvestigationResult
from agent_azure_openai_multi_agent.models.receipts import ParsedReceipt, RawReceipt
from agent_azure_openai_multi_agent.models.policy import PolicyRules
from agent_azure_openai_multi_agent.models.decision import ClaimDecision


class ClaimReviewState(TypedDict, total=False):

    claim_id: str
    claim: Claim
    raw_receipts: list[RawReceipt]
    receipts: list[ParsedReceipt]
    policy: PolicyRules
    decision: ClaimDecision
    investigation: InvestigationResult | None