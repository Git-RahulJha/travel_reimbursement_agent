from typing import TypedDict

from agent_azure_openai.models.claim import Claim
from agent_azure_openai.models.receipts import ParsedReceipt, RawReceipt
from agent_azure_openai.models.policy import PolicyRules
from agent_azure_openai.models.decision import ClaimDecision


class ClaimReviewState(TypedDict, total=False):

    claim_id: str
    claim: Claim
    raw_receipts: list[RawReceipt]
    receipts: list[ParsedReceipt]
    policy: PolicyRules
    decision: ClaimDecision