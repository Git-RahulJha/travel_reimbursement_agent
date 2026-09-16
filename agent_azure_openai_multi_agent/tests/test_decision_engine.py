from agent_azure_openai.data_loader import ClaimRepository
#from services.policy_service import get_policy_rules
from ..rag.rag_retrieval import policy_rules
from ..services.decision_engine import DecisionEngine
from ..models.receipts import ParsedReceipt


# --------------------------------
# 1. Load claim
# --------------------------------

claim_repo = ClaimRepository()

claim = claim_repo.claim("CLM-002")

print(f"Claim: {claim.claim_id}")
print(f"Employee: {claim.employee_name}")


# --------------------------------
# 2. Get policy from RAG
# --------------------------------

policy = policy_rules(
    "flight hotel meals taxi reimbursement policy"
)

print("\nPolicy loaded from RAG")


# --------------------------------
# 3. Temporary parsed receipts
# --------------------------------

receipts = [
    ParsedReceipt(
        receipt_id="R-2001",
        merchant="IndiGo",
        receipt_date="2026-08-18",
        amount=6400,
        currency="INR",
        category="FLIGHT",
        fare_class="ECONOMY",
    ),

    ParsedReceipt(
        receipt_id="R-2002",
        merchant="Lemon Tree Premier Bengaluru",
        receipt_date="2026-08-18",
        amount=11600,
        currency="INR",
        category="HOTEL",
        room_type="STANDARD",
        nights=1,
    ),
]


# --------------------------------
# 4. Run Decision Engine
# --------------------------------

engine = DecisionEngine()

decision = engine.evaluate(
    claim=claim,
    receipts=receipts,
    policy=policy,
)


# --------------------------------
# 5. Display result
# --------------------------------

print("\n========== DECISION ==========")

print(f"Claim ID       : {decision.claim_id}")
print(f"Claimed Amount : ₹{decision.claimed_amount:.2f}")
print(f"Approved Amount: ₹{decision.approved_amount:.2f}")
print(f"Rejected Amount: ₹{decision.rejected_amount:.2f}")
print(f"Status         : {decision.status}")
print(f"Approval       : {decision.approval_authority}")


print("\n========== LINE DECISIONS ==========")

for line in decision.line_decisions:

    print(f"\nLine       : {line.line_id}")
    print(f"Category   : {line.category}")
    print(f"Claimed    : ₹{line.claimed_amount:.2f}")
    print(f"Approved   : ₹{line.approved_amount:.2f}")
    print(f"Rejected   : ₹{line.rejected_amount:.2f}")
    print(f"Status     : {line.status}")
    print(f"Receipt    : {line.receipt_present}")
    print(f"Reason     : {line.reason}")