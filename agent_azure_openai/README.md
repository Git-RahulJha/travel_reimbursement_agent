Agents orchestrate and reason; tools provide capabilities; deterministic services enforce critical business rules.

Run this command to run the test
python -m agent_azure_openai.tests.test_decision_engine

Output:
Claim: CLM-002
Employee: Anita Rao
Number of vectors: 29

Policy loaded from RAG

========== DECISION ==========
Claim ID       : CLM-002
Claimed Amount : ₹20840.00
Approved Amount: ₹18700.00
Rejected Amount: ₹2140.00
Status         : MANUAL_REVIEW
Approval       : None

========== LINE DECISIONS ==========

Line       : L-1
Category   : FLIGHT
Claimed    : ₹6400.00
Approved   : ₹6400.00
Rejected   : ₹0.00
Status     : APPROVED
Receipt    : True
Reason     : Expense is within the applicable policy limit.

Line       : L-2
Category   : HOTEL
Claimed    : ₹9500.00
Approved   : ₹8000.00
Rejected   : ₹1500.00
Status     : PARTIALLY_APPROVED
Receipt    : True
Reason     : Claimed amount exceeds the applicable policy limit. Approved amount: INR 8000.00.

Line       : L-3
Category   : MEALS
Claimed    : ₹4300.00
Approved   : ₹4300.00
Rejected   : ₹0.00
Status     : APPROVED
Receipt    : False
Reason     : Expense is within the applicable policy limit.

Line       : L-4
Category   : TAXI
Claimed    : ₹640.00
Approved   : ₹0.00
Rejected   : ₹640.00
Status     : MANUAL_REVIEW
Receipt    : False
Reason     : No applicable policy rule found.

===================================
