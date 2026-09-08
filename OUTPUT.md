In LangGraph, you can add initial entry point and then simple and conditional edges. Like here in below example, there are nine edges started from intake. At every edge, you can define next simple proceeding edge or conditional edge.

===========================================
python run_demo.py --rules

============================================================================================
CLAUSE           LIMIT  BASIS            VERIFIED  READ FROM THE CLAUSE           BARRED
============================================================================================
AGT-001         25,000  per_claim        yes       INR 25,000,                    -
APR-001         25,000  per_claim        yes       INR 25,000                     -
FLT-001         15,000  per_sector       yes       INR 15,000 per one-way sector  BUSINESS_CLASS, FIRST_CLASS, PREMIUM-ECONOMY
HTL-001          8,000  per_night        yes       INR 8,000 per night            SUITE, CLUB_FLOOR, EXECUTIVE_FLOOR
INC-001              -  none             NO        -                              -
MEL-001          1,500  per_travel_day   yes       INR 1,500 per travel day       -
RCP-001            500  threshold        yes       above INR 500                  -
TXI-001          2,000  per_travel_day   yes       INR 2,000 per travel day       SIGHTSEEING, LEISURE, PERSONAL_ERRAND
============================================================================================
approval ladder read from APR-001: up to 25,000 -> REPORTING_MANAGER  up to 50,000 -> SENIOR_MANAGER  above -> FINANCE_CONTROLLER
Not one of these figures is a constant in the code. Verified means the
number was found verbatim in the clause it was read from; an unverified
rule prices nothing and the line is held.


===============================================
Claims with approval status:

recommend Approve	Done	CLM-001
Partially Approve	Done	CLM-002 — 17,450 of 20,840
Reject	Done	CLM-003 — duplicate
Manual Review	Done	CLM-004, 005, 006 — three different reasons
with evidence	Done	Clause ID per line, retrieved clause list with source + effective date, 9-step trace



python run_demo.py CLM-002

policy corpus : TRV-POL_v1.0.pdf
                16 clauses -> 16 chunks; priced: FLT-001, HTL-001, MEL-001, TXI-001
embeddings    : tfidf-offline
chat model    : offline-deterministic
                (no OPENAI_API_KEY found - set one, or create a .env file, to use OpenAI)

============================================================================================
CLM-002   PARTIALLY_APPROVE
============================================================================================

TRACE
  [1] intake            0 ms  CLM-002: 4 line(s), Anita Rao, trip TRP-9014 2026-08-18..2026-08-20
  [2] evidence          0 ms  extraction confidence - L-1:0.95, L-2:0.94, L-3:0.92
  [3] retrieval         0 ms  L-1 -> FLT-001 (priced, 0.00); L-2 -> HTL-001 (priced, 0.14); L-3 -> MEL-001 (priced, 0.00); L-4 -> TXI-001 (priced, 0.16)
  [4] validation        0 ms  assessed 4 line(s), approvable INR 17450.00
  [5] tools             0 ms  0 duplicate(s); approver REPORTING_MANAGER
  [6] decision          0 ms  PARTIALLY_APPROVE - claimed 20840, approved 17450.00, rejected 3390.00, held 0
  [7] safety_gate       0 ms  finalised within the agent's delegation
  [8] finalise          0 ms  decision final; explanation generated from the decision object
  [9] audit             0 ms  trace written, digest d5de28b7cc40b6f3

LINES
  LINE   CATEGORY         CLAIMED     APPROVED     REJECTED         HELD  CLAUSE     REASON
  --------------------------------------------------------------------------------------------
  L-1    FLIGHT          6,400.00     6,400.00         0.00         0.00  FLT-001    within policy
  L-2    HOTEL           9,500.00     8,000.00     1,500.00         0.00  HTL-001    tariff above the nightly limit
  L-3    MEALS           4,300.00     3,050.00     1,250.00         0.00  MEL-001    non-reimbursable items removed
  L-4    TAXI              640.00         0.00       640.00         0.00  RCP-001    receipt missing above the INR 500 threshold
  --------------------------------------------------------------------------------------------
  TOTAL                 20,840.00    17,450.00     3,390.00         0.00
  balances (approved + rejected + held == claimed): yes

MISSING DOCUMENTS
  L-4    TAX_INVOICE    RCP-001

POLICY EVIDENCE
  FLT-001   Air travel                               priced      TRV-POL, effective 2026-04-01
  HTL-001   Hotel accommodation                      priced      TRV-POL, effective 2026-04-01
  MEL-001   Meals                                    priced      TRV-POL, effective 2026-04-01
  TXI-001   Local transport                          priced      TRV-POL, effective 2026-04-01

OPEN TRIGGERS
  MEDIUM   RECEIPT_MISSING            RCP-001   receipt missing above the INR 500 threshold

APPROVAL
  tier REPORTING_MANAGER, within the agent's delegation: True
  confidence 0.87   manual review: False

EXPLANATION TO THE CLAIMANT
  Your claim has been partly approved. Of the INR 20,840.00 claimed, INR 17,450.00 is approved. Reduced: hotel INR 1,500.00 (tariff above the nightly limit, HTL-001); meals INR 1,250.00 (non-reimbursable items removed, MEL-001); taxi INR 640.00 (receipt missing above the INR 500 threshold, RCP-001). Please supply: tax invoice for line L-4.

AUDIT
  policy_version     TRV-POL v1.0
  workflow_version   1.0.0
  model_backend      offline-deterministic
  trace_digest       d5de28b7cc40b6f3


=====================================================
