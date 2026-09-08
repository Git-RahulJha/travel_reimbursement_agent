# travel_reimbursement_agent
Travel reimbursement agent using LangChain and LangGraph
A working prototype for the architecture described in
`../Travel_Reimbursement_Approval_Agent_Architecture_Note.pdf`.

The design in one line: **a language model reads documents and writes sentences;
deterministic code owns every rupee.** The package is organised so this is
enforced rather than asserted — `agent/rules.py` imports no model client and
opens no socket, and nothing else is allowed to produce an approved amount.

---

## Running it

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

### Supply an OpenAI key

The pipeline uses **OpenAI for both halves of the RAG loop** — embeddings for
retrieval, and a chat model for the three language tasks. Create a `.env` file
in this folder:

```
OPENAI_API_KEY=sk-...
```

`agent/load_env()` reads it on import. An already-exported environment variable
always wins, the file is never written or logged, and it must stay out of
version control.

```bash
python run_demo.py                    # all six sample claims, one line each
python run_demo.py CLM-002            # one claim in full: trace, lines, evidence
python run_demo.py --rules            # the rules read out of the policy PDF
python run_demo.py --pending          # claims suspended, waiting on a human
python run_demo.py --graph            # the workflow graph, printed from LangGraph
python run_demo.py --fail duplicate   # inject a ledger outage and watch it escalate
python run_demo.py --offline          # no key needed - deterministic fallback

# act as the reviewer on a suspended claim
python run_demo.py --review CLM-005 --action approve --reviewer "P. Menon"                    --note "conference fee is a valid business expense"
python run_demo.py --review CLM-006 --action amend --amount 6000 --reviewer "A. Rao"
python run_demo.py --review CLM-006 --action reject --reviewer "A. Rao"
```

The run banner states exactly which models are in play, so there is never any
doubt about whether a real model was called:

```
policy corpus : TRV-POL_v1.0.pdf
                16 clauses -> 16 chunks; priced: FLT-001, HTL-001, MEL-001, TXI-001
embeddings    : openai:text-embedding-3-small
chat model    : openai:gpt-5-mini
```

### Models used

| Where | Model | Bound to |
|---|---|---|
| Policy retrieval | `text-embedding-3-small` | FAISS index over 16 clause chunks |
| Receipt extraction | `gpt-5-mini` | `ReceiptFields` schema |
| Item classification | `gpt-5-mini` | `Exclusions` schema |
| Claimant explanation | `gpt-5-mini` | free text, from the decision object only |

Temperature is zero throughout, and every structured call is bound to a Pydantic
schema with `with_structured_output`, so a response that does not fit the
contract raises rather than reaching the rule engine.

**The model never sees a Decimal.** The schemas it is bound to use plain JSON
types; values become Decimal on the way back, before any arithmetic happens.

### Without a key

`--offline`, or simply no key, swaps in a TF-IDF embedding and a deterministic
backend. Every other line of the pipeline is unchanged, because both are
LangChain `Embeddings` / backend implementations. Useful for demoing on a
machine with no network, and for reproducible test runs.

---

## Layout

```
reimbursement_demo/
  agent/
    contracts.py    Pydantic contracts - Claim, Decision, LineAssessment, Trigger
    knowledge.py    policy corpus: PDF loader, embeddings, FAISS index, retrieval
    llm.py          the three model tasks; offline and OpenAI backends
    services.py     evidence, duplicate, approval-matrix services; tool contracts; audit
    rules.py        the deterministic core - owns every rupee
    workflow.py     the LangGraph state machine
    repository.py   loads the sample claims, receipts and ledger
  data/
    claims.json     six claims, one per decision path
    receipts.json   seventeen receipt documents as OCR text
    ledger.json     settled claim lines, for the duplicate check
  policies/
    TRV-POL_v1.0.pdf     the knowledge base - and the only source of the limits
  run_demo.py
  requirements.txt
  .env            your OPENAI_API_KEY - create this, do not commit it
```

**The knowledge base is the policy PDF.** `PdfClauseLoader` extracts it, finds
each `<ID> — <title>` heading and the `category: ... | type: ...` line beneath
it, and produces one chunk per clause carrying that metadata. A
`RecursiveCharacterTextSplitter` splits any clause too long to embed well; every
sub-chunk keeps the full clause metadata, so a hit on any part of a clause still
cites the clause.

---

## What runs where

| Component | Kind | What it does |
|---|---|---|
| `PolicyKnowledgeBase` | model-backed | Loads 16 clauses from the PDF, embeds them with OpenAI, indexes in FAISS, retrieves filtered by category and effective date |
| `PolicyRuleResolver` | model-backed | Reads the limit out of the retrieved clause and verifies it appears in that clause text |
| `ReceiptEvidenceService` | model-backed | Reads a receipt into structured fields with a calibrated confidence |
| `RuleEngine` | deterministic | Four gates, then the category rule. Applies the retrieved limit; holds no policy figure of its own |
| `DuplicateCheckService` | deterministic | DUP-001 key match against the ledger |
| `ApprovalMatrixService` | deterministic | APR-001 tier and the AGT-001 delegation |
| `DecisionEngine` | deterministic | Aggregates lines into one outcome |
| `SafetyGate` | deterministic | Reads open triggers and the delegation. A model has no input here |
| `AuditStore` | deterministic | Trace, clause citations, versions, reproducible digest |

Two containment rules make the split real. A model output can only **reduce** an
approved amount or **hold** a line — it can never create entitlement. And no
model emits a number that reaches the ledger: models return field values,
categories and confidence, and the engine does the arithmetic.

---

## The graph

Eight nodes, two conditional edges, four outcomes. `python run_demo.py --graph`
prints this from the graph that actually executes, so it cannot drift from the
code.

```
__start__ --> intake
intake       -. proceed  .-> evidence
intake       -. reject   .-> reject_intake
evidence  --> retrieval --> validation --> tools --> decision --> safety_gate
safety_gate  -. finalise .-> finalise
safety_gate  -. escalate .-> manual_review
finalise, manual_review, reject_intake --> audit --> __end__
```

Both conditional edges matter. The first means a malformed claim is rejected
before any model call. The second is the governance boundary: the gate reads
open triggers and the delegation, and nothing talks it round.

---

## The limits come from the PDF, not from the code

This is where RAG stops being decorative. There is no table of amounts in
`rules.py`. When a claim line arrives, the retriever finds the clause that
governs it, and the limit is **read out of that clause's text**:

```
python run_demo.py --rules

CLAUSE       LIMIT  BASIS            VERIFIED  READ FROM THE CLAUSE           BARRED
FLT-001     15,000  per_sector       yes       INR 15,000 per one-way sector  BUSINESS_CLASS, FIRST_CLASS, ...
HTL-001      8,000  per_night        yes       INR 8,000 per night            SUITE, CLUB_FLOOR, ...
MEL-001      1,500  per_travel_day   yes       INR 1,500 per travel day       -
TXI-001      2,000  per_travel_day   yes       INR 2,000 per travel day       SIGHTSEEING, LEISURE, ...
RCP-001        500  threshold        yes       above INR 500                  -
AGT-001     25,000  per_claim        yes       INR 25,000                     -
INC-001          -  none             NO        -                              -
```

The receipt threshold, the agent's delegation and the whole approval ladder are
read the same way. `EngineParameters` in `rules.py` holds exactly two numbers,
and neither is policy: the extraction confidence floor and the receipt/claim
mismatch tolerance.

### What stops the model inventing a limit

`verified` is the guard. A figure may price a line only if it appears **verbatim
in the clause it was read from** — `PolicyRuleResolver._appears_in()`
[knowledge.py](agent/knowledge.py). An unverified rule prices nothing and the
line is held for a human. There is no fallback constant to fall back to.

So the honest statement of the design is: the model *reads* a number that is
written in the document, and the engine does every calculation with it. It never
decides an amount.

### The demonstration

Change one word in the policy and the payout changes, with no code edit:

```
policy as written  — HTL-001 says INR 8,000 per night
  hotel line: claimed 9,500 -> approved 8,000, rejected 1,500
  claim total approved: 17,450

same code, policy edited to INR 6,000 per night
  hotel line: claimed 9,500 -> approved 6,000, rejected 3,500
  claim total approved: 15,450
```

That is what makes this a policy-driven system rather than a hardcoded one:
Finance edits the document, not the codebase.

---

## Human in the loop

This is a real interrupt, not a flag. The graph is compiled with a
`MemorySaver` checkpointer and `interrupt_before=["manual_review"]`, so when the
safety gate escalates, **execution suspends with its state checkpointed and
nothing settles**. The claim waits until a person acts.

```
python run_demo.py --pending

CLAIM     SUSPENDED AT   APPROVER                  CLAIMED     ASSESSED         HELD  WAITING ON
CLM-004   manual_review  SENIOR_MANAGER          45,100.00    45,100.00         0.00  ABOVE_AGENT_DELEGATION
CLM-005   manual_review  REPORTING_MANAGER       17,900.00     5,900.00    12,000.00  NO_PRICING_CLAUSE
CLM-006   manual_review  REPORTING_MANAGER        9,650.00       450.00     9,200.00  EVIDENCE_UNREADABLE, ...
```

Three claims stopped mid-graph. `submit()` returned `PENDING_REVIEW`; the
`manual_review` node has not run.

A reviewer resumes it with one of three actions — `ReviewOutcome`
[contracts.py](agent/contracts.py):

| Action | Effect |
|---|---|
| `approve` | Releases the **held** bucket. The agent declined to assess it; a human approval is precisely the act of releasing it. |
| `amend` | The reviewer sets the settled total; it is distributed across the lines pro rata so line and claim figures cannot disagree. |
| `reject` | Refuses the claim in full. |

The outcome is applied in `_apply_review()` [workflow.py](agent/workflow.py) and
recorded on the decision as `review` — who decided, what they chose, when, and
their note. It lands in the audit trace as step 8:

```
[7] safety_gate    escalated: NO_PRICING_CLAUSE
[8] manual_review  APPROVE by P. Menon (Reporting Manager) - approved INR 17900
                   of INR 17900; conference fee is a valid business expense
[9] audit          trace written, digest 12e09733621e6105
```

The invariant survives every path: `approved + rejected + held == claimed`, and
the line totals reconcile to the claim totals. The runner checks both.

---

## The sample set

Six claims, chosen to exercise every decision path. Amounts in INR.

| Claim | Decision | Claimed | Approved | Rejected | Held | Why |
|---|---|---:|---:|---:|---:|---|
| CLM-001 | APPROVE | 24,000 | 24,000 | 0 | 0 | Clean claim inside the delegation |
| CLM-002 | PARTIALLY_APPROVE | 20,840 | 17,450 | 3,390 | 0 | Hotel over the nightly cap, alcohol on a meal receipt, a taxi line with no receipt |
| CLM-003 | REJECT | 8,000 | 0 | 8,000 | 0 | Both lines already settled on claim CLM-0912 |
| CLM-004 | MANUAL_REVIEW | 45,100 | 45,100 | 0 | 0 | Fully assessed, escalated only because it exceeds AGT-001 |
| CLM-005 | MANUAL_REVIEW | 17,900 | 5,900 | 0 | 12,000 | A conference fee no clause prices — held, not guessed at |
| CLM-006 | MANUAL_REVIEW | 9,650 | 450 | 0 | 9,200 | An unreadable invoice and a receipt that disagrees with the claim |

`approved + rejected + held == claimed` holds on every line and every claim, and
CLM-004, CLM-005 and CLM-006 suspend awaiting a human rather than settling. The
runner checks and prints both.

---

## Where to look first

If you have five minutes and want to judge the design rather than the plumbing:

| Read | For |
|---|---|
| `rules.py`, `assess_line()` | The four gates every line passes before any category rule runs, and the three-bucket result |
| `rules.py`, `_gate_authority()` | Why a topically relevant clause is not authority to pay — only a *priced, verified* rule is |
| `knowledge.py`, `PolicyRuleResolver` | Where the limits actually come from, and the verbatim check that guards them |
| `workflow.py`, `_build()` | The graph, its two conditional edges, and where a model is and is not invoked |
| `llm.py`, `ModelBackend.guard()` | Schema validation, one repair retry, then escalation — why no malformed response becomes a payment |
| `rules.py`, `SafetyGate.evaluate()` | The last deterministic check, which reads triggers and the delegation and nothing else |
| `workflow.py`, `_build()` + `resume()` | The real interrupt: checkpointed suspension, and how a human decision re-enters the graph |
| `services.py`, `ToolContract` | Every external call with its timeout, retries and stated failure behaviour |

### Two things worth running

**The trap fires.** `python run_demo.py CLM-005` — the ₹12,000 conference fee
retrieves INC-001, a genuine topical hit. The trace shows
`L-2 -> INC-001 (context, 0.24)`. Because the clause is `context` and not
`priced`, the amount is **held** rather than paid. A naive pipeline pays it.

**Outages escalate, never approve.** `python run_demo.py --fail duplicate` fails
the claims ledger past its retries. All six claims become MANUAL_REVIEW —
including CLM-003, the duplicate, which escalates rather than being silently
paid. There is no path in this codebase that reads "approve anyway".

---

## Caveats

Everything below the tool-contract boundary is mocked: HRIS, the travel booking
system, the settled-claim ledger, OCR, ERP posting and the human review queue.
The policy document is fictional but realistic, and the receipts are OCR text
rather than images.

The offline fallback embedding is TF-IDF over the corpus vocabulary — real
vector similarity, but lexical rather than neural, so retrieval scores under
`--offline` are much lower than the OpenAI embeddings give. It is a LangChain
`Embeddings` implementation, so the vector store code is identical either way.

What is real is the retrieval and its filtering, the entitlement arithmetic, the
orchestration graph and both its branches, the schema-guarded model I/O, the tool
contracts with fault injection, and the audit trace — which is where the
architectural claims actually live.

