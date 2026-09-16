from langgraph.graph import StateGraph, START, END

from agent_azure_openai.models.graph_state import ClaimReviewState
from agent_azure_openai.rag.rag_retrieval import parse_receipt
from agent_azure_openai.models.receipts import ParsedReceipt
from agent_azure_openai.data_loader import ClaimRepository
from agent_azure_openai.services.policy_service import policy_rules
from agent_azure_openai.services.decision_engine import DecisionEngine

import agent_azure_openai.tools as tool

repo = ClaimRepository()
engine = DecisionEngine()

def load_claim(state: ClaimReviewState):
    print("Loading claim with ID:", state["claim_id"])
    claim = repo.claim(state["claim_id"])  
    print("Claim loaded:", claim)
    return { "claim": claim }

# This is raw receipts
def load_receipts(state: ClaimReviewState):
    print("Loading raw receipts..")
    claim = state["claim"]
    receipts = []

    for line in claim.lines:
        if not line.receipt_id:
            continue

        receipt = tool.get_raw_receipt.invoke({
            "receipt_id": line.receipt_id
        })
        print('raw receipt fetched:', receipt)

        receipts.append(receipt)

    return { "raw_receipts": receipts }

def parse_receipts(state: ClaimReviewState): 
    parsed_receipts = [] 
    for raw_receipt in state["raw_receipts"]: 
        parsed = parse_receipt( raw_receipt ) 
        parsed_receipts.append(parsed) 

    print("Parsed receipts:", parsed_receipts)
    return { "receipts": parsed_receipts }

def load_policy(state: ClaimReviewState):
    print("Loading policy rules from RAG..")
    policy = policy_rules(
        "travel reimbursement policy for flights hotels meals taxi"
    )
    print("Policy loaded from RAG:", policy)
    return { "policy": policy }

def make_decision(state: ClaimReviewState):
    print("Making decision based on claim, receipts, and policy..")
    decision = engine.evaluate(
        claim=state["claim"],
        receipts=state["receipts"],
        policy=state["policy"]
    )

    return { "decision": decision }

# This function is called when both branches (receipts and policy) 
# are completed. It guarantees that the state has been populated by both branches
def synchronize_inputs(state: ClaimReviewState):
    print("Both receipt and policy branches completed.")
    return {}

def build_graph():
    builder = StateGraph(ClaimReviewState)

    # Add nodes to the graph
    builder.add_node("load_claim", load_claim)
    builder.add_node("load_receipts", load_receipts)
    builder.add_node( "parse_receipts", parse_receipts )
    builder.add_node("load_policy", load_policy)
    builder.add_node("synchronize_inputs", synchronize_inputs)
    builder.add_node("make_decision", make_decision)

    # Add edges to define the flow of the graph
    builder.add_edge(START, "load_claim")

    # Parallel execution of load_receipts and load_policy after load_claim
    builder.add_edge("load_claim", "load_receipts")
    builder.add_edge("load_claim", "load_policy")

    builder.add_edge( "load_receipts", "parse_receipts" )

    # Both branches converge here 
    builder.add_edge( "parse_receipts", "synchronize_inputs" ) 
    builder.add_edge( "load_policy", "synchronize_inputs" )
    builder.add_edge( "synchronize_inputs", "make_decision" )
    #builder.add_edge(["load_receipts", "load_policy"], "make_decision")

    builder.add_edge( "make_decision", END )

    claim_review_graph = builder.compile()
    return claim_review_graph

def build_graph1():
    graph = StateGraph()

    graph.add_state(
        state_name=START,
        state_type="start",
        next_states=["load_claim"]
    )

    graph.add_state(
        state_name="load_claim",
        state_type="function",
        function=load_claim,
        next_states=["load_receipts"]
    )

    graph.add_state(
        state_name="load_receipts",
        state_type="function",
        function=load_receipts,
        next_states=["load_policy"]
    )

    graph.add_state(
        state_name="load_policy",
        state_type="function",
        function=load_policy,
        next_states=["make_decision"]
    )

    graph.add_state(
        state_name="make_decision",
        state_type="function",
        function=make_decision,
        next_states=[END]
    )

    graph.add_state(
        state_name=END,
        state_type="end"
    )

    return graph