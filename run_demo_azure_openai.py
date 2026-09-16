from agent_azure_openai.rag.rag_retrieval import parse_receipt, policy_rules
from agent_azure_openai.tools import get_raw_receipt
from agent_azure_openai.workflow import build_graph


def run_agent_pipeline():
    #pipeline.create_agent_pipeline()
    graph = build_graph()
    result = graph.invoke({"claim_id": "CLM-002"})
    decision = result["decision"]
    print(decision.model_dump_json(indent=2))


def test_run():
    policy = policy_rules(
            "What are the hotel reimbursement and approval rules?"
        )
    #print(type(policy))
    #print(isinstance(policy, PolicyRules))
    print(policy.model_dump_json(indent=2))
    
    
    raw_receipt = get_raw_receipt.invoke({ "receipt_id": "R-2002" }) 
    parsed_receipt = parse_receipt(raw_receipt) 
    print(parsed_receipt) 
    print(parsed_receipt.model_dump_json(indent=2))

if __name__ == "__main__":
    run_agent_pipeline()
    #test_run()