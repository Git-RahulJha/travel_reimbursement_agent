from agent_azure_openai_multi_agent import pipeline
from agent_azure_openai_multi_agent.agents import reciept_info_extractor_agent, test_llm
from agent_azure_openai_multi_agent.rag.rag_retrieval import parse_receipt, policy_rules
from agent_azure_openai_multi_agent.tools import get_raw_receipt
from agent_azure_openai_multi_agent.workflow import build_graph


def run_agent_pipeline():
    #pipeline.create_agent_pipeline()
    graph = build_graph()
    result = graph.invoke({"claim_id": "CLM-003"})
    decision = result["decision"]
    print(decision.model_dump_json(indent=2))

def test_run_agent():
    #reciept_agent = reciept_info_extractor_agent()
    result = reciept_info_extractor_agent().invoke({
        "messages": [
            {
                "role": "user",
                "content": "Retrieve and parse receipt R-2002."
            }
        ]
    })
    print(result)
    
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
    #test_llm()
    #test_run_agent()