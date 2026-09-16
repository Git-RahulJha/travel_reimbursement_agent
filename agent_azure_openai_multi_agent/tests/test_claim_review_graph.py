from ..workflow import build_graph as claim_review_graph

result = claim_review_graph.invoke({
    "claim_id": "CLM-002"
})


decision = result["decision"]


print("\n========== FINAL DECISION ==========")

print(
    decision.model_dump_json(
        indent=2
    )
)