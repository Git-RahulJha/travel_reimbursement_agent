from agent_azure_openai_multi_agent.rag.rag_retrieval import policy_rules

from agent_azure_openai_multi_agent.models.policy import PolicyRules, ExpenseRule

def find_expense_rule(policy: PolicyRules, category: str) -> ExpenseRule | None:
    category = category.lower()

    for rule in policy.expense_rules:
        if rule.category.lower() == category:
            return rule

    return None

def get_policy_rules():
    policy = policy_rules(
                "What are the hotel reimbursement and approval rules?"
            )
    return policy