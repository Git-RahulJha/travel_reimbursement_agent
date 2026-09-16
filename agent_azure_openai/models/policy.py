from typing import Optional
from pydantic import BaseModel, Field


class ExpenseRule(BaseModel):
    category: str
    eligible: bool
    max_amount: Optional[float] = None
    max_amount_unit: Optional[str] = None
    receipt_required: bool = False
    allowed_values: Optional[dict[str, list[str]]] = None
    explanation: Optional[str] = None

class ApprovalRule(BaseModel):
    authority: str
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None

class PolicyRules(BaseModel):
    expense_rules: list[ExpenseRule] = Field(default_factory=list)
    approval_rules: list[ApprovalRule] = Field(default_factory=list)