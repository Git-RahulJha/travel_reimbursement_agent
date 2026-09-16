from pydantic import BaseModel


class InvestigationResult(BaseModel):
    reason: str
    findings: list[str]
    missing_information: list[str]
    recommended_action: str