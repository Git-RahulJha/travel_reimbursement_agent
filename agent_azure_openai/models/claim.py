from pydantic import BaseModel, Field
from typing import Optional


class ClaimLine(BaseModel):
    line_id: str
    category: str
    description: str
    amount: float = Field(gt=0)
    date_from: str
    date_to: str
    sectors: Optional[int] = None
    nights: Optional[int] = None
    travel_days: Optional[int] = None
    receipt_id: Optional[str] = None
    fare_class: Optional[str] = None
    room_type: Optional[str] = None
    purpose: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    employee_id: str
    employee_name: str
    trip_id: str
    trip_from: str
    trip_to: str
    submitted_on: str
    lines: list[ClaimLine]