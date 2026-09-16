from typing import Any, Optional
from pydantic import BaseModel, Field

from datetime import date

class RawReceipt(BaseModel):
    receipt_id: str
    page_quality: float = Field(ge=0, le=1)
    ocr_text: str

class ParsedReceipt(BaseModel):
    receipt_id: str
    merchant: Optional[str] = None
    receipt_date: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None

    fare_class: Optional[str] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None

    room_type: Optional[str] = None
    nights: Optional[int] = None

    purpose: Optional[str] = None
    
class Receipt(BaseModel):
    receipt_id: str
    merchant: str
    receipt_date: date
    amount: float = Field(gt=0)
    currency: str
    category: str
    details: Optional[dict[str, Any]] = None