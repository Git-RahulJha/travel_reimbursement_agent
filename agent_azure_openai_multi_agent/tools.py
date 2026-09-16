import json

from langchain.tools import tool
import os
from rich import print
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from .data_loader import ClaimRepository
from agent_azure_openai_multi_agent.models.claim import Claim
from agent_azure_openai_multi_agent.models.receipts import RawReceipt, ParsedReceipt

load_dotenv()

llm = ChatOpenAI(model='gpt-5.1', 
                 api_key=os.getenv('OPENAI_API_KEY'), 
                 temperature=0, 
                 base_url=os.getenv('AZURE_OPENAI_ENDPOINT'))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
repo = ClaimRepository()

@tool
def get_claim(claim_id: str) -> Claim:
        """Search the claim by claim_id"""
        return repo.claim(claim_id=claim_id)

@tool
def get_receipt(receipt_id: str) -> Claim:
        """Search the receipt by receipt_id"""
        return repo.receipt(receipt_id=receipt_id)

@tool
def get_raw_receipt(receipt_id: str) -> Claim:
        """Search the raw receipt by receipt_id"""
        return repo.raw_receipt(receipt_id=receipt_id)


def test_call():
        claim = get_claim.invoke({
        "claim_id": "CLM-002"
        })


        print("Claim ID:", claim.claim_id)
        print("Employee:", claim.employee_name)

        print("\nClaim Lines:")

        for line in claim.lines:

                print(
                        line.line_id,
                        line.category,
                        line.amount
                )

        receipt = get_receipt.invoke({
        "receipt_id": "R-2002"
        })

        print(receipt)

#test_call()