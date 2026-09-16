from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from langchain.agents import create_agent

from agent_azure_openai_multi_agent.models.receipts import ParsedReceipt
from agent_azure_openai_multi_agent.models.investigation import InvestigationResult
from agent_azure_openai_multi_agent.tools import get_raw_receipt

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)

def test_llm():
    response = llm.invoke("Hello, are you working?")
    print(response.content)

def reciept_info_extractor_agent():
    print('hello')
    receipt_agent = create_agent(model=llm, tools=[get_raw_receipt],
    system_prompt="""
        You are a receipt review agent.

        Your responsibility is to retrieve and extract information from travel
        expense receipts.

        Rules:
        - Use the get_raw_receipt tool to retrieve the receipt.
        - Extract information only from the OCR text returned by the tool.
        - Do not invent missing information.
        - Do not calculate or derive values.
        - If a field is unavailable, return null.
        - Return the final result as a ParsedReceipt.
        """,
        response_format=ParsedReceipt
        )
    return receipt_agent

def investigation_agent():

    return create_agent(
        model=llm,
        tools=[],
        system_prompt="""
        You are a travel claim investigation agent.

        Your responsibility is to investigate claims that require
        manual review.

        Review the claim decision and identify:
        - why manual review was triggered
        - what information is missing or ambiguous
        - what evidence is available
        - what Finance should verify

        Do not change approved or rejected amounts.
        Do not override the deterministic decision engine.
        Do not invent facts.

        Return the investigation findings as an InvestigationResult.
        """,
        response_format=InvestigationResult
    )

#reciept_info_extractor_agent()

def policy_rule_extractor_agent():
    print('hello')
    #call tool openai_web_search('what is llm')