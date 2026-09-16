import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from agent_azure_openai.models.receipts import ParsedReceipt, RawReceipt
from agent_azure_openai.tools import get_raw_receipt
from agent_azure_openai.models.policy import PolicyRules
import agent_azure_openai.rag.rag_embedding as embed


llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)



def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def get_policy_rules_old(user_query):
    structured_llm = llm.with_structured_output(PolicyRules)
    vector_store = embed.load_embeddings('faiss_index')
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5
        }
    )

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a policy extraction assistant.

        Extract policy rules only from the supplied policy context.

        Rules:
        - Do not invent or assume values.
        - If a rule is not present in the context, do not create it.
        - Preserve monetary limits and their units exactly.
        - Extract eligibility, receipt requirements, allowed values,
        and approval authority rules where explicitly stated.
        - Return only information supported by the supplied context.
        """
            ),
            (
                "human",
                """
        Policy context:

        {context}

        Extract the applicable policy rules.
        """
            )
        ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | structured_llm
    )

    response = rag_chain.invoke(
        user_query
    )
    print(response)


def policy_rules(query: str) -> PolicyRules:
    structured_llm = llm.with_structured_output(PolicyRules)
    vector_store = embed.load_embeddings('faiss_index')
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5
        }
    )

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a policy extraction assistant.

        Extract policy rules only from the supplied policy context.

        Rules:

        1. Do not invent or assume values.
        2. Extract only rules explicitly supported by the context.
        3. Preserve monetary limits and their units.
        4. Extract actual expense categories such as Hotel, Meals,
        Air Travel, Taxi, etc.
        5. Do not use policy clause IDs as expense categories.
        6. Extract receipt requirements.
        7. Extract allowed values such as room type or fare class.
        8. Extract explicit exclusions.
        9. Extract important conditions that affect reimbursement.
        10. Extract approval authority and amount ranges.
        11. Do not create executable code or arbitrary rule expressions.
        12. If information is not present, leave the corresponding
            field empty or null.

        Return the result using the supplied PolicyRules schema.
        """
            ),
            (
                "human",
                """
        Policy context:

        {context}

        Extract the applicable policy rules.
        """
        )
    ])

    documents = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    chain = prompt | structured_llm

    return chain.invoke({
        "context": context
    })

# this is not RAG, just a simple chain to parse the receipt text into structured data
def parse_receipt(receipt: RawReceipt) -> ParsedReceipt:
    structured_llm = llm.with_structured_output(ParsedReceipt)
    receipt_parser_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a receipt extraction assistant.

            Extract structured receipt information only from the supplied OCR text.

            Rules:
            - Extract only information explicitly present in the OCR text.
            - Do not invent or assume missing values.
            - Do not calculate or derive values.
            - Preserve dates, amounts and locations as they appear in the receipt.
            - Identify the expense category only when supported by the OCR text.
            - Extract fare class for flights when explicitly present.
            - Extract room type and number of nights for hotels when explicitly present.
            - Extract purpose only when explicitly present.
            - If a field is not available, return null.
            """
        ),
        (
            "human",
            """
            Receipt ID: {receipt_id}
            OCR text: {ocr_text}
            
            Extract the receipt information.
            """
        )
    ])

    receipt_parser_chain = receipt_parser_prompt | structured_llm

    parsed = receipt_parser_chain.invoke({
        "receipt_id": receipt.receipt_id,
        "ocr_text": receipt.ocr_text
    })

    # receipt_id comes from our trusted source, not OCR.
    return parsed.model_copy(
        update={"receipt_id": receipt.receipt_id}
    )

    


#get_policy_rules('What are the hotel reimbursement and approval rules?')