import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def embed_documents():
    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    return embeddings

def save_embeddings(chunks, index):
    embeddings=embed_documents()    

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local("faiss_index")

def load_embeddings(index):
    embeddings=embed_documents()    

    vector_store = FAISS.load_local(
        index,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Number of vectors:", vector_store.index.ntotal)

    return vector_store