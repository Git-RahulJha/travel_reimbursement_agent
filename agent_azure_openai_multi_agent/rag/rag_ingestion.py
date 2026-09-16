from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
#from .rag_embedding import embed_documents, save_embeddings
import rag_embedding as embed

# load pdf document
def load_document() -> List[Document]:
    print('Document loading...')
    loader = PyPDFLoader("policies/TRV-POL_v1.0.pdf")
    documents = loader.load()
    print(f"Documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    print('Chunking started...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")

    print('Chunks created.')
    return chunks

def save_embeddings(chunks, index):
    print('Embeddings saving started into vector store...')
    embed.save_embeddings(chunks=chunks, index=index)
    print('Embeddings saved into vector store...')

def build_ingestion():
    print('Build RAG pipeline: Ingestion started...')
    documents = load_document()
    chunks = chunk_documents(documents)
    save_embeddings(chunks, 'faiss_index')
    print('Build RAG pipeline: Ingestion completed...')

if __name__ == "__main__":
    build_ingestion()