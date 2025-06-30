import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

file_path = "data/10050-medicare-and-you_0.pdf"

async def data_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    return pages

def configure_embedding_model():
    embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embedding_model

async def split_into_chunks(file_path, chunk_size=1000, chunk_overlap=200):
    # Step 1: Load pages
    pages = await data_loader(file_path)

    # Step 2: Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Step 3: Split the documents
    chunks = text_splitter.split_documents(pages)

    # Step 4: Generate the embeddings
    embedding_model = configure_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    return vectorstore


if __name__ == "__main__":
    vs = asyncio.run(split_into_chunks(file_path))
    print(vs)



