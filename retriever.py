import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

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

def content_aware_chunk(sentences):
    chunks = []
    current_chunk = []
    for sent in sentences:
        current_chunk.append(sent)
        if len(current_chunk) == 12:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-2:]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


async def split_into_chunks(file_path):

    pages = await data_loader(file_path)

    pages_list = []
    for pg in pages[1:]:
        pages_list.append(pg.page_content)
     
    pages = " ".join(pages_list)

    paragraphs = [p.strip() for p in pages.split('\n\n') if p.strip()]

    all_chunks = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        chunks = content_aware_chunk(sentences)
        all_chunks.extend(chunks)

    chunk_docs = []
    for chunk in all_chunks:
        document = Document(
        page_content=chunk,
        metadata={"source": "data/10050-medicare-and-you_0.pdf"}
    )
    chunk_docs.append(document)
    return chunk_docs

def get_retriever():
    chunks = asyncio.run(split_into_chunks(file_path))
    embedding_model = configure_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 4})
    return retriever




