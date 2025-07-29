from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA, LLMChain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from retriever import get_retriever



def configure_llm():
    llm = OllamaLLM(model="llama3.2")
    return llm

def prompt_template():
    prompt = PromptTemplate(
        template="""You are a helpful assistant for a medical domain. You need to answer medical related queries from the given context.

        Instructions:
        1. If the context is insufficient, just say "Context is insufficiant for answer".
        2. Your focus is to provide clear, concise and correct answer.
        3. Strictly follow the context for user queries.
        4. Answer should strictly follows the json format.

        Output Schema:
        "answer": <response>

        Context:
        {context}

        Question:
        {question} 
    """,
    input_variables = ['context', 'question']
    )
    return prompt

def cosine_similarity_score(result, question):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = model.encode([result])[0]
    emb2 = model.encode([question])[0]
    return cosine_similarity([emb1], [emb2])[0][0]

def evaluate_chunks(retrieved_context, question):
    score = cosine_similarity_score(retrieved_context, question)
    return score


    
    





