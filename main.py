from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from retriever import get_retriever
from utils import prompt_template, configure_llm, evaluate_chunks
import json

# FastAPI app
app = FastAPI()

# Request schema
class QuestionRequest(BaseModel):
    question: str

# Response schema
class AnswerResponse(BaseModel):
    answer: str
    source: str
    confidence_score: float

# LangChain response structure model
class ResponseStructure(BaseModel):
    answer: str

@app.post("/qa", response_model=AnswerResponse)
def get_answer(request: QuestionRequest):
    try:
        # Setup components
        retriever = get_retriever()
        prompt = prompt_template()
        llm = configure_llm()
        parser = JsonOutputParser()

        # Retrieve context
        doc = retriever.invoke(request.question)
        context_text = "\n".join([d.page_content for d in doc]) if isinstance(doc, list) else str(doc)

        # Evaluate confidence
        confidence_score = evaluate_chunks(context_text, request.question)

        # QA chain
        parallel_chain = RunnableParallel({
            'context': retriever,
            'question': RunnablePassthrough()
        })
        qa_chain = parallel_chain | prompt | llm | parser
        result = qa_chain.invoke(request.question)

        print("result: ", result)

        # result = json.loads(result)

        return AnswerResponse(
            answer=result,
            source=context_text,
            confidence_score=confidence_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))