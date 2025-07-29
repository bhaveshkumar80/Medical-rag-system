# FastAPI-Based RAG System

This project implements a simple FastAPI server that wraps a Retrieval-Augmented Generation (RAG) pipeline using LangChain. The API accepts a natural language question, retrieves relevant documents, and generates a structured answer along with context and confidence score.

---

## How to Run the Server Locally

### 1. Clone the Repository or Download the Code

Make sure your project includes:

- `main.py` (FastAPI app)
- `retriever.py` (returns LangChain retriever)
- `utils.py` (contains `prompt_template`, `configure_llm`, `evaluate_chunks`)

### 2. Create & Activate a Virtual Environment (optional but recommended)

python -m venv rag_env
rag_env\Scripts\activate          # On Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Start the FastAPI Server 
uvicorn main:app --reload

### 5. Test the API
You can test the /qa POST endpoint directly in your browser:

http://127.0.0.1:8000/docs

Curl Request:
curl -X POST "http://127.0.0.1:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the important deadlines for Medicare enrollment?"}'






