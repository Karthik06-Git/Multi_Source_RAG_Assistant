# LLM-Powered Multi-Source RAG Assistant

A Retrieval-Augmented Generation (RAG) based conversational assistant that enables developers and researchers to query information across multiple documentation websites, articles, and blogs from a single interface.

The system ingests user-provided URLs, builds a semantic vector index, and answers queries using large language models with conversational memory support.

---

## Project Overview

This project solves the problem of scattered technical documentation by providing a unified, conversational interface for querying multiple online sources. Instead of searching individual websites, users can ask questions in natural language and receive context-aware answers grounded in the indexed documents.

The system is designed with a clean separation between frontend, backend, and data persistence layers, making it scalable and production-ready.

---

## Key Features

- Multi-URL document ingestion (documentation sites, blogs, articles)
- Retrieval-Augmented Generation (RAG) using vector similarity search
- Conversational memory with session-based persistence
- Context-aware query reformulation using chat history
- FastAPI backend with clean REST-API endpoints
- Streamlit frontend for interactive chat experience
- SQLite-based session history storage
- FAISS vector store for efficient semantic search
- Dockerized and deployable on Hugging Face Spaces

---

## System Architecture

### Frontend
- Streamlit web interface
- Maintains UI-level chat history using session state
- Communicates with backend via REST APIs

### Backend
- FastAPI application
- Handles URL processing, vector retrieval, and LLM Response
- Implements history-aware RAG pipeline using LangChain

### Storage
- FAISS for vector embeddings (generated at runtime)
- SQLite for persistent chat session history

### LLM & Embeddings
- LLM (Llama-3.3-70b-versatile) via Groq API
- Hugging Face embeddings for semantic representation

---


## Evaluation & Performance

This RAG system was evaluated using the RAGAS (Retrieval-Augmented Generation Application System) framework to assess the quality of retrieval, answer relevance, and grounding of generated responses. The evaluation was conducted on curated queries derived from real-world technical documentations, including Pandas and NumPy sites.

### Evaluation Metrics
### Evaluation Metrics (RAGAS)

| Metric             | Score |
|--------------------|-------|
| Faithfulness       | 0.96  |
| Answer Relevancy   | 0.93  |
| Context Precision  | 1.00  |
| Context Recall     | 1.00  |


### Interpretation

- High faithfulness (0.96) indicates that the generated responses are strongly grounded in the retrieved documentation, with minimal hallucination.

- High answer relevancy (0.93) demonstrates accurate understanding of user queries.

- Perfect context precision and recall (1.00) confirm that the retriever consistently selects relevant and sufficient document chunks.

- These results validate the effectiveness of the history-aware retrieval strategy and the overall reliability of the RAG pipeline for documentation-based question answering.


---


## Tech Stack

- Python 3.10
- FastAPI
- Streamlit
- LangChain (modular architecture)
- FAISS
- SQLite
- Hugging Face Embeddings
- Groq API LLM
- Docker
- Hugging Face Spaces

---

## Project Structure

```
Multi_Source_RAG_Assistant/
│
├── client.py                 # Streamlit frontend
├── serve.py                  # FastAPI backend
├── Chat_History/
│   └── utils.py              # SQLite session history management
|   └── chat_memory.db
│
├── Research/
│   └── working.ipynb           # Notebook for applying RAG pipeline step-by-step
|   └── faiss_vectorstore.joblib
|
├── VectorStoreDB/
│   └── faiss_vectorstore.joblib  # For VectorStore index management  
│
├── requirements.txt
├── Dockerfile                 # a script of instructions for building a Docker image
└── README.md
```

---

## How It Works (System Flow)

1. User provides documentation URLs via the Streamlit interface.
2. FastAPI backend loads and parses content from the URLs.
3. Documents are split into chunks.
4. Embeddings are generated using a Hugging Face model.
5. Vectors are stored in a FAISS index.
6. User submits a query.
7. Chat history is retrieved from SQLite.
8. Query is reformulated using conversation history.
9. Relevant document chunks are retrieved.
10. LLM generates a grounded response.
11. Chat history is updated.
12. Answer is displayed in the UI.




---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Karthik06-Git/Multi_Source_RAG_Assistant.git
cd Multi_Source_RAG_Assistant
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY = <your_groq_api_key>
HF_TOKEN = <your_huggingface_token>
```

---


### 5. Run Backend

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000

[or] 

python serve.py
```

---

### 6. Run Frontend

```bash
streamlit run client.py
```

---

## Docker & Hugging Face Spaces

The project is fully Dockerized and compatible with Hugging Face Spaces.

Add required secrets (`HF_TOKEN`, `GROQ_API_KEY`) in Hugging Face Spaces and trigger a factory rebuild.

---

## API Endpoints

### Health Check
```
GET /
```

### Process URLs
```
POST /process_urls
```

### Chat Response
```
POST /chat_response
```

Request Body:
```json
{
  "session_id": "default_session",
  "user_query": "Your question here"
}
```

Response:
```json
{
  "answer": "Generated response"
}
```

---

## Author

Karthik Nayanala  

AI/ML | Generative AI | RAG Systems
