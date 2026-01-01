## Backend

from fastapi import FastAPI 
import uvicorn

from pydantic import BaseModel
from typing import List, Optional
import joblib

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_groq import ChatGroq

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from Chat_History.utils import SQLiteChatMessageHistory



import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")





def create_vectorstore(urls: List[Optional[str]]) -> bool:
    """
    Given a list of URLs, process these URLs to create a vector store using HuggingFace-Embeddings and FAISS-DB.
    """
    try:
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators = [
                "\n### ",     # Markdown-style or section headers
                "\n## ",
                "\n# ",
                "\n\n",       # Paragraphs
                "\n",         # Newlines
            ],
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs)

        hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(
            documents = doc_splits,
            embedding = hf_embeddings
        )

        with open("VectorStoreDB/faiss_vectorstore.joblib", "wb") as index_file:
            joblib.dump(vector_store, index_file)

    except Exception as e:
        print(f"Error in creating vectorstore: {e}")
        return False
    
    return True
    




def create_rag_pipeline(get_session_history: callable):
    """
    Create a retrieval chain using the vector store.
    """
    if os.path.exists("VectorStoreDB/faiss_vectorstore.joblib"):
        with open("VectorStoreDB/faiss_vectorstore.joblib", "rb") as file:
            vector_store = joblib.load(file)

    retriever = vector_store.as_retriever()

    ## LLM-model
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)


    contextualize_q_prompt = """
            Given the chat history and the latest user question, reformulate the current question into a standalone version that can be understood without the previous context.

            Use the chat history only to clarify references (like “it”, “this function”, or “that library”).
            Do not answer the question.

            ---
            ### Chat History:
            {chat_history}
            ---
            ### Latest User Question:
            {input}
            ---

            ### Reformulated Standalone Question

    """

    contextualize_q_with_history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_with_history_prompt)

    qa_system_prompt = """
            You are an advanced AI documentation assistant designed to help users by answering queries using 
            the provided context from multiple documentation sources.

            Your primary goal is to deliver technically accurate, concise, and complete answers **strictly grounded in the retrieved context** and chat history.
            Do not hallucinate, assume, or fabricate information that is not supported by the given documentation.

            ---
            ### Context Information:
            {context}
            ---
            ### Chat History:
            {chat_history}
            ---
            ### User Query:
            {input}
            ---

            ### Response Guidelines:
            1. **Grounding:** Use only the information provided in the context to answer the question.  
            2. **Missing Info:** If the answer cannot be fully determined from the context, respond exactly with:  
            > "The provided documentation does not contain enough information to answer that precisely."
            3. **Clarity:** Write in a clear, developer-friendly tone. Avoid unnecessary repetition or overly generic statements.  
            4. **Structure:**  
                - Use Markdown formatting.  
                - Include **headings**, **bullet points**, and **code blocks** where appropriate.  
                - When multiple documents support the answer, synthesize them into a cohesive explanation.  
            5. **Source Attribution (if available):** At the end, just list URLs from which the information was derived.  
            6. **Consistency:** Maintain conversation context and continuity with previous answers (from chat history).  
            7. **Style:** Be factual, concise, and instructional — as if you were a senior developer or API mentor.

            ---

            ### Response Format:
            **Answer:**
            (Provide a clear explanation or step-by-step guide.)

            **Example (if applicable):**
            ```python
            # relevant code sample or command

    """

    qa_system_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )

    parser = StrOutputParser()

    qa_chain = qa_system_prompt_template | llm | parser

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_rag_chain






####### FastAPI app ########


app = FastAPI(
        title="LLM-Powered Multi-URL RAG Chatbot",
        version="1.0",
        description="A RAG chatbot using multiple URLs as context",
)


# basic endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot Backend is running successfully!"}



class URLsRequest(BaseModel):
    urls : List[Optional[str]]


# API Endpoint to process URLs and create vectorstore
@app.post("/process_urls")
async def process_urls(request: URLsRequest):
    """
    API endpoint to process a list of URLs and create a vectorstore.
    """
    urls = request.urls
    is_created = create_vectorstore(urls)
    if is_created:
        return {"status": True,
                "message": "Vectorstore created successfully from the provided URLs."}
    else:
        return {"status": False,
                "message": "Failed to create vectorstore. Please check the URLs and try again."}
    



class ChatResponseRequest(BaseModel):
    session_id : str
    user_query : str


# API Endpoint to get chat response   
@app.post("/chat_response")
async def chat_response(request: ChatResponseRequest):
    """
    API endpoint to get a chat response for a given user query and session ID.
    """

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLiteChatMessageHistory(session_id)
    
    # Extracting session_id and user_query from the JSON-request
    session_id = request.session_id
    user_query = request.user_query

    rag_chain = create_rag_pipeline(
        get_session_history=get_session_history
    )

    response = rag_chain.invoke(
        {"input": user_query},
        config = {
            "configurable": {"session_id": session_id}
        }
    )

    return {"answer": response["answer"]}








if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

    # To run this server, use the command:  "uvicorn serve:app --reload"  or  "python serve.py"


