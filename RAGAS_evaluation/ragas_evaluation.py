"""
RAG Pipeline Evaluation using RAGAS with Groq (Open-Source LLM)

Metrics evaluated:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings


import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)



# Configuring Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key  
)


# Preparing Evaluation Dataset

data = {
    "question": [
        "What is a pandas Series and what kind of data can it hold?",
        "What is a pandas DataFrame and how is it structured?",
        "What is a NumPy ndarray and why is it used?",
        "How is a NumPy array different from a Python list?",
        "What are the main data structures provided by pandas?"
    ],

    "answer": [
        "A pandas Series is a one-dimensional labeled array capable of holding any data type such as integers, floats, strings, or Python objects.",

        "A pandas DataFrame is a two-dimensional labeled data structure with rows and columns. It is size-mutable and can contain heterogeneous data types, similar to a table.",

        "A NumPy ndarray is a homogeneous, multi-dimensional array of fixed-size items. It is used for efficient numerical computation and supports fast mathematical operations.",

        "Unlike Python lists, NumPy arrays store elements of the same data type and are optimized for numerical operations, making them faster and more memory-efficient.",

        "The primary data structures provided by pandas are Series and DataFrame. Series represents one-dimensional data, while DataFrame represents two-dimensional tabular data."
    ],

    "contexts": [
        [
            "Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.)."
        ],
        [
            "A DataFrame is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)."
        ],
        [
            "NumPy provides the ndarray, a homogeneous array object that contains elements of the same type.",
            "The ndarray is efficient for numerical operations."
        ],
        [
            "NumPy arrays are faster and more memory-efficient than Python lists.",
            "Python lists can hold elements of different data types, while NumPy arrays typically contain elements of a single type."
        ],
        [
            "pandas provides two primary data structures: Series and DataFrame.",
            "Series is one-dimensional, while DataFrame is two-dimensional."
        ]
    ],

    "ground_truth": [
        "A pandas Series is a one-dimensional labeled array that can hold various data types.",
        "A pandas DataFrame is a two-dimensional labeled table that can store heterogeneous data.",
        "A NumPy ndarray is a homogeneous array designed for fast numerical computation.",
        "NumPy arrays differ from Python lists by being homogeneous and optimized for numerical operations.",
        "pandas mainly provides Series and DataFrame as its core data structures."
    ]
}



dataset = Dataset.from_dict(data)


# RAGAS Evaluation

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=vector_embeddings
)




print("RAGAS Evaluation Results :-")
print(results)

df = results.to_pandas()

# Save results to CSV
df.to_csv("ragas_evaluation_results.csv")

