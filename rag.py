# rag.py
import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load the local, fine-tuned model correctly
embedding_model = HuggingFaceEmbeddings(model_name="finetuned_legal_bert")

def load_pdf_text(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def load_json_qa(json_path: str) -> list[str]:
    """Extracts text from a JSON file with Q&A pairs."""
    qa_texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        qa_text = f"Question: {entry.get('question', '')}\nAnswer: {entry.get('answer', '')}"
        qa_texts.append(qa_text)
    return qa_texts

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits a long string of text into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def store_in_faiss(chunks: list[str], db_path: str):
    """Stores text chunks in a specific FAISS vector database."""
    if os.path.exists(os.path.join(db_path, "index.faiss")):
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        db.add_texts(chunks)
    else:
        db = FAISS.from_texts(chunks, embedding_model)
    db.save_local(db_path)
    print(f"Stored {len(chunks)} chunks in FAISS DB at {db_path}.")

def retrieve_context(query: str, db_path: str, top_k: int = 3) -> str:
    """Searches a specific FAISS database and returns the most relevant text chunks."""
    if not os.path.exists(os.path.join(db_path, "index.faiss")):
        print(f"⚠️ No FAISS index found at {db_path}. Please upload PDFs first!")
        return ""
    
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=top_k)
    context = "\n\n".join([res.page_content for res in results])
    return context