# setup_core_db.py
import os
from rag import load_pdf_text, chunk_text, store_in_faiss
from main import BASE_DB_PATH

CORE_DB_PATH = os.path.join(BASE_DB_PATH, 'core_knowledge')
PDF_PATH = "data/core_knowledge_base/ipc_book.pdf"

if not os.path.exists(PDF_PATH):
    print(f" Error: The file {PDF_PATH} was not found.")
else:
    print("⏳ Starting to process the core legal database...")
    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text)
    store_in_faiss(chunks, CORE_DB_PATH)
    print(" Core legal database setup is complete.")