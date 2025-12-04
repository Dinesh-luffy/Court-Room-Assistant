# process_json.py
import os
import json
from rag import load_json_qa, chunk_text, store_in_faiss
from main import BASE_DB_PATH

CORE_DB_PATH = os.path.join(BASE_DB_PATH, 'core_knowledge')
JSON_FILE_PATH = "data/core_knowledge_base/constitution_qa.json"

def main():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"⚠️ Error: The file {JSON_FILE_PATH} was not found.")
        return

    print(f"⏳ Processing JSON file: {JSON_FILE_PATH}")
    
    # Load and process the JSON data
    qa_texts = load_json_qa(JSON_FILE_PATH)
    all_text = "\n".join(qa_texts)
    chunks = chunk_text(all_text)
    
    # Store the chunks in the core database
    store_in_faiss(chunks, CORE_DB_PATH)
    
    print(f"✅ JSON data from {JSON_FILE_PATH} has been added to the core legal database.")

if __name__ == "__main__":
    main()