# AI Courtroom Assistant

A lightweight **LLM‑powered legal assistant** designed to help extract facts from legal case files, identify questions, generate answers using RAG (Retrieval-Augmented Generation), and assist during court proceedings.

This project processes **legal judgments or transcripts**, cleans them, generates structured Q/A pairs, retrieves relevant sections via embeddings, and produces context‑aware answers.

---

## 🧠 Key Features

* **Case File Parsing** – Extracts text, sections, facts, issues, and summaries.
* **Text Cleaning Pipeline** – Removes noise for better LLM reasoning.
* **Question Generator** – Produces questions from legal case facts.
* **RAG Engine** – Retrieves relevant case chunks for accurate answer generation.
* **LLM Response Generation** – Produces explanations, summaries, and Q/A.
* **Modular Architecture** – Each component is independently maintainable.

---

## 📁 Folder Structure

```
ai-courtroom-assistant/
│
├── main.py                  # Main entry point to run the assistant end-to-end
├── core.py                  # Core logic for orchestrating workflows
├── cleantxt.py              # Text cleaning and preprocessing utilities
├── parsecases.py            # Extracts structured info from raw case files
├── generate_case_qa.py      # Auto-generates questions from parsed case data
├── qna.py                   # Q/A generation helpers
├── rag.py                   # Embedding + vector search to retrieve context
├── llm_gen.py               # LLM wrapper for generating answers/summaries
│
├── data/                    # Store your uploaded case files or text
├── output/                  # Stores generated questions, answers, embeddings
└── requirements.txt         # Python dependencies
```

---

## 📌 What Each Script Does (Short + Clear)

### **1. main.py**

Runs the complete pipeline:

* Load case file → Clean → Parse → Generate Q/A → Retrieve context → Produce final outputs.

### **2. core.py**

Contains the **central functions** that tie all parts together.
Basically the "brain" that coordinates cleaning, parsing, RAG, and generation.

### **3. cleantxt.py**

Handles cleaning of raw legal text, such as:

* Removing headers/footers
* Removing extra spaces
* Correcting line breaks
* Normalizing unicode

### **4. parsecases.py**

Reads raw case files and extracts:

* Facts
* Issues
* Case background
* Important paragraphs

### **5. generate_case_qa.py**

Creates **relevant legal questions** from parsed case content using an LLM.

### **6. qna.py**

Generates **answers** for questions using context given by the RAG engine.

### **7. rag.py**

Handles the **Retrieval-Augmented Generation** workflow:

* Splits case text into chunks
* Generates embeddings
* Searches for relevant context

### **8. llm_gen.py**

Wrapper around **Ollama running DeepSeek** for generating for generating to produce:

* Answers
* Summaries
* Explanations

---

## ⚙️ Installation

```bash
git clone <your_repo_url>
cd ai-courtroom-assistant
pip install -r requirements.txt
```

You must also configure your **Ollama installation (no API key needed)**. Example:

```bash
export DEEPSEEK_API_KEY="your_key"
```

---

## ▶️ Usage

### **Make sure Ollama is running locally with the DeepSeek model, then run the full pipeline**

```bash
python main.py --file data/sample_case.txt
```

This will:

1. Clean the text
2. Parse the case
3. Generate questions
4. Retrieve context
5. Produce answers
6. Save all outputs inside **/output**

---

## 📦 Requirements

Install via `requirements.txt` (plus ensure **Ollama** is installed with `deepseek` model), which contains:

* transformers
* sentence-transformers
* Ollama (DeepSeek model) – primary LLM backend
* pandas
* numpy
* python-dotenv
* langchain 

---



