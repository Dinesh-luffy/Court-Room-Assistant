# llm_gen.py
import ollama
from rag import retrieve_context  # make sure your rag.py has this

client = ollama.Client(host="http://127.0.0.1:11434")

# Path to your core legal knowledge FAISS database
CORE_DB_PATH = "data/vector_store/core_knowledge"

def generate_answer(query, context=""):
    """
    Generate a legal response using DeepSeek with retrieved context or general knowledge.
    Works for:
    - Case-specific queries (context from FAISS)
    - General legal queries (with core knowledge)
    - Opponent argument analysis (special handling)
    """

    # --- Mode 1: Opponent's argument ---
    if context and "The opponent argued:" in query:
        prompt = f"""
You are a top-tier legal strategist. Your task is to analyze the opponent's argument and suggest a counter-strategy.

**Step 1 – Analyze Opponent's Strategy**  
Explain their core legal theory or line of reasoning.

**Step 2 – Propose Counter-Points**  
Using the provided legal context, propose specific legal arguments, relevant citations, or pointed questions to undermine their strategy.

Opponent's statement: {query.replace("The opponent argued:", "").strip()}

Relevant legal context:
{context}

Provide your response clearly separated into:
- Analysis of Opponent's Strategy
- Counter-Strategic Points
"""

    # --- Mode 2: Case-specific Q&A (with case FAISS context) ---
    elif context:
        prompt = f"""
You are a helpful legal research assistant. Answer the lawyer's question **only using the provided legal context**.  
If the answer is not in the context, state clearly: "The provided case documents do not contain relevant information."

Relevant legal context:
{context}

Lawyer's question: "{query}"

Answer concisely and professionally, citing sections or precedents from the context where possible.
"""

    # --- Mode 3: General legal Q&A (Option 5) ---
    else:
        # Try to retrieve context from core knowledge first
        core_context = ""
        try:
            core_context = retrieve_context(query, CORE_DB_PATH, top_k=5)
        except Exception as e:
            print("⚠️ Could not retrieve from core knowledge FAISS:", e)
            core_context = ""

        if core_context:
            prompt = f"""
You are a knowledgeable assistant for the Indian legal system. Answer the lawyer's question using the provided core legal knowledge.  
If the answer is not present in the context, state clearly: "The core knowledge does not contain relevant information."

Core legal knowledge:
{core_context}

Lawyer's general legal question: "{query}"

Answer concisely with proper legal context, sections, or process explanation.
"""
        else:
            # fallback to general model knowledge
            prompt = f"""
You are a knowledgeable assistant for the Indian legal system.  
Provide a clear and professional answer to the lawyer's question based on your general knowledge of Indian law, statutes, and procedure.  
If the question refers to a section number, explain its meaning in law.  

Lawyer's general legal question: "{query}"

Answer concisely with proper legal context, sections, or process explanation.
"""

    try:
        response = client.generate(model="deepseek-r1:7b", prompt=prompt)
        # Ollama returns dict with 'response'
        answer_text = response.get('response', '')
        # Clean any leftover internal markers
        answer_text = (
            answer_text.replace("<think>", "")
                       .replace("</think>", "")
                       .strip()
        )
        return answer_text or "⚠️ No response generated."
    except Exception as e:
        print("⚠️ Error generating response:", e)
        return "⚠️ No response from LLM."
