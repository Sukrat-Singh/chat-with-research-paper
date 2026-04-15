import os
from pathlib import Path
from ingest import build_vectorstore, load_vectorstore, get_index_dir
from rag import answer_question
from config import GROQ_API_KEY
import sys

def test():
    pdf_path = Path("dummy.pdf")
    if not pdf_path.exists():
        print("dummy.pdf not found.")
        sys.exit(1)
        
    print("Testing build_vectorstore...")
    try:
        vectorstore = build_vectorstore(pdf_path)
        print("Vectorstore built successfully.")
    except Exception as e:
        print(f"Failed to build vectorstore: {e}")
        sys.exit(1)

    print("Testing answer_question...")
    try:
        # Check if GROQ_API_KEY is available during test execution to prevent 401 error.
        if not GROQ_API_KEY:
            print("Skipping answer_question test, GROQ_API_KEY not found.")
            return

        answer, docs = answer_question(vectorstore, "What accuracy did the authors achieve?")
        print(f"Answer: {answer}")
        print(f"Docs retrieved: {len(docs)}")
    except Exception as e:
        print(f"Failed to answer question: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test()
