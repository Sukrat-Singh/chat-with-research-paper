import streamlit as st
from pathlib import Path
from ingest import save_uploaded_pdf, build_vectorstore, load_vectorstore, get_index_dir
from rag import answer_question
from config import GROQ_API_KEY
import os

st.set_page_config(page_title="Chat with Research Paper", page_icon="📄", layout="centered")

st.title("📄 Chat with Research Paper")
st.markdown("Upload a PDF and ask questions. The app will extract context using Hugging Face embeddings & ChromaDB, and generate answers using a Groq LLM.")

# Add custom CSS to make it look a bit more premium
st.markdown("""
<style>
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    .stButton button {
        border-radius: 8px;
        background-color: #2e66ff;
        color: white;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1b4bcf;
    }
</style>
""", unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please set it in your .env file or environment variables.")
    st.stop()

uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type=["pdf"])

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        file_path = save_uploaded_pdf(uploaded_file)
        index_dir = get_index_dir(file_path)
        
        with st.spinner("Processing PDF and indexing content... This may take a moment."):
            if index_dir.exists():
                st.session_state.vectorstore = load_vectorstore(file_path)
            else:
                st.session_state.vectorstore = build_vectorstore(file_path)
        
        st.session_state.current_file = uploaded_file.name
        st.success(f"Successfully processed: **{uploaded_file.name}**")

if st.session_state.vectorstore is not None:
    st.divider()
    question = st.chat_input("Ask a question about the paper...")
    
    if question:
        st.chat_message("user").write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing research paper..."):
                try:
                    answer, docs = answer_question(st.session_state.vectorstore, question)
                    st.write(answer)
                    
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(docs):
                            page = doc.metadata.get('page', '?')
                            st.markdown(f"**Page {page + 1 if isinstance(page, int) else page}**")
                            st.write(doc.page_content)
                            st.divider()
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")