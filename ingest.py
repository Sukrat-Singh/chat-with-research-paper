import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    UPLOAD_DIR,
)

def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Save Streamlit UploadedFile to disk and return the saved path.
    """
    ensure_dirs()
    file_path = UPLOAD_DIR / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def file_hash(file_path: Path) -> str:
    """
    Create a stable hash for the PDF so each document gets its own index.
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()

def get_index_dir(pdf_path: Path) -> Path:
    return CHROMA_DIR / file_hash(pdf_path)

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vectorstore(pdf_path: Path) -> Chroma:
    """
    Load PDF -> split into chunks -> embed -> store in Chroma.
    """
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    index_dir = get_index_dir(pdf_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(index_dir),
    )
    return vectorstore

def load_vectorstore(pdf_path: Path) -> Chroma:
    """
    Load an existing Chroma index for the PDF.
    """
    embeddings = get_embeddings()
    index_dir = get_index_dir(pdf_path)
    return Chroma(
        persist_directory=str(index_dir),
        embedding_function=embeddings,
    )