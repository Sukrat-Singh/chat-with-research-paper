# Chat with Research Paper (RAG)

---

## Overview

A Retrieval-Augmented Generation (RAG) system that allows users to query research papers using semantic search + LLMs.

---

## Demo

![demo gif](/utils/gif1.gif)

---

## Features

- PDF upload
- Semantic search (ChromaDB)
- Context-aware answers (Groq Llama3)
- Source citation (page-level)
- Fast inference

---

## Architecture

`PDF → Chunking → Embeddings → Vector DB → Retrieval → LLM`

---

## Tech Stack

- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq (Llama3)
- Streamlit

---

## Setup

```bash
uv sync
uv run streamlit run app.py
```

---

## Folder Structure

```bash
chat-with-research-paper/
├── app.py # Streamlit UI (main entry point)
├── config.py # Configuration (paths, constants, env vars)
├── ingest.py # PDF loading, chunking, embedding, vector DB creation
├── rag.py # Retrieval + LLM response generation
├── prompts.py # Prompt templates for LLM
├── pyproject.toml # Project dependencies (uv)
├── uv.lock # Locked dependencies (reproducibility)
├── .env.example # Environment variables template
├── .gitignore # Git ignore rules
├── README.md # Project documentation
│
├── data/
│ ├── uploads/ # Uploaded PDFs (runtime)
│ └── chroma/ # Vector database storage
│
└── .venv/ # Virtual environment (not committed)
```

---

## Future Improvements

- Chat memory
- Multi-PDF support
- Hybrid search

---
