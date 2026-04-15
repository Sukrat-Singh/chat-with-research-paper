from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K
from prompts import QA_PROMPT

def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", None)
        page_label = f"Page {page + 1}" if page is not None else "Page ?"
        formatted.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
    )

def answer_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    context = format_docs(docs)
    chain = QA_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, docs