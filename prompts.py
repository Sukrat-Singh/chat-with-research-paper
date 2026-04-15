from langchain_core.prompts import ChatPromptTemplate

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise research assistant. "
            "Answer using only the provided context. "
            "If the answer is not present, say: 'Not found in the uploaded paper.' "
            "Keep the answer concise and mention page numbers when possible.",
        ),
        (
            "user",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)