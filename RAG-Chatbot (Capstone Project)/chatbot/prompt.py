from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    template = """
You are an AI assistant.

Use the conversation history and the provided context to answer.

If the answer is not found in the context, reply:

"I couldn't find that information in the provided documents."

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

    return ChatPromptTemplate.from_template(template)