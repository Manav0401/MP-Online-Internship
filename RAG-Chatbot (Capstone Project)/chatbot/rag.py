from chatbot.retriever import get_retriever
from chatbot.llm import get_llm
from chatbot.prompt import get_prompt
from chatbot.memory import (
    add_user_message,
    add_ai_message,
    get_chat_history,
)

retriever = get_retriever()
llm = get_llm()
prompt = get_prompt()


def ask_question(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    history = "\n".join(
        f"{msg.type}: {msg.content}"
        for msg in get_chat_history()
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )

    add_user_message(question)
    add_ai_message(response.content)

    return response.content