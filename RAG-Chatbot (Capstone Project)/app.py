from chatbot.rag import ask_question
from chatbot.memory import clear_memory

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    if question.lower() == "clear":
        clear_memory()
        print("Conversation memory cleared.")
        continue

    answer = ask_question(question)

    print("\nBot:", answer)