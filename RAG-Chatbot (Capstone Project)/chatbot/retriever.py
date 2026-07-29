from chatbot.embeddings import get_embedding_model
from chatbot.vector_db import load_vector_store


def get_retriever():
    """
    Load the saved FAISS vector store and create a retriever.
    """

    embedding_model = get_embedding_model()

    vector_store = load_vector_store(embedding_model)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    return retriever