from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector database from document chunks.
    """

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


def save_vector_store(vector_store, path="vector_store"):
    """
    Save the FAISS index locally.
    """

    vector_store.save_local(path)


def load_vector_store(embedding_model, path="vector_store"):
    """
    Load the FAISS index from disk.
    """

    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store