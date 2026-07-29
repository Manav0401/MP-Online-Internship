from chatbot.loader import load_documents
from chatbot.splitter import split_documents
from chatbot.embeddings import get_embedding_model
from chatbot.vector_db import create_vector_store, save_vector_store


def main():
    print("=" * 60)
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("\nSplitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("\nLoading embedding model...")
    embedding_model = get_embedding_model()

    print("\nCreating FAISS vector database...")
    vector_store = create_vector_store(chunks, embedding_model)

    print("\nSaving vector database...")
    save_vector_store(vector_store)

    print("\n✅ Vector database saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()