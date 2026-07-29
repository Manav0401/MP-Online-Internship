import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


def load_documents(data_folder="data"):
    """
    Load all supported documents from the data folder.
    Supported formats:
        - PDF
        - TXT
        - DOCX
    """

    documents = []

    if not os.path.exists(data_folder):
        raise FileNotFoundError(
            f"Folder '{data_folder}' does not exist."
        )

    for file in os.listdir(data_folder):

        file_path = os.path.join(data_folder, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)

        else:
            continue

        documents.extend(loader.load())

    return documents