# 🤖 RAG Chatbot using LangChain, FAISS & Groq

A modular **Retrieval-Augmented Generation (RAG)** chatbot built with **Python**, **LangChain**, **FAISS**, **Sentence Transformers**, and the **Groq API**. The chatbot retrieves relevant information from a custom knowledge base before generating accurate, context-aware responses using a Large Language Model (LLM).

---

## 📌 Features

- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using FAISS Vector Database
- 🧠 SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
- 💬 Groq LLM integration for natural language responses
- 📝 Supports multiple text documents as a knowledge base
- 🧩 Modular project structure for easy maintenance
- 🗂️ Conversation memory support
- ⚡ Fast document retrieval using vector search
- 🔒 Environment variable support for API keys

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| LangChain | RAG Framework |
| FAISS | Vector Database |
| Sentence Transformers | Text Embeddings |
| Groq API | Large Language Model |
| HuggingFace | Embedding Model |
| dotenv | Environment Variables |

---

## 📂 Project Structure

```
RAG-Chatbot/
│
├── app.py
├── build_index.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── chatbot/
│   ├── embeddings.py
│   ├── llm.py
│   ├── loader.py
│   ├── memory.py
│   ├── prompt.py
│   ├── rag.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vector_db.py
│
├── data/
│   ├── *.txt
│   └── ...
│
├── vector_store/
│
└── .env
```

---

## ⚙️ How It Works

```
                User Question
                      │
                      ▼
          Generate Query Embedding
                      │
                      ▼
          Search FAISS Vector Store
                      │
                      ▼
       Retrieve Relevant Document Chunks
                      │
                      ▼
    Construct Prompt with Retrieved Context
                      │
                      ▼
               Groq Large Language Model
                      │
                      ▼
                Context-Aware Response
```

The chatbot first searches the vector database to retrieve the most relevant information from the knowledge base. The retrieved context is then provided to the Groq LLM, which generates an accurate and natural language response.

---

## 📖 Knowledge Base

The chatbot retrieves information from documents stored inside the **data/** folder.

Current knowledge base includes topics such as:

- Python Programming
- Programming Concepts
- Computer Networks
- Operating Systems
- Databases
- Cyber Security
- Cloud Computing
- Mathematics
- Physics
- Chemistry
- Biology
- Astronomy
- Geography
- World History
- Indian History
- Indian Constitution
- Economics
- Finance
- Sports
- Health
- World Capitals
- Countries
- Literature
- Current Affairs
- Inventions
- Famous People
- University Information

New documents can be added by simply placing them inside the **data/** folder and rebuilding the vector database.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Manav0401/RAG-Chatbot.git
cd RAG-Chatbot
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 🏗️ Build the Vector Database

After adding or updating documents inside the **data/** folder, generate the FAISS index using:

```bash
python build_index.py
```

This creates the vector database inside:

```
vector_store/
```

---

## ▶️ Run the Chatbot

```bash
python app.py
```

Example

```
You: What is Python?

Bot:
Python is a high-level, interpreted programming language known for its readability and extensive standard library.
```

---

## 🧠 RAG Pipeline

```
Documents
     │
     ▼
Load Documents
     │
     ▼
Text Splitting
     │
     ▼
Generate Embeddings
     │
     ▼
Store in FAISS
     │
     ▼
User Question
     │
     ▼
Query Embedding
     │
     ▼
Similarity Search
     │
     ▼
Relevant Chunks
     │
     ▼
Groq LLM
     │
     ▼
Final Response
```

---

## 📌 Advantages

- Accurate responses using retrieved context
- Reduces hallucinations
- Easily extensible knowledge base
- Fast semantic search
- Modular architecture
- Easy to maintain and expand

---

## 📦 Requirements

- Python 3.10+
- LangChain
- FAISS
- Sentence Transformers
- Groq API
- dotenv

Install using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- Streamlit Web Interface
- PDF and DOCX document support
- Source citations
- Confidence score for retrieved results
- Hybrid Retrieval (BM25 + FAISS)
- Groq fallback for general knowledge when no relevant document is found
- Multi-file ingestion
- Chat history persistence

---

## 👨‍💻 Author

**Manav M George**

Integrated M.Tech in Artificial Intelligence  
VIT Bhopal University

GitHub: https://github.com/Manav0401

---
