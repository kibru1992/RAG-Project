# RAG Chatbot - Retrieval-Augmented Generation Application

A powerful Retrieval-Augmented Generation (RAG) chatbot that enables intelligent conversations using your own documents as knowledge base. Built with Python and powered by Groq's fast LLM inference.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)
![Groq](https://img.shields.io/badge/LLM-Groq_API-green)

## 🚀 Features

- **Document Intelligence**: Load and process your own documents (PDFs, text files, etc.)
- **Vector Search**: Semantic similarity search using ChromaDB
- **Fast Inference**: Leverages Groq's high-speed LLM processing
- **Context-Aware Responses**: Answers based on your document content
- **Multiple LLM Support**: Configurable for different language models
- **Docker Support**: Easy deployment with containerization

## 📁 Project Structure



## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/kibru1992/RAG-Project.git
   cd RAG-Project
2. Create virtual environment
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
3. Install dependencies
   pip install -r requirements.txt
4. Set up environment variables
   cp .env.example .env
   Edit .env and add your Groq API key
5. Add your documents
   Place your text files in the data/ directory
   Or use the existing sample documents

Usage

Running Locally
1. Start the chatbot:
   cd src
   python app.py
2. Interact with the bot:
   Enter your question: What is artificial intelligence?

Configuration

Environment Variables
Create a .env file with:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant  # Optional: specify model

License
This project is licensed under the MIT License - see the LICENSE file for details.
