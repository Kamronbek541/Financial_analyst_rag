# RAG-Powered Financial Analyst 🤖📈

## Overview

This project is a sophisticated, end-to-end Question-Answering system designed for financial analysis. It allows users to ask questions in natural language about the latest annual 10-K filings of any publicly traded US company. The system fetches data directly from the SEC EDGAR database, processes it, and uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers.

This project demonstrates a professional, modular, and scalable approach to building modern LLM-powered applications, moving beyond simple notebooks to a fully functional AI system.

### Key Features

*   **Automated Data Ingestion**: Automatically downloads the latest 10-K filing for a given company ticker using the `sec-edgar-downloader`.
*   **Advanced Text Processing**: Utilizes `Unstructured` for robust parsing of complex HTML documents and `RecursiveCharacterTextSplitter` for intelligent text chunking.
*   **Vector-Based Retrieval**: Employs `Hugging Face Transformers` for local text embeddings and `ChromaDB` for efficient, persistent vector storage.
*   **State-of-the-Art RAG Pipeline**: Built with LangChain Expression Language (LCEL) for a declarative, powerful, and modular chain.
*   **High-Quality Generation**: Leverages OpenAI's `gpt-3.5-turbo` for generating coherent and contextually accurate answers.
*   **Interactive User Interface**: A user-friendly web interface built with `Streamlit` allows for seamless interaction.

## Project Architecture

The project follows a modular architecture to separate concerns, making it scalable and maintainable.

```
financial-analyst-rag/
│
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml            # Configuration for paths, models, etc.
├── .env                   # For API keys and secrets
│
├── app.py                 # Streamlit application entry point
│
└── src/                   # Core source code
    ├── data_ingestion/
    │   └── downloader.py    # Fetches SEC filings
    │
    ├── rag_pipeline/
    │   ├── document_processor.py # Loads and splits documents
    │   └── vector_store_manager.py # Manages ChromaDB and embeddings
    │
    └── core/
        └── chain.py         # The main RAG logic using LCEL
```

## How to Run

### 1. Prerequisites

*   Python 3.9+
*   An OpenAI API Key

### 2. Setup

Clone the repository:
```bash
git clone https://github.com/your-username/financial-analyst-rag.git
cd financial-analyst-rag
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY="sk-..."
```

Modify the `config.yaml` file if needed. The default `edgar_user_agent` should be updated with your information:
```yaml
edgar_user_agent: "Your Name YourEmail@example.com"
```

### 4. Launch the Application

Run the Streamlit app from the root directory:
```bash
streamlit run app.py
```
Open your browser and navigate to the provided local URL (usually `http://localhost:8501`).

## Future Improvements

*   **Evaluation Framework**: Integrate an evaluation framework like RAGAs to quantitatively measure the performance of the RAG pipeline (e.g., faithfulness, answer relevancy).
*   **Source Highlighting**: Enhance the UI to show the exact text chunks from the source document that were used to generate the answer.
*   **Support for Multiple Documents**: Extend the system to analyze multiple filings at once (e.g., last three 10-K reports) to answer questions about trends.
*   **Containerization**: Dockerize the application for easier deployment and scalability.
*   **Use Open Source LLMs**: Add support for using locally-hosted LLMs (e.g., via Ollama or vLLM) as an alternative to the OpenAI API.