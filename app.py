import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import streamlit as st
import yaml
from dotenv import load_dotenv
import os

from src.data_ingestion.downloader import SECDownloader
from src.rag_pipeline.document_processor import DocumentProcessor
from src.rag_pipeline.vector_store_manager import VectorStoreManager
from src.core.chain import QASystem

st.set_page_config(
    page_title="RAG Financial Analyst",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_configuration():
    """Loads configuration from YAML and .env files."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    load_dotenv()
    return config

def process_and_display_answer(prompt):
    """Adds user prompt to history and gets/displays model's answer."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing document..."):
        response = st.session_state.qa_system.ask_question(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.title("🤖 RAG-Powered Financial Analyst")
    st.markdown("Ask questions about the latest 10-K filings of any public US company.")

    config = load_configuration()

    with st.sidebar:
        st.header("Setup")
        
        if 'ticker_input' not in st.session_state:
            st.session_state.ticker_input = "AAPL"

        st.markdown("#### Pick a popular ticker:")
        
        suggested_tickers = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMZN","Custom"]
        
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)
        
        all_cols = row1_cols + row2_cols

        for i, ticker_suggestion in enumerate(suggested_tickers):
            if all_cols[i].button(ticker_suggestion, key=f"ticker_btn_{i}", use_container_width=True):
                st.session_state.ticker_input = ticker_suggestion
        
        st.markdown("#### Or enter your own:")
        ticker = st.text_input(
            "Enter a company ticker:", 
            key="ticker_input",
            label_visibility="collapsed"
        ).upper()
        
        if st.button("Load Filing", key="load_filing", use_container_width=True, type="primary"):
            if not ticker:
                st.error("Please enter a ticker.")
            else:
                if 'ticker' not in st.session_state or st.session_state.ticker != ticker:
                    st.session_state.messages = []
                    st.session_state.qa_system = None
                    st.session_state.ticker = ticker
                
                with st.status(f"Processing 10-K for {ticker}...", expanded=True) as status:
                    try:
                        status.update(label="Step 1/4: Downloading filing...")
                        downloader = SECDownloader(config['data_path'], config['edgar_user_agent'])
                        filing_path = downloader.download_latest_10k(ticker)
                        
                        if filing_path is None:
                            st.error(f"Could not retrieve the 10-K filing for {ticker}.")
                            status.update(label="Download failed.", state="error")
                            return

                        status.update(label="Step 2/4: Cleaning and splitting...")
                        doc_processor = DocumentProcessor(config['chunk_size'], config['chunk_overlap'])
                        documents = doc_processor.load_and_split_document(filing_path)
                        
                        if not documents:
                            st.error("Failed to process the document.")
                            status.update(label="Processing failed.", state="error")
                            return
                        
                        status.update(label="Step 3/4: Creating knowledge base...")
                        vector_store_path = os.path.join(config['vector_store_path'], ticker)
                        vector_store_manager = VectorStoreManager(
                            vector_store_path, 
                            config['embedding_model_name']
                        )
                        db = vector_store_manager.create_or_get_vector_store(documents, force_recreate=True)
                        
                        status.update(label="Step 4/4: Initializing QA System...")
                        st.session_state.qa_system = QASystem(db.as_retriever(), config['llm_model_name'])
                        
                        status.update(label=f"Knowledge base for {ticker} is ready!", state="complete")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        status.update(label="An error occurred during processing.", state="error")

    if 'qa_system' in st.session_state and st.session_state.qa_system:
        st.header(f"Ask Questions about {st.session_state.ticker}'s 10-K Filing")

        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        st.markdown("---")
        st.subheader("Try these examples:")
        
        suggested_questions = [
            "What are the main business segments?",
            "Summarize the primary business risks.",
            "What was the total revenue last year?",
            "Who are the main competitors?"
        ]

        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    process_and_display_answer(question)

        st.markdown("---")

        if prompt := st.chat_input("What is your question?"):
            process_and_display_answer(prompt)
    else:
        st.info("Enter a ticker in the sidebar and click 'Load Filing' to start.")

if __name__ == '__main__':
    main()