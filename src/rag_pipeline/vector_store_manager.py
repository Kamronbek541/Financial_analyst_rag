# src/rag_pipeline/vector_store_manager.py

import os # <--- ВОТ ИСПРАВЛЕНИЕ! Добавляем этот импорт.
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document

class VectorStoreManager:
    """
    Manages the creation and retrieval of the vector store.
    """
    def __init__(self, vector_store_path: str, embedding_model_name: str):
        """
        Initializes the VectorStoreManager.
        
        Args:
            vector_store_path (str): Directory to save/load the vector store.
            embedding_model_name (str): The name of the Hugging Face embedding model to use.
        """
        self.vector_store_path = vector_store_path
        # Initialize the embedding model. device='cpu' is fine for this model.
        # You can change to 'cuda' if you have a powerful GPU.
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.db = None

    def create_or_get_vector_store(self, documents: List[Document], force_recreate: bool = False):
        """
        Creates a new vector store from documents or loads an existing one.
        
        Args:
            documents (List[Document]): A list of document chunks to be added.
            force_recreate (bool): If True, deletes the existing store and creates a new one.
        
        Returns:
            Chroma: The Chroma vector store instance.
        """
        if force_recreate and os.path.exists(self.vector_store_path):
            print(f"Forcing recreation. Deleting existing vector store at: {self.vector_store_path}")
            shutil.rmtree(self.vector_store_path)

        # Здесь os.path.exists() теперь будет работать, потому что мы импортировали os
        if not os.path.exists(self.vector_store_path):
            print("No existing vector store found. Creating a new one...")
            if not documents:
                raise ValueError("Cannot create a new vector store without documents.")
            
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.vector_store_path
            )
            print(f"Successfully created and persisted vector store at: {self.vector_store_path}")
        else:
            print(f"Loading existing vector store from: {self.vector_store_path}")
            self.db = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_function
            )
        
        return self.db

    def get_retriever(self, search_k: int = 4):
        """
        Gets a retriever from the vector store.
        
        Args:
            search_k (int): The number of relevant documents to retrieve.
        
        Returns:
            A retriever object.
        """
        if not self.db:
            raise ValueError("Vector store is not initialized. Call create_or_get_vector_store first.")
        
        return self.db.as_retriever(search_kwargs={'k': search_k})