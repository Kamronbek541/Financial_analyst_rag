from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class QASystem:
    """
    The core Question-Answering system using the RAG pipeline.
    """
    def __init__(self, retriever, llm_model_name: str = "gpt-3.5-turbo-0125"):
        """
        Initializes the QA System.
        
        Args:
            retriever: The retriever object from the VectorStoreManager.
            llm_model_name (str): The name of the OpenAI model to use.
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=llm_model_name)
        self.prompt = self._build_prompt()
        self.chain = self._build_chain()

    @staticmethod
    def _build_prompt() -> ChatPromptTemplate:
        """
        Builds the prompt template for the RAG chain.
        """
        template = """
        You are an expert financial analyst assistant. Your task is to answer questions based on the context from a company's 10-K filing.
        Provide a concise, factual answer based *only* on the provided context. If the context does not contain the answer, say "The provided context does not contain information about this."
        Do not use any outside knowledge.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        return PromptTemplate(input_variables=["context", "question"], template=template)

    def _build_chain(self):
        """
        Builds the full RAG chain using LangChain Expression Language (LCEL).
        """
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def ask_question(self, question: str) -> str:
        """
        Asks a question to the RAG chain.
        
        Args:
            question (str): The user's question.
            
        Returns:
            str: The generated answer.
        """
        return self.chain.invoke(question)