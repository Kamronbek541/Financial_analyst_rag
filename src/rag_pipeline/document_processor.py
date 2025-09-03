import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class DocumentProcessor:
    """
    Handles loading, cleaning, and splitting of documents.
    """
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

    def _clean_raw_submission_text(self, file_path: str) -> str:
        """
        Reads the raw full-submission.txt, extracts the core 10-K document,
        and cleans it of HTML tags.
        """
        print("Starting pre-processing of raw submission file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        doc_start_pattern = re.compile(r'<DOCUMENT>')
        doc_end_pattern = re.compile(r'</DOCUMENT>')
        type_pattern = re.compile(r'<TYPE>10-K[^\n]*')

        doc_starts = [match.end() for match in doc_start_pattern.finditer(raw_text)]
        doc_ends = [match.start() for match in doc_end_pattern.finditer(raw_text)]
        
        docs = list(zip(doc_starts, doc_ends))

        html_content = ''
        for doc_start, doc_end in docs:
            doc_content = raw_text[doc_start:doc_end]
            if type_pattern.search(doc_content):
                text_start_pattern = re.compile(r'<TEXT>')
                text_end_pattern = re.compile(r'</TEXT>')
                
                text_start = text_start_pattern.search(doc_content)
                text_end = text_end_pattern.search(doc_content)
                
                if text_start and text_end:
                    html_content = doc_content[text_start.end():text_end.start()]
                    break

        if not html_content:
            print("Warning: Could not find a 10-K document section in the file. Trying to clean the whole file.")
            soup = BeautifulSoup(raw_text, 'html.parser')
        else:
            print("Successfully extracted 10-K HTML section. Cleaning text...")
            soup = BeautifulSoup(html_content, 'html.parser')
            
        clean_text = soup.get_text("\n", strip=True)
        print("Pre-processing and cleaning complete.")
        return clean_text

    def load_and_split_document(self, file_path: str) -> List[Document]:
        """
        Loads a document, cleans it, and splits it into chunks.
        
        Args:
            file_path (str): The path to the 'full-submission.txt' file.
            
        Returns:
            List[Document]: A list of document chunks.
        """
        try:
            clean_text = self._clean_raw_submission_text(file_path)

            source_document = Document(
                page_content=clean_text,
                metadata={"source": file_path}
            )

            split_docs = self.text_splitter.split_documents([source_document])
            
            print(f"Successfully loaded, cleaned, and split '{file_path}' into {len(split_docs)} clean chunks.")
            
            return split_docs
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return []