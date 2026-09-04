import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def process_pdf_files(pdf_directory: str):
    pdf_path = Path(pdf_directory)
    print(pdf_path)
    pdf_files = list(pdf_path.glob("**/*.pdf"))

    all_documents =[]
    
    print(f"Number of files in directory: {pdf_directory} is {len(pdf_files)}")

    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for document in documents:
                document.metadata["source_file"] = pdf_file.name
                document.metadata["file_type"] = "pdf"
            
            all_documents.extend(documents)

            print(f"Loaded {len(documents)} pages for file: {pdf_file}")
        except Exception as e:
            print(f" Error: {e}")

    print(f" Total documents loaded: {len(all_documents)}")

    return all_documents


