from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_documents = []

    for document in documents:
        chunks = splitter.split_text(document.page_content)
        print(f"Split {document.metadata.get('source_file')} page {document.metadata.get('page')} into {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = dict(document.metadata)
            chunk_metadata["chunk_index"] = i
            chunked_documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    print(f"Total chunks created: {len(chunked_documents)}")
    return chunked_documents
