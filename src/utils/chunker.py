import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .timing import log_duration

logger = logging.getLogger(__name__)


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_documents = []

    with log_duration(logger, f"Chunking {len(documents)} documents"):
        for document in documents:
            chunks = splitter.split_text(document.page_content)
            logger.debug(f"Split {document.metadata.get('source_file')} page {document.metadata.get('page')} into {len(chunks)} chunks")

            for i, chunk_text in enumerate(chunks):
                chunk_metadata = dict(document.metadata)
                chunk_metadata["chunk_index"] = i
                chunked_documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

    logger.info(f"Total chunks created: {len(chunked_documents)}")
    return chunked_documents
