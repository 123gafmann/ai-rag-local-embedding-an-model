from . import logging_config  # noqa: F401  configures logging on import
from .pdf_loader import process_pdf_files
from .chunker import chunk_documents
from .timing import log_duration
from .config import (
    PINECONE_API_KEY,
    PINECONE_HOST,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    LOG_LEVEL,
)
