import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pclocal")
PINECONE_HOST = os.getenv("PINECONE_HOST", "http://localhost:5080")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-3-0-index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
