import logging
import uuid

from utils import (
    process_pdf_files,
    chunk_documents,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PINECONE_INDEX_NAME,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)
from managers import EmbeddingManager, PineconeManager, LLMManager

logger = logging.getLogger(__name__)


def ingest(embedding_manager: EmbeddingManager, pinecone_manager: PineconeManager):
    documents = process_pdf_files("test-pdf-files")
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["text"] = chunk.page_content[:1000]
        metadatas.append(metadata)

    pinecone_manager.upsert_documents(ids, embeddings, metadatas)

    stats = pinecone_manager.index.describe_index_stats()
    logger.info(f"Done. Index stats: {stats}")


def query_loop(embedding_manager: EmbeddingManager, pinecone_manager: PineconeManager, llm_manager: LLMManager):
    print("\nEnter a search query (or 'exit' to quit):")
    while True:
        query_text = input("> ").strip()
        if not query_text or query_text.lower() in ("exit", "quit"):
            break

        query_embedding = embedding_manager.generate_embeddings([query_text])[0]
        matches = pinecone_manager.query(query_embedding, top_k=8)

        if not matches:
            print("No results found.")
            continue

        context = "\n\n".join(match.get("metadata", {}).get("text", "") for match in matches)
        answer = llm_manager.generate_response(query_text, context)
        print(f"\n{answer}\n")

        print("Sources:")
        for i, match in enumerate(matches, start=1):
            metadata = match.get("metadata", {})
            print(f"{i}. score={match.get('score'):.4f} source={metadata.get('source_file')} page={metadata.get('page')}")


def main():
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
    pinecone_manager = PineconeManager(
        index_name=PINECONE_INDEX_NAME,
        dimension=embedding_manager.get_embedding_dimension(),
    )
    llm_manager = LLMManager(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)

    if pinecone_manager.get_vector_count() == 0:
        logger.info("Index is empty, ingesting PDFs...")
        ingest(embedding_manager, pinecone_manager)
    else:
        logger.info(f"Index already has {pinecone_manager.get_vector_count()} vectors, skipping ingest.")

    query_loop(embedding_manager, pinecone_manager, llm_manager)
