import logging
from typing import List, Dict, Any

from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec

from utils.config import PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME
from utils.timing import log_duration

logger = logging.getLogger(__name__)


class PineconeManager:
    def __init__(self, index_name: str = PINECONE_INDEX_NAME, dimension: int = 384):
        self.index_name = index_name
        self.dimension = dimension
        self.client = PineconeGRPC(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
        self.index = None
        self._connect_index()

    def _connect_index(self):
        try:
            existing_indexes = self.client.list_indexes().names()

            if self.index_name not in existing_indexes:
                logger.info(f"Index '{self.index_name}' not found, creating it with dimension {self.dimension}...")
                with log_duration(logger, f"Creating index {self.index_name}"):
                    self.client.create_index(
                        name=self.index_name,
                        vector_type="dense",
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                        deletion_protection="disabled",
                    )

            index_host = self.client.describe_index(name=self.index_name).host
            self.index = self.client.Index(host=index_host, grpc_config=GRPCClientConfig(secure=False))
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Pinecone index connection error: {e}")
            raise

    def upsert_documents(self, ids: List[str], vectors, metadatas: List[Dict[str, Any]], batch_size: int = 100):
        if self.index is None:
            raise ValueError("Index not connected")

        records = list(zip(ids, vectors, metadatas))
        logger.info(f"Upserting {len(records)} vectors in batches of {batch_size}...")

        with log_duration(logger, f"Upserting {len(records)} vectors"):
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                formatted_batch = [
                    {"id": vec_id, "values": vector.tolist(), "metadata": metadata}
                    for vec_id, vector, metadata in batch
                ]
                try:
                    self.index.upsert(vectors=formatted_batch)
                    logger.info(f"Upserted batch {i // batch_size + 1} ({len(formatted_batch)} vectors)")
                except Exception as e:
                    logger.error(f"Upsert error on batch {i // batch_size + 1}: {e}")

    def get_vector_count(self) -> int:
        if self.index is None:
            raise ValueError("Index not connected")
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def query(self, vector, top_k: int = 5):
        if self.index is None:
            raise ValueError("Index not connected")

        query_vector = vector.tolist() if hasattr(vector, "tolist") else vector
        with log_duration(logger, f"Pinecone query (top_k={top_k})"):
            results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return results.get("matches", [])
