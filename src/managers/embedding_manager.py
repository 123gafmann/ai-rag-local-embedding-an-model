import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from utils.timing import log_duration

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            with log_duration(logger, f"Loading model {self.model_name}"):
                self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Model loading exception: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        with log_duration(logger, f"Embedding {len(texts)} texts"):
            embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_embedding_dimension()
