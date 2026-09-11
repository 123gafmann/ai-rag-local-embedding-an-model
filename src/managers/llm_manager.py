import logging
import ollama

from utils.timing import log_duration

logger = logging.getLogger(__name__)
llm_io_logger = logging.getLogger("llm_io")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


class LLMManager:
    def __init__(self, model_name: str, host: str):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_response(self, query: str, context: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        llm_io_logger.info(f"REQUEST model={self.model_name}\n--- system ---\n{system_prompt}\n--- user ---\n{user_message}")

        try:
            logger.info(f"Generating response with {self.model_name}...")
            with log_duration(logger, f"LLM generation ({self.model_name})"):
                response = self.client.chat(model=self.model_name, messages=messages)
            answer = response["message"]["content"]
            llm_io_logger.info(f"RESPONSE model={self.model_name}\n{answer}")
            return answer
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            llm_io_logger.info(f"ERROR model={self.model_name}\n{e}")
            raise
