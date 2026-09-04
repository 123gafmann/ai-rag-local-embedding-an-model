import ollama

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


class LLMManager:
    def __init__(self, model_name: str, host: str):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_response(self, query: str, context: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        try:
            print(f"Generating response with {self.model_name}...")
            response = self.client.chat(model=self.model_name, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            print(f"LLM generation error: {e}")
            raise
