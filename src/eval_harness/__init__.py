import json
import logging

from utils import EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME, OLLAMA_HOST, OLLAMA_MODEL
from managers import EmbeddingManager, PineconeManager, LLMManager

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI-generated answer against a reference answer for a question. "
    "Reply with exactly one word on the first line: CORRECT if the generated answer captures "
    "the key information in the reference answer, or INCORRECT if it does not. "
    "On the next line, give a one-sentence reason."
)


def load_dataset(path: str = "eval/dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(llm_manager: LLMManager, question: str, reference_answer: str, generated_answer: str) -> bool:
    context = f"Question: {question}\nReference answer: {reference_answer}\nGenerated answer: {generated_answer}"
    verdict = llm_manager.generate_response(
        "Is the generated answer correct?",
        context,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    print(f"   Judge: {verdict.strip().splitlines()[0]}")
    return verdict.strip().upper().startswith("CORRECT")


def main():
    dataset = load_dataset()
    logger.info(f"Loaded {len(dataset)} eval cases")

    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
    pinecone_manager = PineconeManager(
        index_name=PINECONE_INDEX_NAME,
        dimension=embedding_manager.get_embedding_dimension(),
    )

    if pinecone_manager.get_vector_count() == 0:
        print("Index is empty. Run `uv run rag-3-0` first to ingest PDFs before evaluating.")
        return

    llm_manager = LLMManager(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)

    retrieval_hits = 0
    reciprocal_ranks = []
    answer_correct = 0

    for i, case in enumerate(dataset, start=1):
        question = case["question"]
        expected_source = case["expected_source_file"]
        reference_answer = case["reference_answer"]

        print(f"\n[{i}/{len(dataset)}] {question}")

        query_embedding = embedding_manager.generate_embeddings([question])[0]
        matches = pinecone_manager.query(query_embedding, top_k=5)

        rank = next(
            (idx for idx, match in enumerate(matches, start=1)
             if match.get("metadata", {}).get("source_file") == expected_source),
            None,
        )
        if rank is not None:
            retrieval_hits += 1
            reciprocal_ranks.append(1 / rank)
            print(f"   Retrieval: HIT (rank {rank})")
        else:
            reciprocal_ranks.append(0)
            print(f"   Retrieval: MISS (expected {expected_source})")

        context = "\n\n".join(match.get("metadata", {}).get("text", "") for match in matches)
        generated_answer = llm_manager.generate_response(question, context)
        print(f"   Answer: {generated_answer[:150]}")

        if judge_answer(llm_manager, question, reference_answer, generated_answer):
            answer_correct += 1

    total = len(dataset)
    print("\n=== Eval Summary ===")
    print(f"Retrieval hit rate: {retrieval_hits}/{total} ({retrieval_hits / total:.0%})")
    print(f"Retrieval MRR: {sum(reciprocal_ranks) / total:.2f}")
    print(f"Answer correctness (LLM judge): {answer_correct}/{total} ({answer_correct / total:.0%})")
