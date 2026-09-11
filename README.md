# rag-3.0

A small, fully local Retrieval-Augmented Generation (RAG) pipeline: load PDFs, chunk them, embed them, store the vectors, and ask natural-language questions about them — answered by a local LLM.

No API keys. No cloud bills. Everything runs on your machine.

## Why local, free tools?

Every piece of the stack that would normally cost money or require an account has a local, free substitute:

| Piece | Usual choice | This project uses | Why |
|---|---|---|---|
| Embeddings | OpenAI / Cohere API | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs on your CPU, downloaded once from Hugging Face, no API key |
| Vector database | Pinecone (cloud) | [Pinecone Local](https://docs.pinecone.io/guides/operations/local-development) (Docker) | Same client SDK and API, but running in a container on `localhost` — no account, no API key |
| LLM | OpenAI / Anthropic API | [Ollama](https://ollama.com) running a local model (e.g. `gemma4:12b`) | Runs entirely on your machine, no per-token cost, no data leaves your computer |

This makes the project easy to run, easy to experiment with, and free to leave running while you learn — nothing here needs a credit card.

## Project structure

The codebase is deliberately small and split by responsibility, so each piece can be read and understood in isolation:

```
src/
├── rag_3_0/          # Entry point — wires everything together (the "app")
│   └── __init__.py       main(): ingest once, then an interactive query loop
├── eval_harness/     # A second, independent entry point for testing quality
│   └── __init__.py       main(): run eval/dataset.json through the same pipeline
├── utils/            # Small, stateless helper functions
│   ├── pdf_loader.py     Load PDFs into LangChain Documents (one per page)
│   ├── chunker.py        Split Documents into overlapping text chunks
│   ├── config.py         All environment-driven settings, in one place
│   ├── logging_config.py Sets up console + file logging (import-time side effect)
│   └── timing.py         log_duration(): a context manager that times and logs a step
└── managers/         # Stateful clients — one class per external system
    ├── embedding_manager.py   Wraps sentence-transformers (text -> vector)
    ├── pinecone_manager.py    Wraps Pinecone (create/connect/upsert/query)
    └── llm_manager.py         Wraps Ollama (context + question -> answer)
```

The idea: `utils/` never talks to a network or a database — it's just data transformation, easy to test and easy to read. `managers/` is where all the "talk to an external service" code lives, one manager per service, each with a narrow, obvious job. `rag_3_0/__init__.py` is the only place that imports both and decides the order of operations. If you're learning how a RAG pipeline fits together, start there and follow the calls outward.

## How it works

```
PDFs  ->  pdf_loader  ->  chunker  ->  embedding_manager  ->  pinecone_manager (store)
                                                                     |
                                                                     v
                                             your question  ->  embedding_manager
                                                                     |
                                                                     v
                                                          pinecone_manager (search)
                                                                     |
                                                                     v
                                                retrieved chunks + question -> llm_manager -> answer
```

1. **Ingest** (runs once, only if the index is empty): PDFs in `test-pdf-files/` are loaded page-by-page, split into overlapping chunks, embedded locally, and upserted into the Pinecone index.
2. **Query loop**: each question you type is embedded the same way, used to search Pinecone for the most similar chunks, and those chunks are handed to the local LLM as context so it can answer in natural language, with sources listed underneath.

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Docker (for running Pinecone Local)
- [Ollama](https://ollama.com) installed and running, with a model pulled — e.g.:
  ```
  ollama pull gemma4:12b
  ```

## Usage

```bash
# 1. Start the local vector database
docker compose up -d

# 2. Copy the example env file (defaults work out of the box)
cp .env.example .env

# 3. Install dependencies
uv sync

# 4. Add your PDFs
#    Drop files into test-pdf-files/

# 5. Run
uv run rag-3-0
```

On first run, it ingests every PDF in `test-pdf-files/` into the vector store; on later runs it skips straight to the query loop. Then just ask questions:

```
Enter a search query (or 'exit' to quit):
> what are deliberative mini-publics
```

Type `exit` (or `quit`, or an empty line) to stop.

## Evaluation

There's a small eval harness for checking retrieval and answer quality against a known set of questions, separate from the interactive app:

```bash
uv run rag-3-0-eval
```

It reads test cases from [`eval/dataset.json`](eval/dataset.json) — each one a `question`, the `expected_source_file` the answer should come from, and a `reference_answer`:

```json
{
  "question": "What are deliberative mini-publics?",
  "expected_source_file": "Democracy.pdf",
  "reference_answer": "Small groups of randomly selected citizens..."
}
```

For each question it runs the exact same retrieval + generation path as the real app, then checks two independent things:

- **Retrieval**: did a chunk from `expected_source_file` actually come back from Pinecone, and at what rank? (hit rate + MRR — deterministic, no LLM involved)
- **Answer quality**: is the generated answer judged correct against `reference_answer`? The judge is the same local Ollama model, given a separate grading prompt and asked to reply `CORRECT`/`INCORRECT`.

Keeping these separate matters: high retrieval with low answer correctness points at prompting/generation, not the vector search. Requires the index to already be populated (run `uv run rag-3-0` first) — it evaluates against existing data rather than re-ingesting. Extend `eval/dataset.json` with your own questions as you add PDFs.

## Observability

Every stage of the pipeline is timed and logged, not just printed:

- **Console**: structured `logging` output (timestamp, level, module) for every stage — model loading, chunking, embedding, Pinecone connect/upsert/query, and LLM generation — each with how long it took, e.g. `LLM generation (gemma4:12b) took 12.95s`. Noisy third-party libraries (`httpx`, `huggingface_hub`, etc.) are quieted to warnings-only so your own signal isn't buried.
- **`logs/app.log`**: the same structured log, persisted across runs.
- **`logs/llm_calls.log`**: a separate, dedicated trace of every LLM call — the full system prompt, the full context + question sent, and the full raw response. Kept out of the console (too verbose for interactive use) but invaluable for figuring out *why* a specific answer came out wrong.

This is all local — no LangSmith/Langfuse account, just the standard library's `logging` module writing to files next to the code. Control verbosity with `LOG_LEVEL` (see below); `logs/` is gitignored.

## Configuration

All settings live in [`.env`](.env.example) — copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Purpose |
|---|---|---|
| `PINECONE_API_KEY` | `pclocal` | Placeholder key — Pinecone Local doesn't check it |
| `PINECONE_HOST` | `http://localhost:5080` | Pinecone Local's control-plane address |
| `PINECONE_INDEX_NAME` | `rag-3-0-index` | Index name (created automatically if missing) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Any `sentence-transformers` model name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Text-splitting parameters |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama server address |
| `OLLAMA_MODEL` | `gemma4:12b` | Any model you've pulled with `ollama pull` |
| `LOG_LEVEL` | `INFO` | Console log verbosity (e.g. `DEBUG`, `WARNING`); `logs/app.log` always captures `INFO`+ regardless |

## Notes

- Pinecone Local doesn't persist data — `docker compose down` clears the index. Just run `docker compose up -d` again and re-run the app to re-ingest.
- Swapping models is a one-line env change: any `sentence-transformers` model works for embeddings, and any model pulled into Ollama works for answers.
