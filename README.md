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
├── utils/            # Small, stateless helper functions
│   ├── pdf_loader.py     Load PDFs into LangChain Documents (one per page)
│   ├── chunker.py        Split Documents into overlapping text chunks
│   └── config.py         All environment-driven settings, in one place
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
> what is the explosives act about
```

Type `exit` (or `quit`, or an empty line) to stop.

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

## Notes

- Pinecone Local doesn't persist data — `docker compose down` clears the index. Just run `docker compose up -d` again and re-run the app to re-ingest.
- Swapping models is a one-line env change: any `sentence-transformers` model works for embeddings, and any model pulled into Ollama works for answers.
