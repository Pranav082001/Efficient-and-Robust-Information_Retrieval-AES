# BSI RAG Benchmark

Dense / BM25 / KG-RAG / Hybrid retrieval benchmark over BSI compliance documents.

## Setup

```bash
pip install "langchain>=0.3" langchain-community langchain-huggingface langchain-neo4j \
    langchain-ollama sentence-transformers transformers accelerate faiss-cpu rank_bm25 \
    neo4j beir pandas matplotlib tqdm
```

- **Neo4j** must be running (defaults to `bolt://localhost:7687`, user `neo4j`; set `NEO4J_PASSWORD` or export `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`).
- **LLM**: defaults to a local HuggingFace model (`meta-llama/Meta-Llama-3-8B-Instruct`), which is gated — run `huggingface-cli login` (or set `HF_TOKEN`) with an account that has accepted the license. To use Ollama instead, run `ollama serve`, `ollama pull llama3:8b`, and pass `--llm-backend ollama --llm-model llama3:8b`.

## Run

```bash
python run_pipeline.py                     # English trial run (20 queries)
python run_pipeline.py --full              # full English run (83 queries)
python run_pipeline.py --lang de --full    # full German run (199 queries)
python run_pipeline.py --subset-check      # quick entity-resolution ablation only
```

Useful overrides: `--llm-model <hf-repo-id>`, `--llm-backend {hf,ollama}`, `--n-queries N`.

See `bsi_rag/config.py` for every other tunable knob.

