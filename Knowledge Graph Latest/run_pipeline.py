#!/usr/bin/env python
"""CLI entrypoint for the pipeline (see bsi_rag/). By default the LLM (triple
+ query-entity extraction) runs locally via transformers/LangChain
(langchain_huggingface), loading meta-llama/Meta-Llama-3-8B-Instruct. That's a
gated HF repo: run `huggingface-cli login` (or set HF_TOKEN) with an account
that has accepted Meta's license first. Pass --llm-backend ollama to use
ChatOllama instead (requires a running `ollama serve` with the model pulled).
Either way, a running Neo4j instance is required.

Usage:
  python run_pipeline.py                                        # English trial run (20 queries), HF backend, Llama-3-8B-Instruct
  python run_pipeline.py --llm-model Qwen/Qwen2.5-3B-Instruct    # smaller HF model if 8B doesn't fit in RAM
  python run_pipeline.py --llm-backend ollama --llm-model llama3:8b   # Ollama backend
  python run_pipeline.py --full                                 # full English run (all 83 queries, full 105-paragraph pool)
  python run_pipeline.py --lang de --full                       # German split (199 queries, 163 paragraphs)
  python run_pipeline.py --subset-check                         # raw-vs-resolved-entities ablation only
"""
import argparse
import sys

from bsi_rag.config import Config
from bsi_rag import data_prep, dense_bm25, evaluation, kg, pipeline


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.lang == "de":
        cfg.qa_file = "bsi_hotpotqa_all.json"
        cfg.corpus_file = "bsi_paragraph_corpus.json"
    if args.full:
        cfg.n_queries = None
    if args.n_queries is not None:
        cfg.n_queries = args.n_queries
    if args.llm_backend:
        cfg.llm_backend = args.llm_backend
    if args.llm_model:
        cfg.llm_model = args.llm_model
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["en", "de"], default="en")
    parser.add_argument("--full", action="store_true", help="evaluate all questions over the full pool")
    parser.add_argument("--n-queries", type=int, default=None, help="override trial size explicitly")
    parser.add_argument("--llm-backend", choices=["hf", "ollama"], default=None,
                         help="hf (default, local transformers) or ollama")
    parser.add_argument("--llm-model", default=None,
                         help="HF repo id (hf backend, e.g. Qwen/Qwen2.5-3B-Instruct) "
                              "or Ollama tag (ollama backend, e.g. llama3:8b)")
    parser.add_argument("--subset-check", action="store_true",
                         help="run only the raw-vs-resolved-entities ablation")
    args = parser.parse_args()

    cfg = build_config(args)

    if args.subset_check:
        cfg.device = pipeline.resolve_device()
        pipeline.set_seed(cfg)
        ds = data_prep.load_dataset(cfg)
        pool = data_prep.build_pool(cfg, ds)
        emb = dense_bm25.build_embedder(cfg)
        chat = kg.build_llm(cfg)
        print(evaluation.run_subset_check(cfg, ds, pool, chat, emb))
        return 0

    result = pipeline.run(cfg)
    print("\n=== Results ===")
    print(result["results"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
