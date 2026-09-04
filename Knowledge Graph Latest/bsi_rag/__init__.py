"""BSI multi-hop RAG benchmark: Dense / BM25 / KG-RAG / Hybrid.

Module map:
  config.py       -> every tunable knob
  data_prep.py    -> load QA/corpus, build the retrieval pool
  dense_bm25.py   -> shared embedder, dense/FAISS retrieval, BM25
  kg.py           -> LLM triple extraction, structural ID layer, entity
                      resolution, Neo4j load, query seeding, graph retrieval
  evaluation.py   -> RRF fusion, BEIR/AllSupport scoring, slice analysis
  pipeline.py     -> end-to-end run, used by run_pipeline.py
"""
