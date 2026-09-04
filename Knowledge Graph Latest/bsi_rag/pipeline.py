"""End-to-end orchestration, wiring the four content modules together:
data_prep -> dense_bm25 -> kg -> evaluation.
"""
import random

import numpy as np
import torch

from . import data_prep, dense_bm25, evaluation, kg
from .config import Config


def resolve_device() -> str:
    """Device for the embedder only. When the LLM backend is Ollama it runs
    out-of-process, so it is never pinned by this choice; the HF backend
    (kg.build_llm) resolves its own device separately via cfg.hf_device_map."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


def run(cfg: Config, verbose: bool = True) -> dict:
    """Runs the full pipeline end to end and returns its artifacts and scores."""
    set_seed(cfg)
    cfg.device = resolve_device()

    ds = data_prep.load_dataset(cfg)
    pool = data_prep.build_pool(cfg, ds)
    if verbose:
        print(data_prep.summarize_dataset(ds))
        print(f"Evaluating {len(pool.eval_qids)} queries over a pool of {len(pool.pool_ids)} paragraphs.")

    # -- dense + bm25 -----------------------------------------------------------
    emb = dense_bm25.build_embedder(cfg)
    vectorstore = dense_bm25.build_vectorstore(emb, pool)
    dense_run = dense_bm25.run_dense(cfg, vectorstore, pool, ds.queries)

    bm25_index = dense_bm25.build_bm25(pool)
    bm25_run = dense_bm25.run_bm25(cfg, bm25_index, pool, ds.queries)

    # -- KG: extract, resolve, load, retrieve ------------------------------------
    chat = kg.build_llm(cfg)
    doc_triples = kg.extract_all(chat, cfg, pool)
    doc_triples = kg.add_structural_layer(pool, doc_triples)
    resolved = kg.resolve_entities(emb, cfg, doc_triples)
    if verbose:
        print(kg.summarize_resolution(resolved))

    graph = kg.connect(cfg)
    load_stats = kg.load_graph(graph, resolved)
    if verbose:
        print(f"Neo4j knowledge graph: {load_stats['n_entities']} entities, "
              f"{load_stats['n_rels']} relations, loaded from {load_stats['n_triples_loaded']} triples.")

    entity_index = kg.build_entity_index(emb, resolved.canon.values())
    kg_run, kg_seed_counts = kg.run_kg(cfg, graph, emb, entity_index, resolved, chat, pool,
                                       ds.queries, dense_bm25.tokenize)

    # -- hybrid -----------------------------------------------------------------
    hybrid_run = evaluation.rrf_fuse(cfg, pool, dense_run, bm25_run, kg_run, kg_seed_counts)

    # -- evaluate -----------------------------------------------------------------
    systems = {
        "Simple RAG (Dense)": dense_run,
        "BM25 (lexical)": bm25_run,
        "KG-RAG (Neo4j + LLM graph)": kg_run,
        "Hybrid (Dense+BM25+KG)": hybrid_run,
    }
    results = evaluation.score_all(cfg, pool, systems)
    slices = evaluation.slice_table(cfg, pool, ds, systems)
    rank_gap = evaluation.per_hop_rank_gap(cfg, vectorstore, pool, ds.queries)

    if verbose:
        print(results)
        print(slices)
        print(rank_gap)

    return {
        "dataset": ds, "pool": pool, "systems": systems, "results": results,
        "slices": slices, "rank_gap": rank_gap, "resolved_entities": resolved,
        "kg_seed_counts": kg_seed_counts, "graph": graph, "embeddings": emb, "chat": chat,
    }
