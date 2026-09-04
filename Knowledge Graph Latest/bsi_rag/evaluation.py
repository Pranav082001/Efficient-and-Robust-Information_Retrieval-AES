"""Fusion + evaluation: Reciprocal Rank Fusion, BEIR/AllSupport@k scoring,
slice analysis by question type/difficulty, and the raw-vs-resolved-entities
ablation check.
"""
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm.auto import tqdm

from . import kg
from .config import Config
from .data_prep import Dataset, Pool, sel_by_meta
from .dense_bm25 import dense_search, tokenize

# =============================================================================
# Hybrid via weighted Reciprocal Rank Fusion
#
# Plain RRF fuses rank position, not relevance, so a paragraph the weaker KG
# channel ranks low can still add enough fused score to push a stronger
# dense/BM25 candidate out of the top 2. Two mitigations:
#   1. Weighted RRF: the KG channel's weight (rrf_weight_kg) defaults below
#      dense/BM25's, so it can help when it agrees but can't dilute as much
#      when it doesn't.
#   2. Per-query KG gating: if the KG retriever matched fewer than
#      rrf_min_kg_seeds entities for a query, that query's fusion falls back
#      to dense+BM25 only instead of fusing in a ranking built on no evidence.
# =============================================================================


def rrf_fuse(cfg: Config, pool: Pool, dense_run: dict, bm25_run: dict,
             kg_run: dict, kg_seed_counts: dict | None = None) -> dict:
    weights = {"dense": cfg.rrf_weight_dense, "bm25": cfg.rrf_weight_bm25, "kg": cfg.rrf_weight_kg}
    fused = {}
    for q in pool.eval_qids:
        runs = [("dense", dense_run), ("bm25", bm25_run)]
        if kg_seed_counts is None or kg_seed_counts.get(q, 0) >= cfg.rrf_min_kg_seeds:
            runs.append(("kg", kg_run))

        agg = defaultdict(float)
        for name, run in runs:
            ranked = sorted(run.get(q, {}).items(), key=lambda x: -x[1])
            for rank, (doc_id, _) in enumerate(ranked):
                agg[doc_id] += weights[name] / (cfg.rrf_k + rank + 1)
        fused[q] = dict(sorted(agg.items(), key=lambda x: -x[1])[: cfg.top_k])
    return fused


# =============================================================================
# BEIR metrics + AllSupport@k
# =============================================================================


def all_support(eval_qrels: dict, run: dict, k: int) -> float:
    hits = []
    for q, gold in eval_qrels.items():
        top = sorted(run.get(q, {}).items(), key=lambda x: -x[1])[:k]
        hits.append(float(set(gold) <= {d for d, _ in top}))
    return float(np.mean(hits))


def score(cfg: Config, eval_qrels: dict, run: dict) -> dict:
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(eval_qrels, run, [2, cfg.top_k])
    return {
        f"nDCG@{cfg.top_k}": ndcg[f"NDCG@{cfg.top_k}"],
        f"Recall@{cfg.top_k}": recall[f"Recall@{cfg.top_k}"],
        "MAP": _map[f"MAP@{cfg.top_k}"],
        f"P@{cfg.top_k}": precision[f"P@{cfg.top_k}"],
        "AllSupport@2": all_support(eval_qrels, run, 2),
        f"AllSupport@{cfg.top_k}": all_support(eval_qrels, run, cfg.top_k),
    }


def score_all(cfg: Config, pool: Pool, systems: dict) -> pd.DataFrame:
    return pd.DataFrame({name: score(cfg, pool.eval_qrels, run) for name, run in systems.items()}).T.round(4)


# =============================================================================
# Slice analysis + per-hop rank gap
# =============================================================================


def slice_table(cfg: Config, pool: Pool, ds: Dataset, systems: dict) -> pd.DataFrame:
    rows = {}
    for key, val in [("type", "bridge"), ("type", "comparison"),
                      ("level", "easy"), ("level", "medium"), ("level", "hard")]:
        qs = sel_by_meta(pool, ds, key, val)
        if not qs:
            continue
        sub_qrels = {q: pool.eval_qrels[q] for q in qs}
        rows[f"{key}={val} (n={len(qs)})"] = {
            name: round(score(cfg, sub_qrels, {q: run[q] for q in qs})[f"AllSupport@{cfg.top_k}"], 3)
            for name, run in systems.items()
        }
    return pd.DataFrame(rows).T


def per_hop_rank_gap(cfg: Config, vectorstore, pool: Pool, queries: dict) -> dict:
    full = {q: dense_search(cfg, vectorstore, queries[q], k=len(pool.pool_ids)) for q in pool.eval_qids}
    first, last = [], []
    for q in pool.eval_qids:
        order = [d for d, _ in sorted(full[q].items(), key=lambda x: -x[1])]
        rk = sorted(order.index(g) + 1 for g in pool.eval_qrels[q])
        first.append(rk[0]); last.append(rk[-1])
    return {"median_rank_easiest_gold": float(np.median(first)),
            "median_rank_hardest_gold": float(np.median(last))}


# =============================================================================
# Quick subset check: does entity resolution actually help?
# =============================================================================


def _all_support_local(qrels: dict, run: dict, qs: list, k: int) -> float:
    hits = [float(set(qrels[q]) <= {d for d, _ in sorted(run.get(q, {}).items(), key=lambda x: -x[1])[:k]})
            for q in qs]
    return float(np.mean(hits))


def run_subset_check(cfg: Config, ds: Dataset, pool: Pool, chat, embeddings,
                      n_queries: int = 12, n_distractors: int = 10) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    bridge_qs = sel_by_meta(pool, ds, "type", "bridge")
    test_qids = rng.sample(bridge_qs, min(n_queries, len(bridge_qs)))

    test_gold_docs = {d for q in test_qids for d in ds.qrels[q]}
    remaining = [d for d in pool.pool_ids if d not in test_gold_docs]
    test_pool = sorted(test_gold_docs) + rng.sample(remaining, min(n_distractors, len(remaining)))

    test_triples = {d: kg.extract_triples(chat, cfg, pool.pool_texts[d])
                     for d in tqdm(test_pool, desc="subset extraction")}

    raw_entities = {e for t in test_triples.values() for h, _, tt in t for e in (h, tt)}
    raw_index = kg.build_entity_index(embeddings, raw_entities)
    raw_doc_freq: dict = defaultdict(set)
    for d, triples in test_triples.items():
        for h, _, t in triples:
            raw_doc_freq[h].add(d); raw_doc_freq[t].add(d)
    raw_doc_freq = {e: len(v) for e, v in raw_doc_freq.items()}

    resolved = kg.resolve_entities(embeddings, cfg, test_triples)
    resolved_index = kg.build_entity_index(embeddings, resolved.canon.values())

    raw_run = {q: kg.kg_search_local(embeddings, raw_index, cfg, chat, test_triples, raw_doc_freq,
                                      ds.queries[q], cfg.kg_hops, cfg.top_k, tokenize) for q in test_qids}
    resolved_run = {q: kg.kg_search_local(embeddings, resolved_index, cfg, chat, resolved.triples_by_doc,
                                           resolved.doc_freq, ds.queries[q], cfg.kg_hops, cfg.top_k, tokenize)
                     for q in test_qids}

    return pd.DataFrame({
        "raw (no entity resolution)": {
            "AllSupport@2": _all_support_local(ds.qrels, raw_run, test_qids, 2),
            f"AllSupport@{cfg.top_k}": _all_support_local(ds.qrels, raw_run, test_qids, cfg.top_k),
        },
        "resolved (complete-linkage fix)": {
            "AllSupport@2": _all_support_local(ds.qrels, resolved_run, test_qids, 2),
            f"AllSupport@{cfg.top_k}": _all_support_local(ds.qrels, resolved_run, test_qids, cfg.top_k),
        },
    }).T
