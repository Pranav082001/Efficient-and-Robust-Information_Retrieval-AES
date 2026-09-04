"""Simple RAG (dense) + BM25 retrieval, plus the shared sentence embedder
(also reused by kg.py for entity resolution/seeding).
"""
import re

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from .config import Config
from .data_prep import Pool

# ---- shared embedder --------------------------------------------------------


def build_embedder(cfg: Config) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=cfg.emb_model,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def as_query(cfg: Config, text: str) -> str:
    return cfg.bge_query_instruction + text


# ---- dense -----------------------------------------------------------------


def build_vectorstore(embeddings: HuggingFaceEmbeddings, pool: Pool) -> FAISS:
    docs = [Document(page_content=pool.pool_texts[d], metadata={"doc_id": d})
            for d in pool.pool_ids]
    return FAISS.from_documents(docs, embeddings)


def dense_search(cfg: Config, vectorstore: FAISS, query: str, k: int | None = None) -> dict:
    hits = vectorstore.similarity_search_with_score(as_query(cfg, query), k=k or cfg.top_k)
    return {h.metadata["doc_id"]: float(-dist) for h, dist in hits}


def run_dense(cfg: Config, vectorstore: FAISS, pool: Pool, queries: dict) -> dict:
    return {q: dense_search(cfg, vectorstore, queries[q]) for q in pool.eval_qids}


# ---- BM25 --------------------------------------------------------------------


def tokenize(s: str) -> list:
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


def build_bm25(pool: Pool) -> BM25Okapi:
    return BM25Okapi([tokenize(pool.pool_texts[d]) for d in pool.pool_ids])


def bm25_search(bm25: BM25Okapi, pool: Pool, query: str, k: int) -> dict:
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return {pool.pool_ids[i]: float(scores[i]) for i in order}


def run_bm25(cfg: Config, bm25: BM25Okapi, pool: Pool, queries: dict) -> dict:
    return {q: bm25_search(bm25, pool, queries[q], cfg.top_k) for q in pool.eval_qids}
