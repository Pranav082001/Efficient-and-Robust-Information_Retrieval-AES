"""Load the BSI-HotpotQA-schema QA file + paragraph corpus, then build the
retrieval pool -- for a trial run (n_queries < all questions) the pool is
shrunk to the trial questions' gold paragraphs plus a few distractors, so KG
extraction cost scales with the trial rather than the full corpus.
"""
import json
import random
from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class Dataset:
    corpus: dict      # doc_id -> {"title": ..., "text": ...}
    queries: dict     # qid -> question text
    qrels: dict       # qid -> {doc_id: 1, ...}
    meta: dict        # qid -> raw QA item (for slicing by type/level)


@dataclass
class Pool:
    eval_qids: list
    pool_ids: list
    pool_texts: dict
    eval_qrels: dict


def load_dataset(cfg: Config) -> Dataset:
    corpus_units = json.load(open(cfg.corpus_file))
    data = json.load(open(cfg.qa_file))

    corpus = {u["title"]: {"title": u["title"], "text": " ".join(u["sentences"])}
              for u in corpus_units}
    queries = {it["_id"]: it["question"] for it in data}
    qrels = {it["_id"]: {t: 1 for t, _ in it["supporting_facts"]} for it in data}
    meta = {it["_id"]: it for it in data}

    return Dataset(corpus=corpus, queries=queries, qrels=qrels, meta=meta)


def summarize_dataset(ds: Dataset) -> str:
    return (f"corpus paragraphs   : {len(ds.corpus):,}\n"
            f"questions           : {len(ds.queries):,}\n"
            f"gold per question   : {np.mean([len(v) for v in ds.qrels.values()]):.2f}")


def make_text(d: dict) -> str:
    return (d.get("title", "") + ". " + d.get("text", "")).strip()


def build_pool(cfg: Config, ds: Dataset) -> Pool:
    all_doc_ids = list(ds.corpus)
    eval_qids = list(ds.queries)[: (cfg.n_queries or len(ds.queries))]

    if cfg.n_queries is not None and cfg.n_queries < len(ds.queries):
        rng = random.Random(cfg.seed)
        trial_gold = {d for q in eval_qids for d in ds.qrels[q]}
        remaining = [d for d in all_doc_ids if d not in trial_gold]
        n_distractors = min(20, len(remaining))
        pool_ids = sorted(trial_gold) + rng.sample(remaining, n_distractors)
    else:
        pool_ids = all_doc_ids

    pool_texts = {d: make_text(ds.corpus[d]) for d in pool_ids}
    eval_qrels = {q: ds.qrels[q] for q in eval_qids}

    return Pool(eval_qids=eval_qids, pool_ids=pool_ids, pool_texts=pool_texts, eval_qrels=eval_qrels)


def sel_by_meta(pool: Pool, ds: Dataset, key: str, val: str) -> list:
    """Questions in the pool whose top-level or metadata field `key` equals `val`
    (English items carry type/level top-level, German items may nest under metadata)."""
    return [q for q in pool.eval_qids
            if (ds.meta[q].get(key) or ds.meta[q].get("metadata", {}).get(key)) == val]
