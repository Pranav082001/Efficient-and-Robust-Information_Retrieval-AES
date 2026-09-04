"""Knowledge-Graph RAG. One file, organized top to bottom exactly like the
pipeline flows: LLM client -> triple extraction -> structural ID layer ->
entity resolution -> Neo4j load -> query-side seeding -> graph retrieval.

Three deliberate design choices worth knowing about here:

  1. Entity resolution (see resolve_entities) uses complete-linkage
     clustering, not single-linkage. Single-linkage chains transitively --
     if sim(A,B) and sim(B,C) both clear the threshold, A and C get merged
     even if sim(A,C) doesn't -- which can collapse meaningfully different
     entities (e.g. "local attacker" and "remote attacker") into one hub
     node mentioned in nearly every threat paragraph. A hub seeds against
     almost any query and floods graph traversal with irrelevant documents.
     Complete-linkage (merge only if *every* cross-cluster pair clears the
     threshold) plus a hard cluster-size cap block this without needing a
     hand-tuned stopword list.
  2. Query-side seeding (see extract_query_entities, match_seeds) uses its
     own keyphrase-extraction prompt rather than the paragraph
     triple-extraction prompt, since a question isn't a factual assertion to
     extract a head|relation|tail from. Phrases are matched into the graph
     by normalized-exact match first, falling back to embedding cosine
     similarity against canonical entity names (capped top-N above a
     threshold) rather than substring containment.
  3. IDF damping (see _damp): entities mentioned in many documents contribute
     a damped score (1/log2(2+doc_freq)), the same intuition as BM25's IDF,
     so a residual hub entity can't dominate purely by frequency.
"""
import math
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_neo4j import Neo4jGraph
from tqdm.auto import tqdm

from .config import Config
from .data_prep import Pool

# =============================================================================
# LLM client -- HuggingFace via LangChain by default; Ollama optional.
#
# cfg.llm_backend selects the backend; both expose the same BaseChatModel
# interface (.invoke(messages).content), so llm_call() and everything
# downstream (triple extraction, query-entity extraction) doesn't care which
# one is active. Imports are lazy per-backend so installing only one of
# langchain-ollama / transformers+accelerate is enough to run the other.
# =============================================================================


def build_llm(cfg: Config):
    if cfg.llm_backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_base_url, temperature=0)
    return _build_hf_chat(cfg)


def _build_hf_chat(cfg: Config):
    """Local transformers model wrapped as a LangChain chat model. Note: unlike
    Ollama (which runs pre-quantized GGUF weights), this loads full/half-precision
    weights via `from_pretrained`, so an 8B-class model can need >=16GB just to
    materialize on a memory-constrained machine -- pick cfg.llm_model accordingly
    (the default is a 3B-class instruct model for that reason) or pass a smaller
    checkpoint / a quantized one your transformers install supports."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model,
        torch_dtype=cfg.hf_dtype,          # "auto" lets transformers pick per-checkpoint
        device_map=cfg.hf_device_map,      # "auto" requires `accelerate` (already a dependency)
    )
    gen = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.hf_max_new_tokens,
        do_sample=False,                   # greedy, the HF equivalent of ChatOllama's temperature=0
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen))


def llm_call(chat, system: str, user: str) -> str:
    return chat.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content


# =============================================================================
# Triple extraction (paragraphs)
# =============================================================================

EXTRACT_SYS = (
    "You extract facts from English IT-security texts. "
    "Output ONLY lines in the format: head | relation | tail . "
    "Keep head and tail as short noun phrases (2-4 words). "
    "Preserve identifiers exactly as written (e.g. O.Firewall, T.DataModificationWAN). "
    "No comments, no numbering, no duplicate lines."
)
EXTRACT_FEWSHOT = (
    "Text: The firewall objective O.Firewall protects the LMN and HAN from threats "
    "originating in the WAN and blocks unauthorised connections.\n"
    "o.firewall | protects | lmn and han\n"
    "o.firewall | blocks | unauthorised connections\n\n"
    "Text: A remote attacker on the WAN may attempt to modify firmware updates in transit, "
    "which is countered by verifying the update's digital signature before installation.\n"
    "remote attacker | attempts | modify firmware updates\n"
    "digital signature verification | counters | firmware modification\n\n"
    "Text: The TOE stores metering data with integrity and confidentiality protection, "
    "and only authorised external entities may access this data via the WAN interface.\n"
    "toe | stores | metering data\n"
    "authorised external entities | may access | metering data\n\n"
    "Text: {doc}\n"
)

# Entities this short/generic carry almost no discriminative signal and are
# exactly the kind of string that later gets embedding-clustered into a hub.
_MIN_ENTITY_LEN = 3
_GENERIC_ENTITIES = {
    "it", "this", "that", "these", "those", "system", "data", "information",
    "process", "component", "the toe", "toe",
}


def _clean_entity(e: str) -> str:
    return re.sub(r"\s+", " ", e.strip().lower())


def parse_triples(text: str) -> list:
    seen, triples = set(), []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        h, r, t = (_clean_entity(parts[0]), parts[1].strip().lower(), _clean_entity(parts[2]))
        if len(h) < _MIN_ENTITY_LEN or len(t) < _MIN_ENTITY_LEN:
            continue
        if h in _GENERIC_ENTITIES or t in _GENERIC_ENTITIES:
            continue
        if h == t or len(h) >= 80 or len(t) >= 80:
            continue
        key = (h, r, t)
        if key in seen:
            continue
        seen.add(key)
        triples.append(key)
    return triples


def extract_triples(chat, cfg: Config, doc_text: str) -> list:
    try:
        raw = llm_call(chat, EXTRACT_SYS, EXTRACT_FEWSHOT.format(doc=doc_text[: cfg.kg_max_chars]))
        return parse_triples(raw)
    except Exception:
        return []


def extract_all(chat, cfg: Config, pool: Pool) -> dict:
    doc_triples = {}
    for d in tqdm(pool.pool_ids, desc="LLM KG extraction"):
        doc_triples[d] = extract_triples(chat, cfg, pool.pool_texts[d])
    return doc_triples


# =============================================================================
# Structural ID layer (deterministic, regex over T./O./OE./A. identifiers)
# =============================================================================

ID_PATTERN = re.compile(r"\b(?:OE|[TOA])\.[A-Za-z][A-Za-z0-9]*\b")


def extract_ids(text: str) -> list:
    return sorted({m.lower() for m in ID_PATTERN.findall(text)})


def add_structural_layer(pool: Pool, doc_triples: dict) -> dict:
    for d in pool.pool_ids:
        ids = extract_ids(pool.pool_texts[d])
        pairs = [(a, "co-referenced-with", b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        doc_triples.setdefault(d, []).extend(pairs)
    return doc_triples


# =============================================================================
# Entity resolution (complete-linkage clustering)
# =============================================================================


@dataclass
class ResolvedEntities:
    triples_by_doc: dict   # doc_id -> [(head, relation, tail), ...], canonicalized
    canon: dict             # raw entity string -> canonical entity string
    doc_freq: dict          # canonical entity -> number of distinct docs it appears in


def _complete_linkage_clusters(sim: np.ndarray, threshold: float, max_cluster_size: int) -> list:
    n = sim.shape[0]
    clusters = [{i} for i in range(n)]
    owner = list(range(n))

    pairs = [(sim[i, j], i, j) for i in range(n) for j in range(i + 1, n) if sim[i, j] >= threshold]
    pairs.sort(reverse=True)

    for _, i, j in pairs:
        ci, cj = owner[i], owner[j]
        if ci == cj:
            continue
        members_i, members_j = clusters[ci], clusters[cj]
        if len(members_i) + len(members_j) > max_cluster_size:
            continue
        if min(sim[a, b] for a in members_i for b in members_j) < threshold:
            continue
        members_i |= members_j
        for m in members_j:
            owner[m] = ci
        clusters[cj] = set()

    return [clusters[c] for c in {owner[i] for i in range(n)}]


def resolve_entities(embeddings, cfg: Config, triples_by_doc: dict) -> ResolvedEntities:
    ents = sorted({h for t in triples_by_doc.values() for h, _, _ in t} |
                  {tt for t in triples_by_doc.values() for _, _, tt in t})
    if not ents:
        return ResolvedEntities(triples_by_doc=dict(triples_by_doc), canon={}, doc_freq={})

    vecs = np.array(embeddings.embed_documents(ents))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    sim = vecs @ vecs.T

    clusters = _complete_linkage_clusters(sim, cfg.entity_sim_threshold, cfg.entity_max_cluster_size)

    canon = {}
    for members in clusters:
        rep = min((ents[i] for i in members), key=len)
        for i in members:
            canon[ents[i]] = rep

    resolved = {d: [(canon[h], r, canon[t]) for h, r, t in triples]
                for d, triples in triples_by_doc.items()}

    doc_freq: dict = defaultdict(set)
    for d, triples in resolved.items():
        for h, _, t in triples:
            doc_freq[h].add(d); doc_freq[t].add(d)

    return ResolvedEntities(triples_by_doc=resolved, canon=canon,
                             doc_freq={e: len(docs) for e, docs in doc_freq.items()})


def summarize_resolution(res: ResolvedEntities) -> str:
    n_raw, n_resolved = len(res.canon), len(set(res.canon.values()))
    merged = defaultdict(list)
    for raw, rep in res.canon.items():
        if raw != rep:
            merged[rep].append(raw)
    lines = [f"Entity resolution: {n_raw} raw entity strings -> {n_resolved} canonical "
             f"entities ({n_raw - n_resolved} merged)."]
    for rep, raws in list(merged.items())[:8]:
        lines.append(f"  {rep!r:35s} <- {raws}")
    if res.doc_freq:
        lines.append(f"  most-mentioned entities (post-resolution): "
                      f"{sorted(res.doc_freq.items(), key=lambda x: -x[1])[:5]}")
    return "\n".join(lines)


# =============================================================================
# Neo4j graph load
# =============================================================================


def connect(cfg: Config) -> Neo4jGraph:
    return Neo4jGraph(url=cfg.neo4j_uri, username=cfg.neo4j_user, password=cfg.neo4j_password,
                       refresh_schema=False)


def load_graph(graph: Neo4jGraph, resolved: ResolvedEntities) -> dict:
    graph.query("MATCH (n) DETACH DELETE n")
    graph.query("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")

    rows = [{"doc": d, "head": h, "relation": r, "tail": t}
            for d, triples in resolved.triples_by_doc.items() for h, r, t in triples]

    graph.query(
        """
        UNWIND $rows AS row
        MERGE (h:Entity {name: row.head})
        MERGE (t:Entity {name: row.tail})
        MERGE (h)-[:REL {type: row.relation}]->(t)
        MERGE (doc:Document {id: row.doc})
        MERGE (h)-[:MENTIONED_IN]->(doc)
        MERGE (t)-[:MENTIONED_IN]->(doc)
        """,
        params={"rows": rows},
    )
    n_entities = graph.query("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]
    n_rels = graph.query("MATCH ()-[r:REL]->() RETURN count(r) AS n")[0]["n"]
    return {"n_entities": n_entities, "n_rels": n_rels, "n_triples_loaded": len(rows)}


# =============================================================================
# Query-side seeding (dedicated prompt + embedding match)
# =============================================================================

QUERY_ENTITY_SYS = (
    "You extract the key technical terms and identifiers from an IT-security "
    "question -- the concrete objects, control IDs, and concepts it is asking "
    "about. Output ONLY the phrases, one per line, no numbering, no commentary. "
    "2-5 phrases is typical. Prefer short noun phrases (1-4 words) and preserve "
    "identifiers exactly as written (e.g. O.Firewall, T.DataModificationWAN)."
)
QUERY_ENTITY_FEWSHOT = (
    "Question: A remote attacker in the WAN who tries to modify a firmware update is "
    "stopped by a combination of two objectives; what does the connection-defining one "
    "require regarding other services on the WAN interface?\n"
    "remote attacker\n"
    "firmware update\n"
    "WAN interface\n"
    "connection-defining objective\n\n"
    "Question: {q}\n"
)
_MIN_PHRASE_LEN = 3


def _parse_phrases(text: str) -> list:
    out = []
    for line in text.splitlines():
        p = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip().lower()
        if _MIN_PHRASE_LEN <= len(p) < 80:
            out.append(p)
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def extract_query_entities(chat, query: str, tokenize_fallback) -> list:
    try:
        phrases = _parse_phrases(llm_call(chat, QUERY_ENTITY_SYS, QUERY_ENTITY_FEWSHOT.format(q=query)))
    except Exception:
        phrases = []
    if phrases:
        return phrases
    # last-resort lexical fallback, only reached when the dedicated prompt found nothing
    return [w for w in tokenize_fallback(query) if len(w) > 4]


@dataclass
class EntityIndex:
    names: list
    vecs: np.ndarray  # L2-normalized, shape (n, d)


def build_entity_index(embeddings, entity_names) -> EntityIndex:
    names = sorted(set(entity_names))
    if not names:
        return EntityIndex(names=[], vecs=np.zeros((0, 1)))
    vecs = np.array(embeddings.embed_documents(names))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EntityIndex(names=names, vecs=vecs / norms)


def match_seeds(embeddings, index: EntityIndex, cfg: Config, phrases: list) -> list:
    if not index.names:
        return []
    name_set = set(index.names)
    matched = {p for p in phrases if p in name_set}
    unmatched = [p for p in phrases if p not in name_set]

    if unmatched:
        qvecs = np.array(embeddings.embed_documents(unmatched))
        qnorms = np.linalg.norm(qvecs, axis=1, keepdims=True)
        qnorms[qnorms == 0] = 1.0
        sims = (qvecs / qnorms) @ index.vecs.T
        for row in sims:
            for i in np.argsort(row)[::-1][: cfg.seed_top_n_per_phrase]:
                if row[i] >= cfg.seed_sim_threshold:
                    matched.add(index.names[i])
    return sorted(matched)


# =============================================================================
# Graph-augmented retrieval (IDF damping)
# =============================================================================

SEEDS_TO_DOCS_QUERY = """
UNWIND $seeds AS s
MATCH (seed:Entity {name: s})-[:MENTIONED_IN]->(doc:Document)
RETURN DISTINCT seed.name AS entity, doc.id AS doc_id
"""

EXPANDED_TO_DOCS_QUERY = """
UNWIND $seeds AS s
MATCH (seed:Entity {name: s})
WITH collect(DISTINCT seed) AS seedNodes
UNWIND seedNodes AS sn
MATCH p = (sn)-[:REL*1..%(hops)d]-(nbr:Entity)
WHERE NOT nbr IN seedNodes
WITH nbr, any(r IN relationships(p) WHERE r.type = 'co-referenced-with') AS used_struct
WITH nbr, max(CASE WHEN used_struct THEN true ELSE false END) AS is_struct
MATCH (nbr)-[:MENTIONED_IN]->(doc:Document)
RETURN DISTINCT nbr.name AS entity, is_struct AS is_struct, doc.id AS doc_id
"""


def _damp(doc_freq: dict, entity: str, use_idf: bool) -> float:
    if not use_idf:
        return 1.0
    return 1.0 / math.log2(2 + doc_freq.get(entity, 1))  # +2 -> single-doc entity gets damp=1.0


def kg_search(graph, embeddings, index: EntityIndex, resolved: ResolvedEntities,
              cfg: Config, chat, query: str, k: int, tokenize) -> tuple:
    """Returns (ranked_docs, n_seeds_matched); n_seeds_matched lets fusion logic
    gate a query's KG contribution without re-running seed extraction."""
    seeds = match_seeds(embeddings, index, cfg, extract_query_entities(chat, query, tokenize))
    if not seeds:
        return {}, 0

    scores: dict = {}
    for row in graph.query(SEEDS_TO_DOCS_QUERY, params={"seeds": seeds}):
        w = cfg.kg_seed_weight * _damp(resolved.doc_freq, row["entity"], cfg.kg_use_idf_damping)
        scores[row["doc_id"]] = scores.get(row["doc_id"], 0.0) + w

    if cfg.kg_hops > 0:
        expand_query = EXPANDED_TO_DOCS_QUERY % {"hops": cfg.kg_hops}
        for row in graph.query(expand_query, params={"seeds": seeds}):
            base = cfg.kg_struct_weight if row["is_struct"] else 1.0
            w = base * _damp(resolved.doc_freq, row["entity"], cfg.kg_use_idf_damping)
            scores[row["doc_id"]] = scores.get(row["doc_id"], 0.0) + w

    return dict(sorted(scores.items(), key=lambda x: -x[1])[:k]), len(seeds)


def run_kg(cfg: Config, graph, embeddings, index: EntityIndex, resolved: ResolvedEntities,
           chat, pool: Pool, queries: dict, tokenize) -> tuple:
    """Returns (kg_run, kg_seed_counts), both keyed by qid."""
    kg_run, kg_seed_counts = {}, {}
    for q in pool.eval_qids:
        ranked, n_seeds = kg_search(graph, embeddings, index, resolved, cfg, chat, queries[q], cfg.top_k, tokenize)
        kg_run[q] = ranked
        kg_seed_counts[q] = n_seeds
    return kg_run, kg_seed_counts


# =============================================================================
# Quick subset check: does entity resolution actually help? Pure-Python BFS
# mirror of kg_search's logic, no Neo4j round-trip needed.
# =============================================================================


def build_adjacency(triples_by_doc: dict):
    adj, adj_struct, ent_docs = defaultdict(set), defaultdict(set), defaultdict(set)
    for d, triples in triples_by_doc.items():
        for h, r, t in triples:
            adj[h].add(t); adj[t].add(h)
            if r == "co-referenced-with":
                adj_struct[h].add(t); adj_struct[t].add(h)
            ent_docs[h].add(d); ent_docs[t].add(d)
    return adj, adj_struct, ent_docs


def kg_search_local(embeddings, index: EntityIndex, cfg: Config, chat, triples_by_doc: dict,
                     doc_freq: dict, query: str, hops: int, k: int, tokenize) -> dict:
    adj, adj_struct, ent_docs = build_adjacency(triples_by_doc)
    seeds = set(match_seeds(embeddings, index, cfg, extract_query_entities(chat, query, tokenize)))
    matched = {e for e in adj if e in seeds}

    struct_reach = defaultdict(bool)
    visited, frontier = set(matched), set(matched)
    for _ in range(hops):
        nxt = set()
        for e in frontier:
            for nb in adj[e]:
                if nb in adj_struct[e] or struct_reach[e]:
                    struct_reach[nb] = True
                if nb not in visited:
                    nxt.add(nb)
        visited |= nxt
        frontier = nxt
    expanded = visited - matched

    scores = defaultdict(float)
    for e in matched:
        w = cfg.kg_seed_weight * _damp(doc_freq, e, cfg.kg_use_idf_damping)
        for d in ent_docs.get(e, ()):
            scores[d] += w
    for e in expanded:
        base = cfg.kg_struct_weight if struct_reach[e] else 1.0
        w = base * _damp(doc_freq, e, cfg.kg_use_idf_damping)
        for d in ent_docs.get(e, ()):
            scores[d] += w
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:k])
