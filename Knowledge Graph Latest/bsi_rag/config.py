"""Every tunable knob for the pipeline lives here."""
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    seed: int = 42

    # ---- models -------------------------------------------------------
    emb_model: str = "BAAI/bge-small-en-v1.5"
    bge_query_instruction: str = "Represent this sentence for searching relevant passages: "

    # LLM backend for triple extraction + query-entity extraction (kg.py).
    # "hf"     -> local transformers model via langchain_huggingface (default,
    #             no external daemon required).
    # "ollama" -> ChatOllama, requires a running `ollama serve` with llm_model pulled.
    llm_backend: str = "hf"
    # Gated HF repo: run `huggingface-cli login` (or set HF_TOKEN) with an
    # account that has accepted Meta's license at the model page before
    # loading it. Loaded via plain from_pretrained rather than a pre-quantized
    # runtime, an 8B model can peak around 16GB RAM while materializing.
    # Drop to "Qwen/Qwen2.5-3B-Instruct" if that's not workable on your machine.
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"   # HF repo id (hf backend)
                                                               # or Ollama tag e.g. "llama3:8b" (ollama backend)
    hf_dtype: str = "auto"                # torch_dtype passed to from_pretrained
    hf_device_map: str = "auto"            # requires `accelerate`; set to a plain
                                            # device string (e.g. "mps") if you hit
                                            # accelerate/device-map issues
    hf_max_new_tokens: int = 256
    ollama_base_url: str = field(default_factory=lambda: os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"))

    # ---- data -----------------------------------------------------------
    qa_file: str = "english_data.json"
    corpus_file: str = "bsi_english_paragraphs.json"

    # ---- retrieval ------------------------------------------------------
    n_queries: int | None = 20            # None = all questions
    top_k: int = 10
    kg_max_chars: int = 900
    kg_hops: int = 1
    rrf_k: int = 60

    # ---- KG quality knobs -------------------------------------------------
    # Entity resolution (kg.py): complete-linkage clustering -- a merge is
    # only allowed if every cross-cluster pair clears the threshold, not just
    # one chained pair -- to stop unrelated entities (e.g. "local attacker"
    # and "remote attacker") from collapsing into one hub node.
    entity_sim_threshold: float = 0.90    # kept high since complete-linkage clustering
                                           # is stricter than single-linkage chaining
    entity_max_cluster_size: int = 4      # hard cap: refuse a merge that would create a
                                           # cluster bigger than this -- a cheap, general
                                           # guard against hub nodes

    # Seeding: normalized-exact match first, then embedding cosine similarity
    # against canonical entity names.
    seed_sim_threshold: float = 0.80
    seed_top_n_per_phrase: int = 3        # cap matches per seed phrase to avoid flooding

    # Scoring: IDF-style damping of entities that appear in many documents,
    # so a residual hub entity can't dominate purely by frequency.
    kg_seed_weight: float = 2.0
    kg_struct_weight: float = 1.0
    kg_use_idf_damping: bool = True

    # Hybrid fusion (evaluation.py): weighted RRF. KG is the weakest, noisiest
    # channel, so it is down-weighted rather than contributing equally to dense/BM25.
    rrf_weight_dense: float = 1.0
    rrf_weight_bm25: float = 1.0
    rrf_weight_kg: float = 0.5
    rrf_min_kg_seeds: int = 1             # if the KG retriever matched fewer seed
                                           # entities than this for a query, drop its
                                           # contribution from that query's fusion
                                           # entirely rather than fusing in pure noise

    # ---- Neo4j ------------------------------------------------------------
    neo4j_uri: str = field(default_factory=lambda: os.environ.get(
        "NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.environ.get(
        "NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.environ.get(
        "NEO4J_PASSWORD", "bsi-rag-proto"))

    # ---- device -----------------------------------------------------------
    device: str = "cpu"                   # resolved at runtime, see pipeline.resolve_device()
