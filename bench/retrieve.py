"""Search each validation query in that model's own Milvus collection.

The embedded text is the normalized query, with no instruction prefix.
F2LLM reuses validation_f2llm_query_vectors.npz. Qwen and OpenAI embed the
same queries if their cache files are missing, then search:

    F2LLM_COLLECTION
    OPENAI_COLLECTION
    QWEN_COLLECTION
    QWEN8B_COLLECTION

top_title is the original spreadsheet title. top_score is Milvus COSINE
similarity: a higher number is a closer neighbor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Font
from pymilvus import MilvusClient

from bench.env import (
    OPENAI_QUERY_VECTORS,
    QWEN8B_QUERY_VECTORS,
    QWEN_QUERY_VECTORS,
    RETRIEVAL_XLSX,
    env_int,
    env_str,
    load_environment,
    milvus_uri,
    setup_logging,
    share_logs,
)
from bench.providers import openai_client, qwen8b_client, qwen_client
from bench.query_embed import VECTOR_NPZ
from bench.validation_set import read_queries
from bench.vectors import fit_embedding

OUTPUT_XLSX = RETRIEVAL_XLSX
LOG_NAME = "validation_retrieval_top1"
SEARCH_BATCH = 50
HNSW_EF = 128


@dataclass(frozen=True)
class SearchModel:
    key: str
    collection_env: str
    vector_path: Path
    batch_env: str | None


MODELS = (
    SearchModel("f2llm_v2_4b", "F2LLM_COLLECTION", VECTOR_NPZ, None),
    SearchModel(
        "openai_small_3",
        "OPENAI_COLLECTION",
        OPENAI_QUERY_VECTORS,
        "OPENAI_EMBED_BATCH_SIZE",
    ),
    SearchModel(
        "qwen3_4b",
        "QWEN_COLLECTION",
        QWEN_QUERY_VECTORS,
        "QWEN_EMBED_BATCH_SIZE",
    ),
    SearchModel(
        "qwen3_8b",
        "QWEN8B_COLLECTION",
        QWEN8B_QUERY_VECTORS,
        "QWEN8B_EMBED_BATCH_SIZE",
    ),
)

QUERY_CLIENTS = {
    "openai_small_3": openai_client,
    "qwen3_4b": qwen_client,
    "qwen3_8b": qwen8b_client,
}


def _align_saved_vectors(
    path: Path,
    records: list[dict[str, object]],
    dimension: int,
) -> np.ndarray | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    required = ("vectors", "gold_id", "query_normalized")
    if any(name not in data.files for name in required):
        return None
    index = {int(gold_id): row for row, gold_id in enumerate(data["gold_id"])}
    chosen = []
    for record in records:
        row = index.get(int(record["gold_id"]))
        if row is None:
            return None
        if str(data["query_normalized"][row]) != record["query_normalized"]:
            return None
        chosen.append(data["vectors"][row])
    matrix = np.vstack(chosen).astype(np.float32, copy=False)
    if matrix.shape != (len(records), dimension) or not np.isfinite(matrix).all():
        return None
    return matrix


def _save_vectors(path: Path, records: list[dict[str, object]], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=vectors.astype(np.float32, copy=False),
        gold_id=np.asarray([int(record["gold_id"]) for record in records], dtype=np.int64),
        query=np.asarray([str(record["query"]) for record in records]),
        query_normalized=np.asarray([str(record["query_normalized"]) for record in records]),
        gold_title=np.asarray([str(record["gold_title"]) for record in records]),
    )


def _embed_queries(client: object, records: list[dict[str, object]], batch_size: int, logger: logging.Logger) -> np.ndarray:
    texts = [str(record["query_normalized"]) for record in records]
    fitted: list[list[float]] = []
    incoming_dim = 0
    dimension = client.dimension  # type: ignore[attr-defined]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        raw_vectors = client.embed(batch)  # type: ignore[attr-defined]
        if len(raw_vectors) != len(batch):
            raise SystemExit(f"provider returned {len(raw_vectors)} vectors for {len(batch)} queries")
        for raw in raw_vectors:
            vector, incoming_dim = fit_embedding(raw, dimension)
            fitted.append(vector)
        logger.info("embedded %s/%s queries", min(start + batch_size, len(texts)), len(texts))
    logger.info("provider vector length before fit was %s", incoming_dim)
    return np.asarray(fitted, dtype=np.float32)


def query_vectors(
    spec: SearchModel,
    records: list[dict[str, object]],
    dimension: int,
    logger: logging.Logger,
) -> np.ndarray:
    saved = _align_saved_vectors(spec.vector_path, records, dimension)
    if saved is not None:
        logger.info("reusing %s", spec.vector_path.name)
        return saved
    if spec.batch_env is None:
        raise SystemExit(
            f"{spec.vector_path.name} is missing or does not match the validation queries. "
            "Run: python -m bench embed-queries"
        )
    factory = QUERY_CLIENTS.get(spec.key)
    if factory is None:
        raise SystemExit(f"{spec.key} has no query embedding client")
    client = factory()
    if client.dimension != dimension:
        raise SystemExit(f"{spec.key} dimension is {client.dimension}, expected {dimension}")
    logger.info("embedding %s queries with %s", len(records), spec.key)
    vectors = _embed_queries(client, records, env_int(spec.batch_env), logger)
    _save_vectors(spec.vector_path, records, vectors)
    logger.info("saved %s", spec.vector_path.name)
    return vectors


def prepare_collections(logger: logging.Logger) -> tuple[MilvusClient, dict[str, str]]:
    uri = milvus_uri()
    try:
        client = MilvusClient(uri=uri, token=env_str("MILVUS_TOKEN"), timeout=60)
    except Exception as exc:
        raise SystemExit(f"Could not connect to Milvus at {uri}: {exc}") from exc
    logger.info("connected to Milvus")

    names: dict[str, str] = {}
    for spec in MODELS:
        name = env_str(spec.collection_env)
        names[spec.key] = name
        if not client.has_collection(name):
            raise SystemExit(f"Collection {name} does not exist. Import that model before retrieval.")
        stats = client.get_collection_stats(name)
        count = int(stats.get("row_count", 0))
        if count <= 0:
            raise SystemExit(f"Collection {name} is empty. Import that model before retrieval.")
        client.load_collection(name, timeout=120)
        logger.info("collection %s rows=%s", name, count)
    return client, names


def _hit_fields(hit: dict) -> tuple[int, str, float]:
    entity = hit.get("entity") or {}
    if not isinstance(entity, dict):
        entity = {}
    raw_id = entity.get("id", hit.get("id"))
    raw_title = entity.get("title_raw", hit.get("title_raw"))
    if raw_title in (None, ""):
        raw_title = entity.get("title", hit.get("title", ""))
    score = hit.get("distance")
    if raw_id is None or score is None:
        raise SystemExit(f"Milvus hit is missing id or score: {hit!r}")
    return int(raw_id), str(raw_title), float(score)


def search_top1(
    client: MilvusClient,
    collection: str,
    vectors: np.ndarray,
    logger: logging.Logger,
) -> list[tuple[int, str, float]]:
    found: list[tuple[int, str, float]] = []
    total = vectors.shape[0]
    for start in range(0, total, SEARCH_BATCH):
        chunk = vectors[start : start + SEARCH_BATCH]
        result = client.search(
            collection_name=collection,
            data=chunk.tolist(),
            anns_field="embedding",
            limit=1,
            output_fields=["id", "title", "title_raw"],
            search_params={"metric_type": "COSINE", "params": {"ef": HNSW_EF}},
            timeout=120,
        )
        if len(result) != len(chunk):
            raise SystemExit(f"{collection} returned {len(result)} result lists for {len(chunk)} queries")
        for hits in result:
            if not hits:
                raise SystemExit(f"{collection} returned no neighbor")
            found.append(_hit_fields(hits[0]))
        logger.info("searched %s/%s in %s", min(start + SEARCH_BATCH, total), total, collection)
    return found


def write_xlsx(
    path: Path,
    records: list[dict[str, object]],
    hits: dict[str, list[tuple[int, str, float]]],
) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Top1"
    headers = ["query", "gold_title", "gold_id"]
    for spec in MODELS:
        headers.extend([f"{spec.key}_top_id", f"{spec.key}_top_title", f"{spec.key}_top_score"])
    sheet.append(headers)
    for row_index, record in enumerate(records):
        values: list[object] = [record["query"], record["gold_title"], int(record["gold_id"])]
        for spec in MODELS:
            top_id, title, score = hits[spec.key][row_index]
            values.extend([top_id, title, score])
        sheet.append(values)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def retrieve_top1() -> int:
    load_environment()
    logger = setup_logging(LOG_NAME)
    share_logs(logger, "qwen3_embedding_4b", "qwen3_embedding_8b", "openai_text_embedding_3_small")
    dimension = env_int("EMBEDDING_DIM")
    records = read_queries()
    if not records:
        raise SystemExit("validation workbook has no queries")
    blank = [int(record["gold_id"]) for record in records if not str(record["query_normalized"]).strip()]
    if blank:
        raise SystemExit(f"{len(blank)} queries are empty after normalization, first ids: {blank[:10]}")
    logger.info("loaded %s validation queries", len(records))

    client, collections = prepare_collections(logger)
    vectors = {spec.key: query_vectors(spec, records, dimension, logger) for spec in MODELS}
    hits = {
        key: search_top1(client, collections[key], matrix, logger)
        for key, matrix in vectors.items()
    }
    write_xlsx(OUTPUT_XLSX, records, hits)
    for spec in MODELS:
        matched = sum(
            top_id == int(record["gold_id"])
            for record, (top_id, _, _) in zip(records, hits[spec.key], strict=True)
        )
        logger.info("%s top1 matches gold_id: %s/%s", spec.key, matched, len(records))
    logger.info("wrote %s", OUTPUT_XLSX.name)
    return 0
