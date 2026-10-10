"""Embed the validation query column with F2LLM-v2-4B and save 1024-d vectors.

Reads validation_human_job_queries.xlsx. Each query is normalized with
clean_text_readable, the same cleaner used for the indexed titles, then sent
to the Hugging Face endpoint. The returned vector is cut to EMBEDDING_DIM and
L2-normalized. Nothing is written to Milvus.

The output is one NumPy file, validation_f2llm_query_vectors.npz:

    vectors            float32 [n, 1024]
    gold_id            int64   [n]
    query              unicode [n]   original typed query
    query_normalized   unicode [n]   text that was embedded
    gold_title         unicode [n]

Finished rows are stored in a SQLite checkpoint first, so stopping the script
does not lose vectors already returned by the endpoint.

Run:
    python embed_validation_queries_f2llm.py
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import openpyxl

from embed_f2llm_v2_4b import HuggingFaceEmbeddings, require_endpoint
from embedding_pipeline import env_int, fit_embedding, load_environment, setup_logging
from text_cleaner import clean_text_readable

ROOT = Path(__file__).resolve().parent
QUERY_XLSX = ROOT / "validation_human_job_queries.xlsx"
VECTOR_NPZ = ROOT / "validation_f2llm_query_vectors.npz"
CHECKPOINT_DB = ROOT / "checkpoints" / "f2llm_validation_queries.sqlite"
LOG_NAME = "f2llm_validation_queries"


def read_queries(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"Query file not found: {path.name}. Run generate_validation_queries.py first.")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
        columns = {name: index for index, name in enumerate(header)}
        for required in ("query", "gold_id", "gold_title"):
            if required not in columns:
                raise SystemExit(f"{path.name} is missing a {required} column")
        records = []
        for values in rows:
            if values is None or values[columns["query"]] is None:
                continue
            query = str(values[columns["query"]]).strip()
            if not query:
                continue
            raw_id = values[columns["gold_id"]]
            gold_id = int(raw_id) if not isinstance(raw_id, str) else int(str(raw_id).strip())
            gold_title = "" if values[columns["gold_title"]] is None else str(values[columns["gold_title"]]).strip()
            records.append(
                {
                    "gold_id": gold_id,
                    "query": query,
                    "query_normalized": clean_text_readable(query),
                    "gold_title": gold_title,
                }
            )
        return records
    finally:
        workbook.close()


class QueryCheckpoint:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                gold_id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                query_normalized TEXT NOT NULL,
                gold_title TEXT NOT NULL,
                embedding BLOB
            )
            """
        )

    def close(self) -> None:
        self.conn.close()

    def sync(self, records: list[dict[str, object]]) -> int:
        """Insert new rows and drop a stored vector when its normalized query changed."""
        reset = 0
        with self.conn:
            for record in records:
                current = self.conn.execute(
                    "SELECT query_normalized FROM vectors WHERE gold_id = ?",
                    (record["gold_id"],),
                ).fetchone()
                if current is None:
                    self.conn.execute(
                        """
                        INSERT INTO vectors (gold_id, query, query_normalized, gold_title)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            record["gold_id"],
                            record["query"],
                            record["query_normalized"],
                            record["gold_title"],
                        ),
                    )
                    continue
                if current[0] == record["query_normalized"]:
                    self.conn.execute(
                        "UPDATE vectors SET query = ?, gold_title = ? WHERE gold_id = ?",
                        (record["query"], record["gold_title"], record["gold_id"]),
                    )
                    continue
                reset += 1
                self.conn.execute(
                    """
                    UPDATE vectors
                    SET query = ?, query_normalized = ?, gold_title = ?, embedding = NULL
                    WHERE gold_id = ?
                    """,
                    (
                        record["query"],
                        record["query_normalized"],
                        record["gold_title"],
                        record["gold_id"],
                    ),
                )
        return reset

    def pending(self, limit: int) -> list[tuple[int, str]]:
        cursor = self.conn.execute(
            """
            SELECT gold_id, query_normalized
            FROM vectors
            WHERE embedding IS NULL AND query_normalized != ''
            ORDER BY gold_id
            LIMIT ?
            """,
            (limit,),
        )
        return [(int(row_id), text) for row_id, text in cursor]

    def save(self, rows: list[tuple[int, list[float]]]) -> None:
        payload = [
            (np.asarray(vector, dtype="<f4").tobytes(), row_id)
            for row_id, vector in rows
        ]
        with self.conn:
            self.conn.executemany(
                "UPDATE vectors SET embedding = ? WHERE gold_id = ?",
                payload,
            )

    def export_rows(self, dimension: int) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        return list(
            self.conn.execute(
                """
                SELECT gold_id, query, query_normalized, gold_title, embedding
                FROM vectors
                ORDER BY gold_id
                """
            )
        )


def write_npz(rows: list[sqlite3.Row], dimension: int, path: Path) -> None:
    missing = [int(row["gold_id"]) for row in rows if row["embedding"] is None]
    if missing:
        raise SystemExit(f"{len(missing)} queries still have no vector, first ids: {missing[:10]}")
    vectors = np.vstack(
        [
            np.frombuffer(row["embedding"], dtype="<f4").astype(np.float32, copy=True)
            for row in rows
        ]
    )
    if vectors.shape != (len(rows), dimension):
        raise SystemExit(f"vector matrix shape is {vectors.shape}, expected {(len(rows), dimension)}")
    np.savez(
        path,
        vectors=vectors,
        gold_id=np.asarray([int(row["gold_id"]) for row in rows], dtype=np.int64),
        query=np.asarray([row["query"] for row in rows]),
        query_normalized=np.asarray([row["query_normalized"] for row in rows]),
        gold_title=np.asarray([row["gold_title"] for row in rows]),
    )


def main() -> int:
    load_environment()
    logger = setup_logging(LOG_NAME)
    endpoint_logger = logging.getLogger("f2llm_v2_4b")
    endpoint_logger.setLevel(logger.level)
    endpoint_logger.propagate = False
    for handler in logger.handlers:
        endpoint_logger.addHandler(handler)
    token, endpoint = require_endpoint()
    client = HuggingFaceEmbeddings(token, endpoint)
    dimension = client.dimension
    batch_size = env_int("F2LLM_EMBED_BATCH_SIZE")

    records = read_queries(QUERY_XLSX)
    logger.info("loaded %s queries from %s", len(records), QUERY_XLSX.name)
    checkpoint = QueryCheckpoint(CHECKPOINT_DB)
    try:
        reset = checkpoint.sync(records)
        if reset:
            logger.info("cleared %s stored vectors because the normalized query changed", reset)
        while True:
            pending = checkpoint.pending(batch_size)
            if not pending:
                break
            texts = [text for _, text in pending]
            raw_vectors = client.embed(texts)
            if len(raw_vectors) != len(pending):
                raise SystemExit(
                    f"endpoint returned {len(raw_vectors)} vectors for {len(pending)} queries"
                )
            fitted = []
            incoming = 0
            for (gold_id, _), raw in zip(pending, raw_vectors, strict=True):
                vector, incoming = fit_embedding(raw, dimension)
                fitted.append((gold_id, vector))
            checkpoint.save(fitted)
            logger.info(
                "saved %s vectors (ids %s..%s, provider dim %s -> %s)",
                len(fitted),
                fitted[0][0],
                fitted[-1][0],
                incoming,
                dimension,
            )
        rows = checkpoint.export_rows(dimension)
        write_npz(rows, dimension, VECTOR_NPZ)
    finally:
        checkpoint.close()

    logger.info("wrote %s vectors to %s", len(records), VECTOR_NPZ.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
