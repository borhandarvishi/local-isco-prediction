"""Load titles, normalize them, and import embeddings into Milvus with a resume checkpoint.

Each model keeps its own checkpoint database and its own collection.
A batch is committed to SQLite only after that step has succeeded:

1. The provider returns vectors, they are truncated to the target dimension,
   L2-normalized, and stored in SQLite.
2. That same batch is upserted into Milvus and flushed.
3. The rows are marked inserted.

Stopping the process keeps every finished batch. Run the same command again
and it continues from the first unfinished row.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openpyxl
from pymilvus import DataType, MilvusClient

from bench.env import env_str, milvus_uri, resolve_path, setup_logging
from bench.providers import EmbedError
from bench.vectors import fit_embedding
from bench.text_cleaner import clean_text_readable

VARCHAR_MAX_LENGTH = 65535
EMPTY_TEXT_ERROR = "empty_after_normalize"

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class SourceRow:
    row_id: int
    title_raw: str
    title_norm: str


@dataclass(frozen=True)
class ImportConfig:
    model_name: str
    collection_name: str
    checkpoint_path: Path
    embed_batch_size: int
    milvus_batch_size: int
    dimension: int
    log_name: str


@dataclass(frozen=True)
class RunStats:
    total: int
    empty: int
    embedded: int
    inserted: int
    failed: int


def _parse_row_id(value: Any, excel_row: int) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"row {excel_row} has an empty id")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"row {excel_row} id is not an integer: {value!r}")
        return int(value)
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"row {excel_row} id is not an integer: {value!r}")
    return int(text)


def read_source_rows(path: Path, limit: int) -> list[SourceRow]:
    if not path.is_file():
        raise SystemExit(f"Spreadsheet not found: {path}")

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook["JobTitles"] if "JobTitles" in workbook.sheetnames else workbook[workbook.sheetnames[0]]
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if not header:
            raise SystemExit(f"{path.name} has no header row")
        columns = [str(cell).strip() if cell is not None else "" for cell in header]
        try:
            id_index = columns.index("id")
            title_index = columns.index("title")
        except ValueError as exc:
            raise SystemExit(
                f"{path.name} sheet {sheet.title!r} must have id and title columns, found {columns}"
            ) from exc

        rows: list[SourceRow] = []
        seen: dict[int, int] = {}
        for offset, values in enumerate(iterator, start=2):
            if values is None:
                continue
            if all(cell is None or str(cell).strip() == "" for cell in values):
                continue
            row_id = _parse_row_id(values[id_index] if id_index < len(values) else None, offset)
            if row_id in seen:
                raise SystemExit(f"duplicate id {row_id} on spreadsheet rows {seen[row_id]} and {offset}")
            seen[row_id] = offset
            raw_value = values[title_index] if title_index < len(values) else None
            title_raw = "" if raw_value is None else str(raw_value).strip()
            rows.append(
                SourceRow(
                    row_id=row_id,
                    title_raw=title_raw,
                    title_norm=clean_text_readable(title_raw),
                )
            )
            if limit and len(rows) >= limit:
                break
    finally:
        workbook.close()

    if not rows:
        raise SystemExit(f"No data rows in {path}")
    return rows


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pack_vector(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _unpack_vector(blob: bytes, dimension: int, row_id: int) -> list[float]:
    array = np.frombuffer(blob, dtype="<f4")
    if array.size != dimension:
        raise RuntimeError(
            f"checkpoint vector for id {row_id} has {array.size} dimensions, expected {dimension}"
        )
    return array.astype(np.float64).tolist()


class Checkpoint:
    """SQLite progress for one model. Embeddings are stored before Milvus is touched."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                title_raw TEXT NOT NULL,
                title_norm TEXT NOT NULL,
                embedding BLOB,
                embedded_at TEXT,
                inserted_at TEXT,
                last_error TEXT
            )
            """
        )
        self.conn.execute("CREATE TEMP TABLE scope (id INTEGER PRIMARY KEY)")

    def close(self) -> None:
        self.conn.close()

    def set_scope(self, rows: list[SourceRow]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM scope")
            self.conn.executemany("INSERT INTO scope (id) VALUES (?)", [(row.row_id,) for row in rows])

    def sync(self, rows: list[SourceRow]) -> int:
        """Insert new rows and invalidate vectors whose normalized title changed.

        Returns how many existing rows were reset because their text changed.
        """
        reset = 0
        with self.conn:
            for row in rows:
                error = None
                if not row.title_norm:
                    error = EMPTY_TEXT_ERROR
                elif len(row.title_raw) > VARCHAR_MAX_LENGTH or len(row.title_norm) > VARCHAR_MAX_LENGTH:
                    error = f"title is longer than Milvus VARCHAR limit ({VARCHAR_MAX_LENGTH})"

                current = self.conn.execute(
                    "SELECT title_raw, title_norm FROM items WHERE id = ?",
                    (row.row_id,),
                ).fetchone()
                if current is None:
                    self.conn.execute(
                        """
                        INSERT INTO items (id, title_raw, title_norm, last_error)
                        VALUES (?, ?, ?, ?)
                        """,
                        (row.row_id, row.title_raw, row.title_norm, error),
                    )
                    continue
                if current[0] == row.title_raw and current[1] == row.title_norm:
                    continue
                reset += 1
                self.conn.execute(
                    """
                    UPDATE items
                    SET title_raw = ?, title_norm = ?, embedding = NULL,
                        embedded_at = NULL, inserted_at = NULL, last_error = ?
                    WHERE id = ?
                    """,
                    (row.title_raw, row.title_norm, error, row.row_id),
                )
        return reset

    def clear_retryable_errors(self) -> int:
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE items
                SET last_error = NULL
                WHERE id IN (SELECT id FROM scope)
                  AND last_error IS NOT NULL
                  AND last_error != ?
                """,
                (EMPTY_TEXT_ERROR,),
            )
        return cursor.rowcount

    def pending_embed(self, limit: int) -> list[tuple[int, str]]:
        cursor = self.conn.execute(
            """
            SELECT items.id, items.title_norm
            FROM items
            JOIN scope ON scope.id = items.id
            WHERE items.embedding IS NULL
              AND items.title_norm != ''
              AND items.last_error IS NULL
            ORDER BY items.id
            LIMIT ?
            """,
            (limit,),
        )
        return [(int(row_id), text) for row_id, text in cursor]

    def save_embeddings(self, rows: list[tuple[int, list[float]]]) -> None:
        stamped = _utcnow()
        with self.conn:
            self.conn.executemany(
                """
                UPDATE items
                SET embedding = ?, embedded_at = ?, last_error = NULL
                WHERE id = ?
                """,
                [(_pack_vector(vector), stamped, row_id) for row_id, vector in rows],
            )

    def mark_error(self, row_id: int, message: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE items SET last_error = ? WHERE id = ?",
                (message[:500], row_id),
            )

    def pending_insert(self, limit: int, dimension: int) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            """
            SELECT items.id, items.title_raw, items.title_norm, items.embedding
            FROM items
            JOIN scope ON scope.id = items.id
            WHERE items.embedding IS NOT NULL
              AND items.inserted_at IS NULL
            ORDER BY items.id
            LIMIT ?
            """,
            (limit,),
        )
        records = []
        for row_id, title_raw, title_norm, blob in cursor:
            records.append(
                {
                    "id": int(row_id),
                    "title_raw": title_raw,
                    "title": title_norm,
                    "embedding": _unpack_vector(blob, dimension, int(row_id)),
                }
            )
        return records

    def mark_inserted(self, row_ids: list[int]) -> None:
        stamped = _utcnow()
        with self.conn:
            self.conn.executemany(
                "UPDATE items SET inserted_at = ?, last_error = NULL WHERE id = ?",
                [(stamped, row_id) for row_id in row_ids],
            )

    def stats(self) -> RunStats:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN items.title_norm = '' THEN 1 ELSE 0 END),
                SUM(CASE WHEN items.embedding IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN items.inserted_at IS NOT NULL THEN 1 ELSE 0 END),
                SUM(
                    CASE
                        WHEN items.last_error IS NOT NULL AND items.last_error != ? THEN 1
                        ELSE 0
                    END
                )
            FROM items
            JOIN scope ON scope.id = items.id
            """,
            (EMPTY_TEXT_ERROR,),
        ).fetchone()
        return RunStats(
            total=int(row[0] or 0),
            empty=int(row[1] or 0),
            embedded=int(row[2] or 0),
            inserted=int(row[3] or 0),
            failed=int(row[4] or 0),
        )

    def error_samples(self, limit: int = 20) -> list[tuple[int, str]]:
        cursor = self.conn.execute(
            """
            SELECT items.id, items.last_error
            FROM items
            JOIN scope ON scope.id = items.id
            WHERE items.last_error IS NOT NULL
              AND items.last_error != ?
            ORDER BY items.id
            LIMIT ?
            """,
            (EMPTY_TEXT_ERROR, limit),
        )
        return [(int(row_id), message) for row_id, message in cursor]


class MilvusSink:
    def __init__(self, collection_name: str, dimension: int, logger: logging.Logger) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self.logger = logger
        uri = milvus_uri()
        try:
            self.client = MilvusClient(uri=uri, token=env_str("MILVUS_TOKEN"), timeout=60)
        except Exception as exc:
            raise SystemExit(f"Could not connect to Milvus at {uri}: {exc}") from exc
        self.logger.info("connected to Milvus at %s", uri)

    def ensure_collection(self, description: str) -> None:
        if self.client.has_collection(self.collection_name):
            self._check_existing_schema()
            self.logger.info("using existing collection %s", self.collection_name)
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=description,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("title", DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field("title_raw", DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(
            "created collection %s dim=%s metric=COSINE index=HNSW",
            self.collection_name,
            self.dimension,
        )

    def _check_existing_schema(self) -> None:
        description = self.client.describe_collection(self.collection_name)
        if description.get("auto_id"):
            raise SystemExit(
                f"Collection {self.collection_name} uses auto_id. "
                "This importer upserts the spreadsheet id as the primary key. "
                "Use a new collection name in .env, or drop this collection yourself."
            )
        fields = {field["name"]: field for field in description.get("fields", [])}
        required = ("id", "title", "title_raw", "embedding")
        missing = [name for name in required if name not in fields]
        if missing:
            raise SystemExit(
                f"Collection {self.collection_name} is missing fields {missing}. "
                "Use a new collection name in .env."
            )
        params = fields["embedding"].get("params") or {}
        actual_dim = int(params.get("dim", 0))
        if actual_dim != self.dimension:
            raise SystemExit(
                f"Collection {self.collection_name} embedding dim is {actual_dim}, "
                f"expected {self.dimension}. Use a new collection name in .env."
            )

    def upsert(self, records: list[dict[str, Any]]) -> None:
        payload = [
            {
                "id": record["id"],
                "title": record["title"],
                "title_raw": record["title_raw"],
                "embedding": record["embedding"],
            }
            for record in records
        ]
        self.client.upsert(collection_name=self.collection_name, data=payload, timeout=120)
        self.client.flush(self.collection_name, timeout=180)

    def row_count(self) -> int | None:
        try:
            stats = self.client.get_collection_stats(self.collection_name)
        except Exception as exc:
            self.logger.warning("could not read collection stats: %s", exc)
            return None
        raw = stats.get("row_count")
        return int(raw) if raw is not None else None


def _log_stats(logger: logging.Logger, checkpoint: Checkpoint, sink: MilvusSink | None) -> RunStats:
    stats = checkpoint.stats()
    logger.info(
        "checkpoint total=%s empty=%s embedded=%s inserted=%s failed=%s db=%s",
        stats.total,
        stats.empty,
        stats.embedded,
        stats.inserted,
        stats.failed,
        checkpoint.path,
    )
    for row_id, message in checkpoint.error_samples():
        logger.error("row id=%s failed: %s", row_id, message)
    if sink is not None:
        count = sink.row_count()
        if count is not None:
            logger.info("milvus collection %s row_count=%s", sink.collection_name, count)
    return stats


def _fit_batch(
    pairs: list[tuple[int, str]],
    raw_vectors: list[list[float]],
    dimension: int,
) -> tuple[list[tuple[int, list[float]]], int]:
    if len(raw_vectors) != len(pairs):
        raise EmbedError(
            f"provider returned {len(raw_vectors)} vectors for {len(pairs)} texts",
            status=None,
            fatal=False,
            split=True,
        )
    fitted: list[tuple[int, list[float]]] = []
    incoming_dim = 0
    for (row_id, _), raw in zip(pairs, raw_vectors, strict=True):
        if not isinstance(raw, list) or not raw or isinstance(raw[0], list):
            raise EmbedError(
                f"id {row_id}: provider did not return a 1-D embedding",
                status=None,
                fatal=False,
                split=True,
            )
        try:
            vector, incoming_dim = fit_embedding(raw, dimension)
        except ValueError as exc:
            raise EmbedError(f"id {row_id}: {exc}", status=None, fatal=False, split=True) from exc
        fitted.append((row_id, vector))
    return fitted, incoming_dim


def _embed_one_batch(
    embed_texts: EmbedFn,
    pairs: list[tuple[int, str]],
    config: ImportConfig,
    checkpoint: Checkpoint,
    logger: logging.Logger,
    announced_dim: set[int],
) -> None:
    started = time.perf_counter()
    try:
        raw_vectors = embed_texts([text for _, text in pairs])
        fitted, incoming_dim = _fit_batch(pairs, raw_vectors, config.dimension)
    except EmbedError as exc:
        if exc.fatal or not exc.split or len(pairs) == 1:
            if len(pairs) == 1 and exc.split and not exc.fatal:
                row_id = pairs[0][0]
                checkpoint.mark_error(row_id, str(exc))
                logger.error("skipped id=%s: %s", row_id, exc)
                return
            raise
        midpoint = len(pairs) // 2
        logger.warning("batch of %s was rejected (%s); splitting it", len(pairs), exc)
        _embed_one_batch(embed_texts, pairs[:midpoint], config, checkpoint, logger, announced_dim)
        _embed_one_batch(embed_texts, pairs[midpoint:], config, checkpoint, logger, announced_dim)
        return

    if incoming_dim not in announced_dim:
        announced_dim.add(incoming_dim)
        if incoming_dim == config.dimension:
            logger.info("provider returned %s dimensions; L2-normalizing before insert", incoming_dim)
        else:
            logger.info(
                "provider returned %s dimensions; keeping the first %s and L2-normalizing",
                incoming_dim,
                config.dimension,
            )
    checkpoint.save_embeddings(fitted)
    stats = checkpoint.stats()
    logger.info(
        "embedded %s rows in %.1fs (ids %s..%s, checkpoint embedded %s/%s)",
        len(fitted),
        time.perf_counter() - started,
        fitted[0][0],
        fitted[-1][0],
        stats.embedded,
        stats.total - stats.empty,
    )


def _embed_pending(
    embed_texts: EmbedFn,
    config: ImportConfig,
    checkpoint: Checkpoint,
    logger: logging.Logger,
) -> None:
    announced: set[int] = set()
    while True:
        pairs = checkpoint.pending_embed(config.embed_batch_size)
        if not pairs:
            logger.info("no titles left to embed")
            return
        _embed_one_batch(embed_texts, pairs, config, checkpoint, logger, announced)


def _insert_pending(
    config: ImportConfig,
    checkpoint: Checkpoint,
    sink: MilvusSink,
    logger: logging.Logger,
) -> None:
    while True:
        records = checkpoint.pending_insert(config.milvus_batch_size, config.dimension)
        if not records:
            logger.info("no embeddings left to insert")
            return
        started = time.perf_counter()
        sink.upsert(records)
        row_ids = [record["id"] for record in records]
        checkpoint.mark_inserted(row_ids)
        stats = checkpoint.stats()
        logger.info(
            "inserted %s rows into %s in %.1fs (ids %s..%s, checkpoint inserted %s/%s)",
            len(records),
            config.collection_name,
            time.perf_counter() - started,
            row_ids[0],
            row_ids[-1],
            stats.inserted,
            stats.total - stats.empty,
        )


def run_import(embed_texts: EmbedFn, config: ImportConfig, *, limit: int, retry_errors: bool) -> int:
    logger = setup_logging(config.log_name)
    logger.info("starting %s collection=%s dim=%s", config.model_name, config.collection_name, config.dimension)
    logger.info(
        "documents are embedded without a query instruction; normalization is clean_text_readable"
    )

    source_path = resolve_path(env_str("SOURCE_XLSX"))
    rows = read_source_rows(source_path, limit)
    logger.info(
        "loaded %s rows from %s%s",
        len(rows),
        source_path.name,
        f" (limit {limit})" if limit else "",
    )
    sample = rows[0]
    logger.info("example id=%s raw=%r normalized=%r", sample.row_id, sample.title_raw, sample.title_norm)

    checkpoint = Checkpoint(config.checkpoint_path)
    sink: MilvusSink | None = None
    exit_code = 0
    try:
        checkpoint.set_scope(rows)
        reset = checkpoint.sync(rows)
        if reset:
            logger.info("reset %s rows because the normalized title changed", reset)
        if retry_errors:
            cleared = checkpoint.clear_retryable_errors()
            logger.info("cleared %s previous row errors", cleared)

        sink = MilvusSink(config.collection_name, config.dimension, logger)
        sink.ensure_collection(
            f"{config.model_name} document embeddings, dim {config.dimension}, "
            "titles passed through clean_text_readable"
        )
        _embed_pending(embed_texts, config, checkpoint, logger)
        _insert_pending(config, checkpoint, sink, logger)
        stats = _log_stats(logger, checkpoint, sink)
        if stats.failed or stats.inserted != stats.total - stats.empty:
            logger.error("import is incomplete")
            exit_code = 1
        else:
            logger.info("import complete")
    except KeyboardInterrupt:
        logger.warning("stopped. rerun the same command; finished Milvus batches stay committed")
        _log_stats(logger, checkpoint, sink)
        exit_code = 130
    except EmbedError as exc:
        logger.error("stopped: %s", exc)
        _log_stats(logger, checkpoint, sink)
        exit_code = 1
    except Exception:
        logger.exception("stopped because of an unexpected error")
        _log_stats(logger, checkpoint, sink)
        exit_code = 1
    finally:
        checkpoint.close()
    return exit_code
