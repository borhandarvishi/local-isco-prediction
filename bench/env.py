"""Paths and .env settings shared by the benchmark scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source"
VALIDATION_DIR = DATA_DIR / "validation"
VECTOR_DIR = VALIDATION_DIR / "vectors"
RESULT_DIR = VALIDATION_DIR / "results"

SOURCE_XLSX = SOURCE_DIR / "test_data_for_local_embedding_models.xlsx"
QUERY_XLSX = VALIDATION_DIR / "validation_human_job_queries.xlsx"
F2LLM_QUERY_VECTORS = VECTOR_DIR / "validation_f2llm_query_vectors.npz"
OPENAI_QUERY_VECTORS = VECTOR_DIR / "validation_openai_query_vectors.npz"
QWEN_QUERY_VECTORS = VECTOR_DIR / "validation_qwen3_query_vectors.npz"
QWEN8B_QUERY_VECTORS = VECTOR_DIR / "validation_qwen3_8b_query_vectors.npz"
RETRIEVAL_XLSX = RESULT_DIR / "validation_retrieval_top1.xlsx"


def load_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def env_str(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name} in .env")
    return value


def env_int(name: str) -> int:
    raw = env_str(name)
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be greater than 0, got {value}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def setup_logging(log_name: str) -> logging.Logger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise SystemExit(f"LOG_LEVEL must be a logging level, got {level_name!r}")

    logger = logging.getLogger(log_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{log_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def share_logs(logger: logging.Logger, *names: str) -> None:
    """Send provider warnings to the same stream and file as the calling script."""
    for name in names:
        child = logging.getLogger(name)
        child.setLevel(logger.level)
        child.propagate = False
        child.handlers.clear()
        for handler in logger.handlers:
            child.addHandler(handler)


def milvus_uri() -> str:
    host = env_str("MILVUS_HOST")
    if host.startswith("http://") or host.startswith("https://"):
        return host
    return f"http://{host}:{env_str('MILVUS_PORT')}"
