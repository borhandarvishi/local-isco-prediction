"""Embed normalized job titles with Qwen3-Embedding-4B and import them into Milvus.

Provider: OpenRouter embeddings API (`qwen/qwen3-embedding-4b`).
The spreadsheet `title` column is first passed through `clean_text_readable`.
That keeps the readable spacing, turns نیم‌فاصله into a normal space, and applies
the shared letter normalization (Arabic yeh/kaf, digits, diacritics, and so on).

Stored passages are embedded without a query instruction. Add the instruction
later only on the query side:

    Instruct: Given a job title, retrieve the most similar job titles
    Query: <title>

The checkpoint database is separate from the F2LLM run. If this process stops,
start it again with the same command; finished Milvus batches are kept.

Env: OPENROUTER_API_KEY, OPENROUTER_BASE_URL, QWEN_EMBED_MODEL, QWEN_COLLECTION,
QWEN_CHECKPOINT_DB, QWEN_EMBED_BATCH_SIZE, QWEN_MILVUS_BATCH_SIZE, EMBEDDING_DIM,
SOURCE_XLSX, EMBED_TIMEOUT_SECONDS, EMBED_MAX_RETRIES, LOG_LEVEL, MILVUS_HOST,
MILVUS_PORT, MILVUS_TOKEN.
"""

from __future__ import annotations

import argparse
import logging

import requests

from embedding_pipeline import (
    ImportConfig,
    env_int,
    env_str,
    load_environment,
    request_json,
    resolve_path,
    run_import,
    vectors_from_openai_payload,
)

LOGGER = logging.getLogger("qwen3_embedding_4b")


class OpenRouterEmbeddings:
    def __init__(self) -> None:
        self.base_url = env_str("OPENROUTER_BASE_URL").rstrip("/")
        self.model = env_str("QWEN_EMBED_MODEL")
        self.dimension = env_int("EMBEDDING_DIM")
        self.timeout = env_int("EMBED_TIMEOUT_SECONDS")
        self.max_retries = env_int("EMBED_MAX_RETRIES")
        self.session = requests.Session()
        self._logged_provider = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = request_json(
            self.session,
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {env_str('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            body={
                "model": self.model,
                "input": texts,
                "dimensions": self.dimension,
            },
            timeout=self.timeout,
            max_retries=self.max_retries,
            logger=LOGGER,
        )
        if not self._logged_provider and isinstance(payload, dict):
            LOGGER.info(
                "openrouter model=%s provider=%s",
                payload.get("model", self.model),
                payload.get("provider", "unknown"),
            )
            self._logged_provider = True
        return vectors_from_openai_payload(payload, len(texts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize titles, embed them with Qwen3-Embedding-4B via OpenRouter, and upsert Milvus."
    )
    parser.add_argument("--limit", type=int, default=0, help="Only the first N spreadsheet rows. 0 means all.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Clear saved per-row embed errors and try those rows again.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    return args


def main() -> int:
    load_environment()
    args = parse_args()
    client = OpenRouterEmbeddings()
    config = ImportConfig(
        model_name="Qwen3-Embedding-4B",
        collection_name=env_str("QWEN_COLLECTION"),
        checkpoint_path=resolve_path(env_str("QWEN_CHECKPOINT_DB")),
        embed_batch_size=env_int("QWEN_EMBED_BATCH_SIZE"),
        milvus_batch_size=env_int("QWEN_MILVUS_BATCH_SIZE"),
        dimension=client.dimension,
        log_name="qwen3_embedding_4b",
    )
    return run_import(client.embed, config, limit=args.limit, retry_errors=args.retry_errors)


if __name__ == "__main__":
    raise SystemExit(main())
