"""Embed normalized job titles with OpenAI text-embedding-3-small and import them into Milvus.

Provider: OpenRouter embeddings API (`openai/text-embedding-3-small`).
The native vector is 1536. The request asks for EMBEDDING_DIM (1024). If the
provider still returns the full vector, the shared pipeline keeps the first
1024 values and L2-normalizes them, which is the Matryoshka shortening this
model was trained for.

The spreadsheet `title` column is first passed through `clean_text_readable`.
That keeps the readable spacing, turns نیم‌فاصله into a normal space, and applies
the shared letter normalization (Arabic yeh/kaf, digits, diacritics, and so on).

This model has no query instruction. The normalized title is embedded as-is.

The checkpoint database is separate from the Qwen and F2LLM runs. If this
process stops, start it again with the same command; finished Milvus batches
are kept.

Env: OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENAI_EMBED_MODEL,
OPENAI_COLLECTION, OPENAI_CHECKPOINT_DB, OPENAI_EMBED_BATCH_SIZE,
OPENAI_MILVUS_BATCH_SIZE, EMBEDDING_DIM, SOURCE_XLSX, EMBED_TIMEOUT_SECONDS,
EMBED_MAX_RETRIES, LOG_LEVEL, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN.
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

LOGGER = logging.getLogger("openai_text_embedding_3_small")


class OpenRouterEmbeddings:
    def __init__(self) -> None:
        self.base_url = env_str("OPENROUTER_BASE_URL").rstrip("/")
        self.model = env_str("OPENAI_EMBED_MODEL")
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
        description=(
            "Normalize titles, embed them with OpenAI text-embedding-3-small "
            "via OpenRouter, and upsert Milvus."
        )
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
        model_name="OpenAI text-embedding-3-small",
        collection_name=env_str("OPENAI_COLLECTION"),
        checkpoint_path=resolve_path(env_str("OPENAI_CHECKPOINT_DB")),
        embed_batch_size=env_int("OPENAI_EMBED_BATCH_SIZE"),
        milvus_batch_size=env_int("OPENAI_MILVUS_BATCH_SIZE"),
        dimension=client.dimension,
        log_name="openai_text_embedding_3_small",
    )
    return run_import(client.embed, config, limit=args.limit, retry_errors=args.retry_errors)


if __name__ == "__main__":
    raise SystemExit(main())
