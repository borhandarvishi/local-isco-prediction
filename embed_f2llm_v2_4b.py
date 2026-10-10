"""Embed normalized job titles with F2LLM-v2-4B and import them into Milvus.

Provider: a Hugging Face Inference Endpoint for `codefuse-ai/F2LLM-v2-4B`.
OpenRouter does not host this model, and Hugging Face serverless inference has
no provider for it either. A Pro account can deploy it as a dedicated endpoint
with the Text Embeddings Inference engine. Point HF_F2LLM_ENDPOINT_URL at that
endpoint. F2LLM_API_STYLE=tei calls `/embed`; openai calls `/v1/embeddings`.

The spreadsheet `title` column is first passed through `clean_text_readable`.
The full model vector is 2560. This script keeps the first EMBEDDING_DIM values
and L2-normalizes after that truncation, which is the Matryoshka procedure from
the model card.

Stored passages are embedded without a query instruction. Add the instruction
later only on the query side:

    Instruct: Given a job title, retrieve the most similar job titles
    Query: <title>

The checkpoint database is separate from the Qwen run.

Env: HF_TOKEN, HF_F2LLM_MODEL, HF_F2LLM_ENDPOINT_URL, F2LLM_API_STYLE,
F2LLM_COLLECTION, F2LLM_CHECKPOINT_DB, F2LLM_EMBED_BATCH_SIZE,
F2LLM_MILVUS_BATCH_SIZE, EMBEDDING_DIM, SOURCE_XLSX, EMBED_TIMEOUT_SECONDS,
EMBED_MAX_RETRIES, LOG_LEVEL, MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import requests

from embedding_pipeline import (
    EmbedError,
    ImportConfig,
    env_int,
    env_str,
    load_environment,
    request_json,
    resolve_path,
    run_import,
    vectors_from_openai_payload,
)

LOGGER = logging.getLogger("f2llm_v2_4b")

ENDPOINT_HELP = """
F2LLM-v2-4B is not on OpenRouter, and Hugging Face does not serve it on serverless inference.
Deploy a Hugging Face Inference Endpoint for codefuse-ai/F2LLM-v2-4B with the Text Embeddings Inference engine, then set:
  HF_TOKEN
  HF_F2LLM_ENDPOINT_URL   (the endpoint origin is enough)
  F2LLM_API_STYLE=tei
If the endpoint speaks the OpenAI embeddings API instead, set F2LLM_API_STYLE=openai.
""".strip()


class HuggingFaceEmbeddings:
    def __init__(self, token: str, endpoint_url: str) -> None:
        self.token = token
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = env_str("HF_F2LLM_MODEL")
        self.api_style = env_str("F2LLM_API_STYLE").lower()
        self.dimension = env_int("EMBEDDING_DIM")
        self.timeout = env_int("EMBED_TIMEOUT_SECONDS")
        self.max_retries = env_int("EMBED_MAX_RETRIES")
        self.session = requests.Session()
        if self.api_style not in {"tei", "openai"}:
            raise SystemExit("F2LLM_API_STYLE must be tei or openai")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.api_style == "tei":
            return self._embed_tei(texts)
        return self._embed_openai(texts)

    def _post(self, url: str, body: dict[str, Any]) -> Any:
        return request_json(
            self.session,
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            body=body,
            timeout=self.timeout,
            max_retries=self.max_retries,
            logger=LOGGER,
        )

    def _embed_tei(self, texts: list[str]) -> list[list[float]]:
        url = self.endpoint_url if self.endpoint_url.endswith("/embed") else f"{self.endpoint_url}/embed"
        payload = self._post(url, {"inputs": texts, "truncate": True})
        return vectors_from_tei_payload(payload, len(texts))

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        url = (
            self.endpoint_url
            if self.endpoint_url.endswith("/embeddings")
            else f"{self.endpoint_url}/v1/embeddings"
        )
        body: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimension,
        }
        try:
            payload = self._post(url, body)
        except EmbedError as exc:
            if exc.status not in {400, 422} or "dimension" not in str(exc).lower():
                raise
            LOGGER.warning(
                "endpoint rejected the dimensions argument; requesting the full vector and truncating locally"
            )
            payload = self._post(url, {"model": self.model, "input": texts})
        return vectors_from_openai_payload(payload, len(texts))


def vectors_from_tei_payload(payload: Any, count: int) -> list[list[float]]:
    """Read a Text Embeddings Inference `/embed` response."""
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return vectors_from_openai_payload(payload, count)

    vectors = payload.get("embeddings") if isinstance(payload, dict) else payload
    if not isinstance(vectors, list):
        raise EmbedError(
            "TEI response was not a list of embeddings",
            status=200,
            fatal=True,
            split=False,
        )
    if count == 1 and vectors and isinstance(vectors[0], (int, float)):
        vectors = [vectors]
    if (
        vectors
        and isinstance(vectors[0], list)
        and vectors[0]
        and isinstance(vectors[0][0], list)
    ):
        raise EmbedError(
            "endpoint returned token embeddings instead of one pooled vector per text. "
            "Deploy Text Embeddings Inference so last-token pooling is done by the server.",
            status=200,
            fatal=True,
            split=False,
        )
    if len(vectors) != count:
        raise EmbedError(
            f"TEI returned {len(vectors)} embeddings for {count} texts",
            status=200,
            fatal=False,
            split=True,
        )
    return vectors


def require_endpoint() -> tuple[str, str]:
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    endpoint = os.environ.get("HF_F2LLM_ENDPOINT_URL", "").strip()
    if token and endpoint:
        return token, endpoint
    raise SystemExit(ENDPOINT_HELP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize titles, embed them with F2LLM-v2-4B via Hugging Face, and upsert Milvus."
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
    token, endpoint = require_endpoint()
    client = HuggingFaceEmbeddings(token, endpoint)
    config = ImportConfig(
        model_name="F2LLM-v2-4B",
        collection_name=env_str("F2LLM_COLLECTION"),
        checkpoint_path=resolve_path(env_str("F2LLM_CHECKPOINT_DB")),
        embed_batch_size=env_int("F2LLM_EMBED_BATCH_SIZE"),
        milvus_batch_size=env_int("F2LLM_MILVUS_BATCH_SIZE"),
        dimension=client.dimension,
        log_name="f2llm_v2_4b",
    )
    return run_import(client.embed, config, limit=args.limit, retry_errors=args.retry_errors)


if __name__ == "__main__":
    raise SystemExit(main())
