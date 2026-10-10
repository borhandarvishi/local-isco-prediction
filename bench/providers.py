"""Embedding clients for OpenRouter and the F2LLM Hugging Face endpoint."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from bench.env import env_int, env_str

LOGGER_F2LLM = "f2llm_v2_4b"

ENDPOINT_HELP = """
F2LLM-v2-4B is not on OpenRouter, and Hugging Face does not serve it on serverless inference.
Deploy a Hugging Face Inference Endpoint for codefuse-ai/F2LLM-v2-4B with the Text Embeddings Inference engine, then set:
  HF_TOKEN
  HF_F2LLM_ENDPOINT_URL   (the endpoint origin is enough)
  F2LLM_API_STYLE=tei
If the endpoint speaks the OpenAI embeddings API instead, set F2LLM_API_STYLE=openai.
""".strip()


class EmbedError(Exception):
    """A provider call failed.

    fatal: stop the whole run (authentication, unexpected response).
    split: the batch itself was rejected, so try smaller batches.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        fatal: bool,
        split: bool,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.fatal = fatal
        self.split = split


def request_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
    max_retries: int,
    logger: logging.Logger,
) -> Any:
    """POST JSON and retry rate limits and transient server errors."""
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            logger.warning("request failed, attempt %s/%s: %s", attempt, max_retries, exc)
            if attempt == max_retries:
                raise EmbedError(str(exc), status=None, fatal=False, split=False) from exc
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                snippet = response.text[:300].replace("\n", " ")
                raise EmbedError(
                    f"response was not JSON: {snippet}",
                    status=200,
                    fatal=True,
                    split=False,
                ) from exc

        snippet = response.text[:500].replace("\n", " ")
        logger.warning(
            "HTTP %s from %s, attempt %s/%s: %s",
            response.status_code,
            url,
            attempt,
            max_retries,
            snippet,
        )
        if response.status_code in {401, 403}:
            raise EmbedError(snippet, status=response.status_code, fatal=True, split=False)
        if response.status_code in {400, 413, 422}:
            raise EmbedError(snippet, status=response.status_code, fatal=False, split=True)
        if response.status_code in {429, 500, 502, 503, 504}:
            if attempt == max_retries:
                raise EmbedError(snippet, status=response.status_code, fatal=False, split=False)
            retry_after = response.headers.get("Retry-After")
            sleep_for = delay
            if retry_after:
                try:
                    sleep_for = min(float(retry_after), 120.0)
                except ValueError:
                    sleep_for = delay
            time.sleep(sleep_for)
            delay = min(delay * 2, 60)
            continue
        raise EmbedError(snippet, status=response.status_code, fatal=True, split=False)

    raise EmbedError("request retry loop ended without a response", status=None, fatal=False, split=False)


def vectors_from_openai_payload(payload: Any, count: int) -> list[list[float]]:
    """Read an OpenAI-style embeddings response, ordered by the `index` field."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise EmbedError("response has no embedding data list", status=200, fatal=True, split=False)
    try:
        ordered = sorted(data, key=lambda item: item["index"])
    except (TypeError, KeyError) as exc:
        raise EmbedError(f"embedding items have no index: {exc}", status=200, fatal=True, split=False) from exc
    indexes = [item.get("index") for item in ordered]
    if indexes != list(range(count)):
        raise EmbedError(
            f"unexpected embedding indexes {indexes}",
            status=200,
            fatal=False,
            split=True,
        )
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list):
            raise EmbedError("embedding field is not a list", status=200, fatal=False, split=True)
        vectors.append(vector)
    return vectors


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


class OpenRouterEmbeddings:
    """OpenAI-compatible embeddings on OpenRouter. Used by Qwen and text-embedding-3-small."""

    def __init__(self, model_env: str, logger_name: str) -> None:
        self.base_url = env_str("OPENROUTER_BASE_URL").rstrip("/")
        self.model = env_str(model_env)
        self.dimension = env_int("EMBEDDING_DIM")
        self.timeout = env_int("EMBED_TIMEOUT_SECONDS")
        self.max_retries = env_int("EMBED_MAX_RETRIES")
        self.session = requests.Session()
        self.logger = logging.getLogger(logger_name)
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
            logger=self.logger,
        )
        if not self._logged_provider and isinstance(payload, dict):
            self.logger.info(
                "openrouter model=%s provider=%s",
                payload.get("model", self.model),
                payload.get("provider", "unknown"),
            )
            self._logged_provider = True
        return vectors_from_openai_payload(payload, len(texts))


class HuggingFaceEmbeddings:
    """F2LLM on a dedicated Hugging Face Inference Endpoint."""

    def __init__(self, token: str, endpoint_url: str) -> None:
        self.token = token
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = env_str("HF_F2LLM_MODEL")
        self.api_style = env_str("F2LLM_API_STYLE").lower()
        self.dimension = env_int("EMBEDDING_DIM")
        self.timeout = env_int("EMBED_TIMEOUT_SECONDS")
        self.max_retries = env_int("EMBED_MAX_RETRIES")
        self.session = requests.Session()
        self.logger = logging.getLogger(LOGGER_F2LLM)
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
            logger=self.logger,
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
            self.logger.warning(
                "endpoint rejected the dimensions argument; requesting the full vector and truncating locally"
            )
            payload = self._post(url, {"model": self.model, "input": texts})
        return vectors_from_openai_payload(payload, len(texts))


def require_endpoint() -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    endpoint = os.environ.get("HF_F2LLM_ENDPOINT_URL", "").strip()
    if token and endpoint:
        return token, endpoint
    raise SystemExit(ENDPOINT_HELP)


def qwen_client() -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings("QWEN_EMBED_MODEL", "qwen3_embedding_4b")


def qwen8b_client() -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings("QWEN8B_EMBED_MODEL", "qwen3_embedding_8b")


def openai_client() -> OpenRouterEmbeddings:
    return OpenRouterEmbeddings("OPENAI_EMBED_MODEL", "openai_text_embedding_3_small")


def f2llm_client() -> HuggingFaceEmbeddings:
    token, endpoint = require_endpoint()
    return HuggingFaceEmbeddings(token, endpoint)
