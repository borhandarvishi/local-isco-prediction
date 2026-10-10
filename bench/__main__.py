"""One entry point for the embedding benchmark.

    python -m bench import qwen
    python -m bench import qwen8b
    python -m bench import openai
    python -m bench import f2llm
    python -m bench queries
    python -m bench embed-queries
    python -m bench retrieve

`import` accepts --limit N and --retry-errors.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from bench.env import env_int, env_str, load_environment, resolve_path
from bench.providers import (
    HuggingFaceEmbeddings,
    OpenRouterEmbeddings,
    f2llm_client,
    openai_client,
    qwen8b_client,
    qwen_client,
)
from bench.query_embed import embed_f2llm_queries
from bench.retrieve import retrieve_top1
from bench.titles import ImportConfig, run_import
from bench.validation_set import write_query_workbook

Client = OpenRouterEmbeddings | HuggingFaceEmbeddings
IMPORTS: dict[str, tuple[str, str, str, str, str, str, Callable[[], Client]]] = {
    "qwen": (
        "Qwen3-Embedding-4B",
        "qwen3_embedding_4b",
        "QWEN_COLLECTION",
        "QWEN_CHECKPOINT_DB",
        "QWEN_EMBED_BATCH_SIZE",
        "QWEN_MILVUS_BATCH_SIZE",
        qwen_client,
    ),
    "qwen8b": (
        "Qwen3-Embedding-8B",
        "qwen3_embedding_8b",
        "QWEN8B_COLLECTION",
        "QWEN8B_CHECKPOINT_DB",
        "QWEN8B_EMBED_BATCH_SIZE",
        "QWEN8B_MILVUS_BATCH_SIZE",
        qwen8b_client,
    ),
    "openai": (
        "OpenAI text-embedding-3-small",
        "openai_text_embedding_3_small",
        "OPENAI_COLLECTION",
        "OPENAI_CHECKPOINT_DB",
        "OPENAI_EMBED_BATCH_SIZE",
        "OPENAI_MILVUS_BATCH_SIZE",
        openai_client,
    ),
    "f2llm": (
        "F2LLM-v2-4B",
        "f2llm_v2_4b",
        "F2LLM_COLLECTION",
        "F2LLM_CHECKPOINT_DB",
        "F2LLM_EMBED_BATCH_SIZE",
        "F2LLM_MILVUS_BATCH_SIZE",
        f2llm_client,
    ),
}


def parse_import_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Only the first N spreadsheet rows. 0 means all.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Clear saved per-row embed errors and try those rows again.",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    return args


def import_model(name: str, *, limit: int, retry_errors: bool) -> int:
    model_name, log_name, collection_env, checkpoint_env, embed_batch_env, milvus_batch_env, factory = IMPORTS[name]
    load_environment()
    client = factory()
    config = ImportConfig(
        model_name=model_name,
        collection_name=env_str(collection_env),
        checkpoint_path=resolve_path(env_str(checkpoint_env)),
        embed_batch_size=env_int(embed_batch_env),
        milvus_batch_size=env_int(milvus_batch_env),
        dimension=client.dimension,
        log_name=log_name,
    )
    return run_import(client.embed, config, limit=limit, retry_errors=retry_errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local job-title embedding benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    import_parser = sub.add_parser("import", help="Normalize titles, embed them, and upsert Milvus.")
    import_parser.add_argument("model", choices=sorted(IMPORTS))
    import_parser.add_argument("--limit", type=int, default=0, help="Only the first N spreadsheet rows. 0 means all.")
    import_parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Clear saved per-row embed errors and try those rows again.",
    )

    sub.add_parser("queries", help="Write validation_human_job_queries.xlsx.")
    sub.add_parser("embed-queries", help="Embed validation queries with F2LLM and save the npz file.")
    sub.add_parser("retrieve", help="Search each model's collection and write the top-1 workbook.")

    args = parser.parse_args(argv)
    if args.command == "import":
        if args.limit < 0:
            parser.error("--limit must be 0 or greater")
        return import_model(args.model, limit=args.limit, retry_errors=args.retry_errors)
    if args.command == "queries":
        path = write_query_workbook()
        print(f"wrote 200 rows to {path.name}")
        return 0
    if args.command == "embed-queries":
        return embed_f2llm_queries()
    if args.command == "retrieve":
        return retrieve_top1()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
