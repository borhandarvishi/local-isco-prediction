# How to run the embedding benchmark

Run every command from the project root, with the project virtualenv:

```bash
cd /Users/borhan/Desktop/PC-8-Sep-2026/PROJECTS/local-isco-prediction
.venv/bin/python -m bench <command>
```

Settings come from `.env`. Titles are normalized with `clean_text_readable` before they are embedded. Vectors are 1024 dimensions and L2-normalized. No query instruction is added.

If a run stops, start the same command again. Finished batches stay in the checkpoint and are not sent to the provider again.

## 1. Build the validation queries

```bash
.venv/bin/python -m bench queries
```

Reads `data/source/test_data_for_local_embedding_models.xlsx` and writes 200 human-typed queries.

**Output**

- `data/validation/validation_human_job_queries.xlsx`
- Sheet `ValidationQueries`, columns: `query`, `gold_id`, `gold_title`, `change`
- Terminal line: `wrote 200 rows to validation_human_job_queries.xlsx`

Nothing is embedded and nothing is sent to Milvus.

## 2. Import job titles into Milvus

Each command embeds the `title` column from `data/source/test_data_for_local_embedding_models.xlsx` (1999 rows) and upserts one Milvus collection. Stored text is the normalized title. `title_raw` keeps the original spreadsheet title.

```bash
.venv/bin/python -m bench import qwen
.venv/bin/python -m bench import qwen8b
.venv/bin/python -m bench import openai
.venv/bin/python -m bench import f2llm
```

Optional flags, same for all three:

```bash
.venv/bin/python -m bench import qwen --limit 10
.venv/bin/python -m bench import qwen --retry-errors
```

`--limit N` embeds only the first N spreadsheet rows. `--retry-errors` clears saved per-row errors and tries those rows again.

F2LLM calls the Hugging Face endpoint in `HF_F2LLM_ENDPOINT_URL`. That endpoint must be running. Qwen and OpenAI go through OpenRouter.

**Output**

| Command | Milvus collection | Checkpoint | Log |
|---|---|---|---|
| `import qwen` | `bench_qwen3_embedding_4b_1024` | `checkpoints/qwen3_embedding_4b.sqlite` | `logs/qwen3_embedding_4b.log` |
| `import qwen8b` | `bench_qwen3_embedding_8b_1024` | `checkpoints/qwen3_embedding_8b.sqlite` | `logs/qwen3_embedding_8b.log` |
| `import openai` | `bench_openai_text_embedding_3_small_1024` | `checkpoints/openai_text_embedding_3_small.sqlite` | `logs/openai_text_embedding_3_small.log` |
| `import f2llm` | `bench_f2llm_v2_4b_1024` | `checkpoints/f2llm_v2_4b.sqlite` | `logs/f2llm_v2_4b.log` |

Each collection has fields `id`, `title`, `title_raw`, and `embedding` (float vector, dim 1024, COSINE, HNSW). `id` is the spreadsheet id.

A finished run logs `import complete` and the collection row count. A full import has 1999 rows.

## 3. Embed the validation queries with F2LLM

```bash
.venv/bin/python -m bench embed-queries
```

Reads the `query` column from `data/validation/validation_human_job_queries.xlsx`, normalizes it, and sends that text to the F2LLM endpoint. `gold_id` and `gold_title` are stored beside the vector and are not embedded. Nothing is written to Milvus.

The Hugging Face endpoint must be running.

**Output**

- `data/validation/vectors/validation_f2llm_query_vectors.npz`
  - `vectors`: float32, shape `(200, 1024)`
  - `gold_id`, `query`, `query_normalized`, `gold_title`
  - Rows are ordered by `gold_id`, not by the spreadsheet row order
- `checkpoints/f2llm_validation_queries.sqlite`
- `logs/f2llm_validation_queries.log`

## 4. Retrieve the nearest title for each query

```bash
.venv/bin/python -m bench retrieve
```

For each of the 200 queries, this searches that model's own collection with that model's own query vector:

- F2LLM reuses `data/validation/vectors/validation_f2llm_query_vectors.npz`. Run `embed-queries` first if that file is missing.
- OpenAI and Qwen embed the same normalized queries when their vector files are missing, then reuse those files on the next run.

All three Milvus collections must already exist and contain the imported titles.

**Output**

- `data/validation/results/validation_retrieval_top1.xlsx`
  - Sheet `Top1`
  - Columns: `query`, `gold_title`, `gold_id`, then for each model `*_top_id`, `*_top_title`, `*_top_score`
  - Models: `f2llm_v2_4b`, `openai_small_3`, `qwen3_4b`, `qwen3_8b`
  - `top_title` is the original spreadsheet title of the nearest Milvus row
  - `top_score` is Milvus cosine similarity. Higher means closer
  - `top_id` equal to `gold_id` means the nearest row is the expected title
- `data/validation/vectors/validation_openai_query_vectors.npz` (created on the first OpenAI embed)
- `data/validation/vectors/validation_qwen3_query_vectors.npz` (created on the first Qwen 4B embed)
- `data/validation/vectors/validation_qwen3_8b_query_vectors.npz` (created on the first Qwen 8B embed)
- `logs/validation_retrieval_top1.log`

The log ends with a count of how many top-1 ids matched `gold_id` for each model, and `wrote validation_retrieval_top1.xlsx`.
