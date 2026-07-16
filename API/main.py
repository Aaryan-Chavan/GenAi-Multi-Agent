# API/main.py
"""
Application entrypoint -- interactive CLI mode.

Run with:
    python main.py                 -> interactive loop, type queries one at a time
    python main.py "your query"    -> answers that one query, then exits
    python main.py --skip-pipeline "your query"  -> skip the dataset pipeline check

Builds StructuredAgent + SemanticAgent + AnswerGenerator -> HybridAgent
directly (no FastAPI, no HTTP, no UI/, no Evaluation/) and sends each
query straight to HybridAgent.run(). This is intentionally the same
construction logic as API/dependencies.py's _build_hybrid_agent(), just
called in-process instead of behind a web server -- once UI/ is ready,
the FastAPI app in this same file's server mode (see git history / ask
if you want it back) can be restored alongside this CLI without either
one duplicating agent-construction logic.

NOTE ON dataset_pipeline.py:
dataset_pipeline.py is now a plain script (not a class) that exposes a
single `main()` function. That function runs the full ingestion ->
embeddings -> DuckDB/Qdrant pipeline stage-by-stage, and every stage
already checks on its own whether its output exists / is valid before
doing any work (e.g. it skips load/clean/map if CLEANED_DATA_FILE
already exists, skips reloading DuckDB tables if row counts already
match, skips regenerating embeddings if the .npy file is already on
disk, skips reloading Qdrant if the vector count already matches).
This means calling dataset_pipeline.main() on every launch is safe and
cheap when everything is already built -- there is no separate
status()/validate() call to make anymore, that logic now lives inside
dataset_pipeline.py itself.
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

# Make the project root importable regardless of the current working
# directory (`python main.py` from inside API/, or `python API/main.py`
# from the project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agents.hybrid_agent import HybridAgent, HybridAgentConfig, HybridResult
from Agents.semantic_agent import (
    EmbedConfig,
    LLMConfig as SemanticLLMConfig,
    QdrantConfig,
    RetrievalConfig,
    SchemaConfig as SemanticSchemaConfig,
    SemanticAgent,
)
from Agents.structured_agent import AgentConfig, StructuredAgent
from Config.settings import (
    DUCKDB_FILE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    LOG_FILE,
    LOG_LEVEL,
    MAX_RETRIEVAL_RESULTS,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    TOP_K_DUCKDB,
)
from LLM.answer_generator import AnswerGenerator

# dataset_pipeline.py lives at the project root (sibling of API/, Agents/,
# Config/, ...), importable once PROJECT_ROOT is on sys.path (see above).
# It now exposes a single `main()` entrypoint -- alias it so it doesn't
# collide with this file's own `main()` below.
from dataset_pipeline import main as run_dataset_pipeline_stages

logger = logging.getLogger(__name__)


# ==========================================================
# ERRORS
# ==========================================================


class DatasetPipelineError(Exception):
    """Raised when dataset_pipeline.main() fails to complete."""


# ==========================================================
# LOGGING
# ==========================================================


def _configure_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"},
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default", "level": LOG_LEVEL},
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "level": LOG_LEVEL,
                    "filename": str(LOG_FILE),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {"level": LOG_LEVEL, "handlers": ["console", "file"]},
        }
    )


# ==========================================================
# DATASET PIPELINE  (ingestion -> embeddings -> DuckDB/Qdrant, runs
# BEFORE any agent is built -- HybridAgent reads from what this writes)
# ==========================================================


def run_dataset_pipeline() -> None:
    """Runs the full ingestion/embedding/storage pipeline via
    dataset_pipeline.main().

    dataset_pipeline.main() is idempotent by design: every stage inside
    it (load/clean/map, DuckDB load, embedding generation, quantization,
    precomputed intelligence, Qdrant load) checks whether its expected
    output already exists and matches the expected row/vector counts
    before doing any work, and skips itself if so. That means it is
    always safe (and normally cheap) to call this on every launch --
    there's no separate "check status first" step anymore since that
    logic now lives inside dataset_pipeline.py itself.

    Raises DatasetPipelineError (uncaught, by design) if the pipeline
    cannot complete -- there is no point starting HybridAgent against a
    DuckDB/Qdrant store the pipeline itself couldn't populate.
    """
    logger.info("Running dataset pipeline (stages skip themselves internally if already done)")
    try:
        run_dataset_pipeline_stages()
        logger.info("Dataset pipeline finished.")
        print("Dataset pipeline: up to date.")
    except Exception as exc:
        raise DatasetPipelineError(f"Dataset pipeline failed: {exc}") from exc


# ==========================================================
# AGENT CONSTRUCTION  (same wiring as API/dependencies.py, no HTTP layer)
# ==========================================================


def build_hybrid_agent() -> HybridAgent:
    logger.info("Building HybridAgent (StructuredAgent + SemanticAgent + AnswerGenerator)")

    structured_agent = StructuredAgent(
        db_path=str(DUCKDB_FILE),
        max_rows=TOP_K_DUCKDB,
        config=AgentConfig(max_row_limit=max(TOP_K_DUCKDB, MAX_RETRIEVAL_RESULTS)),
    )

    semantic_agent = SemanticAgent(
        qdrant_cfg=QdrantConfig(
            host=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            collection=QDRANT_COLLECTION,
        ),
        embed_cfg=EmbedConfig(
            model_name=EMBEDDING_MODEL,
            device=None if EMBEDDING_DEVICE == "auto" else EMBEDDING_DEVICE,
        ),
        schema_cfg=SemanticSchemaConfig(),
        retrieval_cfg=RetrievalConfig(),
        llm_cfg=SemanticLLMConfig(),
    )

    answer_generator = AnswerGenerator()

    return HybridAgent(
        structured_agent=structured_agent,
        semantic_agent=semantic_agent,
        answer_generator=answer_generator,
        config=HybridAgentConfig(),
    )


# ==========================================================
# OUTPUT
# ==========================================================


def _print_result(result: HybridResult) -> None:
    print(f"\nAnswer:\n{result.answer or '(no answer produced)'}")
    print(f"\nConfidence: {result.overall_confidence:.2f}")
    print(f"Execution path: {result.routing.mode.execution_path}")
    print(f"Latency: {result.total_latency_ms:.1f}ms")
    if result.errors:
        print(f"Errors: {'; '.join(result.errors)}")
    if result.warnings:
        print(f"Warnings: {'; '.join(result.warnings)}")


# ==========================================================
# CLI
# ==========================================================


async def _run_one(agent: HybridAgent, query: str, conversation_id: str = "default") -> None:
    result = await agent.run(query=query, conversation_id=conversation_id)
    _print_result(result)


async def _run_interactive(agent: HybridAgent) -> None:
    print("Hybrid RAG -- interactive mode. Type a query, or 'exit'/'quit' to stop.")
    conversation_id = "cli-session"
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break
        try:
            await _run_one(agent, query, conversation_id)
        except Exception:
            logger.exception("Query failed: %r", query)
            print("Something went wrong processing that query -- see logs for details.")


async def _amain(argv: Optional[list] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    agent = build_hybrid_agent()
    try:
        if argv:
            # python main.py "your query" -> single-shot mode
            await _run_one(agent, " ".join(argv))
        else:
            # python main.py -> interactive loop, dynamic queries
            await _run_interactive(agent)
        return 0
    finally:
        await agent.close()


def main() -> int:
    _configure_logging()

    argv = sys.argv[1:]
    skip_pipeline = "--skip-pipeline" in argv
    if skip_pipeline:
        argv = [a for a in argv if a != "--skip-pipeline"]

    if not skip_pipeline:
        try:
            run_dataset_pipeline()
        except DatasetPipelineError:
            logger.exception("Dataset pipeline failed -- not starting agents against an unready store")
            print(
                "Dataset pipeline failed; see logs for details. "
                "Fix the dataset/config, or pass --skip-pipeline to bypass this check "
                "if you know DuckDB/Qdrant are already populated."
            )
            return 1
    else:
        logger.info("--skip-pipeline passed; skipping dataset pipeline")

    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception:
        logger.exception("Fatal error building or running HybridAgent")
        return 1


if __name__ == "__main__":
    sys.exit(main())