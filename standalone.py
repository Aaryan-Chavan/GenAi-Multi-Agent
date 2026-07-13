"""
test_hybrid_pipeline.py
============================================================
THIN, EFFICIENT CLI HARNESS FOR hybrid_agent.HybridAgent

This version stops duplicating logic that already lives in
`hybrid_agent.py`. The previous revision of this file hand-rolled its
own schema introspection, query planner, join-key discovery, Qdrant
filter construction, fusion, and confidence scoring -- all of which
`HybridAgent` already does, generically, adaptively, and with caching,
MMR diversification, LLM-assisted routing, and adaptive confidence
weighting that this file could never keep in sync with by hand.

What this file does now:

  1. Builds a single `HybridAgent`, wired to your real DuckDB file and
     real Qdrant collection/embedding model, via `Config.settings` or
     environment variables (same knobs as before).
  2. Runs each query through `HybridAgent.run_streaming()` so the plan
     and each branch's completion are printed as soon as they're
     available, instead of waiting for the whole pipeline to finish.
  3. Hands the resulting `compressed_context` to a real LLM
     `AnswerGenerator` (LLM.answer_generator) if importable, and falls
     back to a data-grounded summary assembled directly from the
     `HybridResult` sections HybridAgent already built
     (structured_summary / metric_summary / trend_observations /
     complaint_summary / key_snippets) -- no reimplementation of
     "what does the evidence say" needed.
  4. Prints latency/confidence metrics straight from `HybridResult`
     and closes the agent cleanly on exit.

No synthetic data anywhere. No duplicate planner, schema-inference,
join-detection, or fusion code -- HybridAgent owns all of that.

Run with:
    python test_hybrid_pipeline.py

Required:
    - hybrid_agent.py importable on PYTHONPATH (this is the only hard
      dependency this file itself checks for; hybrid_agent.py in turn
      pulls in StructuredAgent / SemanticAgent, which enforce their own
      duckdb / qdrant-client / sentence-transformers requirements).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("test_hybrid_pipeline")


class HarnessConfigurationError(RuntimeError):
    """Raised when a required real dependency/service is unavailable.
    Fatal and actionable -- this harness never falls back to synthetic
    data."""


# ============================================================
# HARD REQUIREMENT: hybrid_agent.py itself. Everything else
# (duckdb, qdrant-client, sentence-transformers) is enforced
# transitively by StructuredAgent / SemanticAgent when HybridAgent
# constructs them.
# ============================================================

try:
    from Agents.hybrid_agent import (
        HybridAgent,
        HybridConfig,
        HybridResult,
        ExecutionMode,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise HarnessConfigurationError(
        "hybrid_agent.HybridAgent could not be imported. This harness is a "
        f"thin wrapper around it and cannot run without it. Original error: {exc}"
    ) from exc

# Same config classes HybridAgent itself imports, resolved the same way
# (package-relative first, flat fallback second) so this file stays in
# lockstep with whatever layout Agents/ actually has.
try:
    from Agents.structured_agent import LLMConfig
    from Agents.semantic_agent import QdrantConfig, EmbedConfig
except ImportError:
    try:
        from Agents.structured_agent import LLMConfig
        from Agents.semantic_agent import QdrantConfig, EmbedConfig
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HarnessConfigurationError(
            f"Could not import LLMConfig / QdrantConfig / EmbedConfig: {exc}"
        ) from exc

REAL_ANSWER_GENERATOR_AVAILABLE = False
try:  # pragma: no cover - environment dependent
    from LLM.answer_generator import AnswerGenerator as _RealAnswerGenerator  # type: ignore
    from LLM.answer_generator import AnswerType as _RealAnswerType  # type: ignore
    REAL_ANSWER_GENERATOR_AVAILABLE = True
except Exception as exc:
    LOGGER.warning("Real AnswerGenerator unavailable (%s); a data-grounded fallback will be used.", exc)
    _RealAnswerGenerator = None  # type: ignore
    _RealAnswerType = None  # type: ignore


# ============================================================
# CONFIGURATION (Config.settings, with env-var fallback)
# ============================================================

class _SettingsShim:
    DUCKDB_FILE = os.environ.get("DUCKDB_FILE", "./database/duckdb/analytics.duckdb")
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    TOP_K = int(os.environ.get("SEMANTIC_TOP_K", "8"))
    SCORE_THRESHOLD = float(os.environ.get("SEMANTIC_SCORE_THRESHOLD", "0.0"))


try:
    from Config import settings  # type: ignore
    LOGGER.info("Loaded configuration from Config.settings")
except Exception as exc:  # pragma: no cover - environment dependent
    LOGGER.warning(
        "Config.settings not importable (%s); using environment-variable "
        "configuration fallback.", exc,
    )
    settings = _SettingsShim()  # type: ignore

DUCKDB_FILE: str = getattr(settings, "DUCKDB_FILE", _SettingsShim.DUCKDB_FILE)
QDRANT_HOST: str = getattr(settings, "QDRANT_HOST", _SettingsShim.QDRANT_HOST)
QDRANT_PORT: int = int(getattr(settings, "QDRANT_PORT", _SettingsShim.QDRANT_PORT))
QDRANT_COLLECTION: str = getattr(settings, "QDRANT_COLLECTION", _SettingsShim.QDRANT_COLLECTION)
EMBEDDING_MODEL: str = getattr(settings, "EMBEDDING_MODEL", _SettingsShim.EMBEDDING_MODEL)
SEMANTIC_TOP_K: int = int(getattr(settings, "TOP_K", _SettingsShim.TOP_K))
SCORE_THRESHOLD: float = float(getattr(settings, "SCORE_THRESHOLD", _SettingsShim.SCORE_THRESHOLD))

# HybridAgent-level knobs. "fast"/"balanced"/"deep" map straight onto
# HybridConfig.execution_profiles -- see hybrid_agent.ExecutionMode.
HYBRID_MODE: str = os.environ.get("HYBRID_MODE", ExecutionMode.BALANCED)
CONTEXT_TOKEN_BUDGET: int = int(os.environ.get("HYBRID_CONTEXT_TOKEN_BUDGET", "6000"))
BRANCH_TIMEOUT: float = float(os.environ.get("HYBRID_BRANCH_TIMEOUT", "60.0"))
CACHE_TTL: float = float(os.environ.get("HYBRID_CACHE_TTL", "300.0"))
ENABLE_CACHE: bool = os.environ.get("HYBRID_ENABLE_CACHE", "1") not in ("0", "false", "False")

# Optional LLM routing/generation config. Only built if at least one of
# these is set -- otherwise HybridAgent uses its default heuristic
# planner and this harness uses the data-grounded answer fallback.
LLM_ENDPOINT: Optional[str] = os.environ.get("LLM_ENDPOINT")
LLM_MODEL: Optional[str] = os.environ.get("LLM_MODEL")
LLM_TIMEOUT: Optional[float] = float(os.environ["LLM_TIMEOUT"]) if os.environ.get("LLM_TIMEOUT") else None
LLM_TEMPERATURE: Optional[float] = float(os.environ["LLM_TEMPERATURE"]) if os.environ.get("LLM_TEMPERATURE") else None
LLM_MAX_TOKENS: Optional[int] = int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None


def _build_config(cls: type, candidates: Dict[str, Any]) -> Any:
    """Instantiate a (dataclass) config object using only the keyword
    arguments it actually accepts. This keeps the harness generic and
    resilient to whatever field names Agents/structured_agent.py and
    Agents/semantic_agent.py happen to use, instead of hardcoding a
    guess that silently breaks if those modules change."""
    try:
        valid = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set(candidates)
    except Exception:
        valid = set(candidates)
    kwargs = {k: v for k, v in candidates.items() if k in valid and v is not None}
    try:
        return cls(**kwargs)
    except Exception:
        LOGGER.warning("Could not construct %s with %s; falling back to defaults.", cls.__name__, kwargs)
        try:
            return cls()
        except Exception:
            LOGGER.warning("%s has no zero-arg default either; leaving it as None.", cls.__name__)
            return None


def _build_llm_config() -> Optional[Any]:
    if not (LLM_ENDPOINT or LLM_MODEL):
        return None
    return _build_config(LLMConfig, {
        "endpoint": LLM_ENDPOINT, "model": LLM_MODEL, "timeout": LLM_TIMEOUT,
        "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS,
    })


def _build_qdrant_config() -> Any:
    return _build_config(QdrantConfig, {
        "host": QDRANT_HOST, "port": QDRANT_PORT,
        "collection": QDRANT_COLLECTION, "collection_name": QDRANT_COLLECTION,
        "url": f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        "score_threshold": SCORE_THRESHOLD, "top_k": SEMANTIC_TOP_K,
    })


def _build_embed_config() -> Any:
    return _build_config(EmbedConfig, {
        "model_name": EMBEDDING_MODEL, "model": EMBEDDING_MODEL, "embedding_model": EMBEDDING_MODEL,
    })


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _hr(char: str = "=", width: int = 92) -> str:
    return char * width


def section(title: str) -> None:
    print(f"\n{_hr('-')}")
    print(title)
    print(_hr('-'))


def _preview_duckdb_schema(db_path: str) -> None:
    """Best-effort, read-only, purely informational listing of what's in
    the DuckDB file -- NOT used for planning (HybridAgent's own
    StructuredAgent does its own schema introspection internally when it
    connects). Skipped silently if duckdb isn't importable or the file
    can't be opened read-only (e.g. it doesn't exist yet)."""
    try:
        import duckdb
    except ImportError:
        LOGGER.info("duckdb not importable here; skipping schema preview (HybridAgent connects independently).")
        return
    section("🔧 DUCKDB SCHEMA PREVIEW (informational only)")
    try:
        conn = duckdb.connect(db_path, read_only=True)
        try:
            tables = sorted({r[0] for r in conn.execute("SHOW TABLES").fetchall()})
            if not tables:
                print("(no tables found)")
            for t in tables:
                try:
                    row_count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                    cols = [r[0] for r in conn.execute(f'DESCRIBE "{t}"').fetchall()]
                except Exception:
                    row_count, cols = "?", []
                print(f"TABLE {t} | rows={row_count} | columns={cols}")
        finally:
            conn.close()
    except Exception as exc:
        LOGGER.warning("Could not preview DuckDB schema (%s); continuing.", exc)


# ============================================================
# ANSWER GENERATION -- real LLM if available, otherwise assembled
# directly from the sections HybridAgent already compressed.
# ============================================================

async def build_answer(query: str, result: HybridResult, real_generator: Optional[Any]) -> Tuple[str, bool]:
    if real_generator is not None:
        try:
            gen_result = real_generator.generate(
                query=query,
                context=result.compressed_context,
                answer_type=_RealAnswerType.HYBRID if _RealAnswerType is not None else None,
            )
            return getattr(gen_result, "answer", str(gen_result)), True
        except Exception as exc:
            LOGGER.warning("Real AnswerGenerator.generate() failed (%s); using data-grounded fallback.", exc)

    parts: List[str] = []
    if result.structured_summary:
        parts.append(f"Structured findings:\n{result.structured_summary}")
    if result.metric_summary:
        parts.append(f"Metrics:\n{result.metric_summary}")
    if result.trend_observations:
        parts.append("Trends:\n" + "\n".join(f"- {t}" for t in result.trend_observations))
    if result.complaint_summary:
        parts.append(f"Breakdown:\n{result.complaint_summary}")
    if result.key_snippets:
        parts.append("Evidence:\n" + "\n".join(result.key_snippets[:5]))
    if not parts:
        parts.append("No structured or semantic evidence was retrieved for this query.")
    parts.append(
        f"[confidence={result.confidence}, path={result.execution_path}, "
        f"reasoning={result.plan_reasoning!r}]"
    )
    return "\n\n".join(parts), False


# ============================================================
# QUERY EXECUTION -- streams plan + branch completion as they land,
# so the user sees progress immediately instead of one big blocking
# call.
# ============================================================

async def run_query(agent: HybridAgent, query: str, real_generator: Optional[Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result: Optional[HybridResult] = None

    section("🧠 HYBRID AGENT")
    async for event in agent.run_streaming(query=query):
        stage = event.get("stage")
        if stage == "plan":
            print(f"Execution path   : {event['path']}")
            print(f"Reasoning        : {event['reasoning']}")
            print(f"Plan confidence  : {event['confidence']}")
        elif stage in ("sql_done", "semantic_done"):
            print(f"{stage.replace('_', ' ').title():<14} -> success={event['success']}")
        elif stage == "cache_hit":
            result = event["result"]
            print("Result served from cache.")
        elif stage == "final":
            result = event["result"]

    if result is None:
        print("[FAILED] No result produced.")
        return {
            "query": query, "success": False, "path": "n/a", "confidence": 0.0,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "facts": 0, "chunks": 0, "used_real_llm": False,
        }

    section(f"📊 STRUCTURED FACTS ({len(result.structured_facts)})")
    if result.structured_facts:
        for row in result.structured_facts[:5]:
            print(f"  - {row}")
        if len(result.structured_facts) > 5:
            print(f"  ... and {len(result.structured_facts) - 5} more")
    else:
        print("(none)")

    section(f"🔎 SEMANTIC CHUNKS ({len(result.semantic_chunks)})")
    if result.semantic_chunks:
        for c in result.semantic_chunks[:5]:
            print(f"  - {c}")
        if len(result.semantic_chunks) > 5:
            print(f"  ... and {len(result.semantic_chunks) - 5} more")
    else:
        print("(none)")

    if result.compressed_context:
        section("🧾 COMPRESSED CONTEXT")
        preview = result.compressed_context
        if len(preview) > 1800:
            preview = preview[:1800] + "\n... [truncated for display] ..."
        print(preview)

    section("🤖 ANSWER")
    if not result.success:
        print(f"[Both branches failed or were skipped] sql_error={result.sql_error} semantic_error={result.semantic_error}")
        answer, used_llm = "", False
    else:
        answer, used_llm = await build_answer(query, result, real_generator)
        print(answer)
    print(f"\n(used_real_llm={used_llm})")

    section("⏱ METRICS")
    print(f"Execution path    : {result.execution_path}")
    print(f"Planner source    : {result.metadata.get('planner_source')}")
    print(f"SQL latency       : {result.sql_latency_ms:.2f} ms")
    print(f"Semantic latency  : {result.semantic_latency_ms:.2f} ms")
    print(f"Total latency     : {result.total_latency_ms:.2f} ms")
    print(f"Confidence        : {result.confidence}")
    print(f"Join key used     : {result.metadata.get('join_key_used')}")
    print(f"Retrieval depth   : {result.metadata.get('retrieval_depth')}")
    print(f"Cache hit         : {result.metadata.get('cache_hit', False)}")

    return {
        "query": query, "success": result.success, "path": result.execution_path,
        "confidence": result.confidence, "total_latency_ms": result.total_latency_ms,
        "facts": len(result.structured_facts), "chunks": len(result.semantic_chunks),
        "used_real_llm": used_llm,
    }


# ============================================================
# RUN SUMMARY
# ============================================================

def print_summary(all_metrics: List[Dict[str, Any]]) -> None:
    if not all_metrics:
        return
    section("📈 RUN SUMMARY (this session)")
    header = f"{'#':<3}{'Path':<15}{'Facts':>7}{'Chunks':>8}{'Conf':>7}{'LLM':>6}{'TotalMs':>10}"
    print(header)
    print(_hr("-", len(header)))
    for i, m in enumerate(all_metrics, start=1):
        print(
            f"{i:<3}{m['path']:<15}{m['facts']:>7}{m['chunks']:>8}"
            f"{m['confidence']:>7.3f}{str(m['used_real_llm']):>6}{m['total_latency_ms']:>10.2f}"
        )
    avg_latency = round(statistics.fmean(m["total_latency_ms"] for m in all_metrics), 3)
    avg_conf = round(statistics.fmean(m["confidence"] for m in all_metrics), 3)
    print(_hr("-", len(header)))
    print(f"Average total latency : {avg_latency} ms")
    print(f"Average confidence    : {avg_conf}")


# ============================================================
# MAIN
# ============================================================

async def main_async() -> None:
    print(_hr("="))
    print("HYBRID RAG PIPELINE -- powered by hybrid_agent.HybridAgent")
    print(f"DuckDB file     : {DUCKDB_FILE}")
    print(f"Qdrant          : {QDRANT_HOST}:{QDRANT_PORT} / collection='{QDRANT_COLLECTION}'")
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Execution mode  : {HYBRID_MODE}")
    print(_hr("="))

    _preview_duckdb_schema(DUCKDB_FILE)

    section("🔧 COMPONENT INITIALIZATION")
    llm_cfg = _build_llm_config()
    qdrant_cfg = _build_qdrant_config()
    embed_cfg = _build_embed_config()

    config = HybridConfig()
    config.default_execution_profile = HYBRID_MODE if HYBRID_MODE in config.execution_profiles else "balanced"

    agent = HybridAgent(
        db_path=DUCKDB_FILE,
        llm_cfg=llm_cfg,
        qdrant_cfg=qdrant_cfg,
        embed_cfg=embed_cfg,
        context_token_budget=CONTEXT_TOKEN_BUDGET,
        branch_timeout=BRANCH_TIMEOUT,
        mode=config.default_execution_profile,
        enable_cache=ENABLE_CACHE,
        cache_ttl=CACHE_TTL,
        config=config,
    )
    print(f"HybridAgent initialized (mode={config.default_execution_profile}, "
          f"planner={'llm' if llm_cfg is not None else 'heuristic'}, cache={'on' if ENABLE_CACHE else 'off'}).")

    real_answer_generator = None
    if REAL_ANSWER_GENERATOR_AVAILABLE:
        try:
            real_answer_generator = _RealAnswerGenerator()  # type: ignore
            print("AnswerGenerator : REAL (LLM-backed).")
        except Exception as exc:
            LOGGER.warning("Real AnswerGenerator failed to initialize (%s); using data-grounded fallback.", exc)
    else:
        print("AnswerGenerator : data-grounded fallback (assembled from HybridResult sections).")

    print(_hr("="))
    print("SETUP COMPLETE. Type any question below -- structured, semantic, or hybrid.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop. Type 'summary' for a run summary so far.")
    print(_hr("="))

    all_metrics: List[Dict[str, Any]] = []
    try:
        while True:
            try:
                query = input("\nEnter your query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            if query.lower() == "summary":
                print_summary(all_metrics)
                continue

            try:
                m = await run_query(agent, query, real_answer_generator)
                all_metrics.append(m)
            except Exception:
                LOGGER.exception("Query failed; you can try another one.")
    finally:
        print_summary(all_metrics)
        await agent.close()

    print(_hr("="))
    print("PIPELINE SESSION COMPLETE -- HybridAgent exercised against real DuckDB + real Qdrant.")
    print(_hr("="))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    try:
        main()
    except HarnessConfigurationError as exc:
        LOGGER.error("Harness cannot run: %s", exc)
        sys.exit(1)
    except Exception:
        LOGGER.exception("Pipeline harness encountered an unrecoverable error.")
        raise
    sys.exit(0)