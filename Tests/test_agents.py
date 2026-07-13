from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

logger = logging.getLogger("test_pipeline")

# ---------------------------------------------------------
# Formatter
# ---------------------------------------------------------
from core.prompt_builder import format_structured_facts

# ---------------------------------------------------------
# Agents
# ---------------------------------------------------------
from Agents.intent_router import IntentRouter
from Agents.intent_router import IntentResult as RouterIntentResult

from Agents.structured_agent import StructuredAgent
from Agents.semantic_agent import SemanticAgent
from Agents.hybrid_agent import HybridAgent, compress_context

from Agents.retrieval_plan_builder import (
    RetrievalPlanBuilder,
    IntentResult as PlanIntentResult,
    RetrievalPlan,
)

# ---------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------
_DB_PATH = os.environ.get(
    "ANALYTICS_DB_PATH",
    r"F:\PROJECT\Database\Duckdb\analytics.duckdb",
)

if not os.path.exists(_DB_PATH) and _DB_PATH != ":memory:":
    logger.warning("DB not found → fallback to :memory: %s", _DB_PATH)
    _DB_PATH = ":memory:"


_QUERY_TIMEOUT = 120.0


# =========================================================
# ADAPTER
# =========================================================

def to_plan_intent(
    router_result: RouterIntentResult,
    normalized_query: str,
    entities: Optional[Dict[str, Any]] = None,
) -> PlanIntentResult:

    return PlanIntentResult(
        query_type=router_result.query_type,
        needs_sql=router_result.needs_sql,
        needs_vector=router_result.needs_vector,
        needs_time_series=router_result.needs_time_series,
        needs_complaint_focus=router_result.needs_complaint_focus,
        normalized_query=normalized_query,
        entities=entities or {},
    )


# =========================================================
# CONTEXT COMPRESSOR
# =========================================================

class ContextCompressor:

    def compress(
        self,
        query: str,
        branch: str,
        agent_result: Any,
        plan: RetrievalPlan,
    ) -> Dict[str, Any]:

        # ---------------- HYBRID ----------------
        if branch == "hybrid":
            return {
                "context": getattr(agent_result, "compressed_context", ""),
                "structured_summary": getattr(agent_result, "structured_summary", ""),
                "complaint_summary": getattr(agent_result, "complaint_summary", ""),
                "key_snippets": getattr(agent_result, "key_snippets", []),
                "trend_observations": getattr(agent_result, "trend_observations", []),
            }

        # ---------------- SQL ----------------
        if branch == "sql":
            rows = []
            if agent_result and getattr(agent_result, "success", False):
                rows = getattr(agent_result, "rows", [])

            structured_block = format_structured_facts(rows)

            return {
                "context": structured_block,
                "structured_summary": structured_block,
                "complaint_summary": "",
                "key_snippets": [],
                "trend_observations": [],
            }

        # ---------------- QDRANT ----------------
        chunks = []
        if agent_result and getattr(agent_result, "success", False):
            chunks = getattr(agent_result, "chunks", [])

        cf = (
            plan.vector_task.complaint_focus
            if plan.vector_task else False
        )

        return compress_context(
            facts=[],
            chunks=chunks,
            query=query,
            needs_time_series=False,
            needs_complaint_focus=cf,
        )


# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(query: str, context: Dict[str, Any]) -> str:
    ctx_text = context.get("context", "")

    return (
        "You are a product analytics assistant.\n"
        "Answer ONLY using the context below.\n\n"
        "## Question\n"
        f"{query}\n\n"
        "## Context\n"
        f"{ctx_text}\n\n"
        "## Answer"
    )


# =========================================================
# SAFE FILTER MERGE
# =========================================================

def _safe_merge_filters(sql_filters, qdrant_filters):
    merged = {}
    if sql_filters:
        merged.update(sql_filters)
    if qdrant_filters:
        merged.update(qdrant_filters)
    return merged


# =========================================================
# PIPELINE RUNNER
# =========================================================

async def run_query(
    query: str,
    router: IntentRouter,
    sql_agent: StructuredAgent,
    semantic_agent: SemanticAgent,
    hybrid_agent: HybridAgent,
    planner: RetrievalPlanBuilder,
    compressor: ContextCompressor,
):

    print("\n" + "=" * 80)
    print("USER QUERY")
    print(query)
    print("=" * 80)

    route = router.route(query)

    print("\nIntent:")
    print(route)

    plan = planner.build(to_plan_intent(route, query, {}))

    print("\nPlan:")
    print(RetrievalPlanBuilder.plan_summary(plan))

    # ---------------- SQL ----------------
    if plan.branch == "sql":

        sql_filters = plan.sql_task.filters if plan.sql_task else {}
        ts = plan.sql_task.time_series if plan.sql_task else False

        retrieved = await asyncio.wait_for(
            sql_agent.run(
                query=query,
                filters=sql_filters,
                needs_time_series=ts,
            ),
            timeout=_QUERY_TIMEOUT,
        )

    # ---------------- QDRANT ----------------
    elif plan.branch == "qdrant":

        eq = plan.vector_task.enhanced_query if plan.vector_task else query
        qf = plan.vector_task.qdrant_filters if plan.vector_task else {}
        cf = plan.vector_task.complaint_focus if plan.vector_task else False

        retrieved = await asyncio.wait_for(
            semantic_agent.run(
                query=eq,
                filters=qf,
                needs_complaint_focus=cf,
            ),
            timeout=_QUERY_TIMEOUT,
        )

    # ---------------- HYBRID ----------------
    else:

        sql_filters = plan.sql_task.filters if plan.sql_task else {}
        qdrant_filters = plan.vector_task.qdrant_filters if plan.vector_task else {}

        ts = plan.sql_task.time_series if plan.sql_task else False
        cf = plan.vector_task.complaint_focus if plan.vector_task else False

        retrieved = await asyncio.wait_for(
            hybrid_agent.run(
                query=query,
                filters=_safe_merge_filters(sql_filters, qdrant_filters),
                needs_time_series=ts,
                needs_complaint_focus=cf,
            ),
            timeout=_QUERY_TIMEOUT,
        )

    print("\nRetrieved:", type(retrieved).__name__, "| success =", retrieved.success)

    if not retrieved.success:
        print("SQL:")
        print(retrieved.sql)

        print("\nERROR:")
        print(retrieved.error)

    context = compressor.compress(query, plan.branch, retrieved, plan)

    print("\nCompressed Context Length:", len(context.get("context", "")))

    prompt = build_prompt(query, context)

    print("\nPrompt Length:", len(prompt))
    print("\nPrompt Preview:\n", prompt[:500])


# =========================================================
# MAIN
# =========================================================

async def main():

    router = IntentRouter()
    router.warmup()

    sql_agent = StructuredAgent(db_path=_DB_PATH)
    semantic_agent = SemanticAgent()
    hybrid_agent = HybridAgent(db_path=_DB_PATH)

    planner = RetrievalPlanBuilder()
    compressor = ContextCompressor()

    queries = [
        "Average rating of Samsung phones",
        "What do users think about battery life?",
        "Top rated headphones",
        "Positive reviews of Lenovo laptops",
        "Show me the monthly trend in returns over 2024",
        "What are the main complaints about delivery?",
    ]

    for q in queries:
        try:
            await run_query(
                q,
                router,
                sql_agent,
                semantic_agent,
                hybrid_agent,
                planner,
                compressor,
            )
        except Exception as e:
            logger.error("Query failed: %s | %s", q, e)

    await semantic_agent.close()
    await hybrid_agent.close()
    sql_agent.close()


if __name__ == "__main__":
    asyncio.run(main())