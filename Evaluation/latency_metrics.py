#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
latency_metrics.py

Performance and latency evaluation for Hybrid RAG pipeline.
Measures timing across all stages: intent routing, retrieval planning,
DuckDB, Qdrant, compression, agent processing, prompt construction,
LLM generation, and post-processing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

# Try to import psutil (required)
try:
    import psutil
except ImportError:
    raise ImportError("psutil is required. Install with: pip install psutil")

# Optional GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# -----------------------------------------------------------------------------
# Project-specific imports (RAG pipeline components)
# -----------------------------------------------------------------------------
try:
    from agents.intent_router import IntentRouter
    from agents.retrieval_plan_builder import RetrievalPlanBuilder
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.context_compressor import ContextCompressor
    from agents.structured_agent import StructuredAgent
    from agents.semantic_agent import SemanticAgent
    from agents.hybrid_agent import HybridAgent
    from llm.qwen_client import QwenClient
    from llm.answer_generator import AnswerGenerator
except ImportError as e:
    raise ImportError(
        "RAG pipeline components not found. Please ensure the project structure "
        "is correct and all required modules are available."
    ) from e

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "latency_metrics.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class Stage(Enum):
    """Enumeration of all pipeline stages."""
    INTENT_ROUTER = "intent_router"
    RETRIEVAL_PLAN = "retrieval_plan"
    DUCKDB_RETRIEVAL = "duckdb_retrieval"
    QDRANT_RETRIEVAL = "qdrant_retrieval"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    CONTEXT_COMPRESSION = "context_compression"
    STRUCTURED_AGENT = "structured_agent"
    SEMANTIC_AGENT = "semantic_agent"
    HYBRID_AGENT = "hybrid_agent"
    PROMPT_CONSTRUCTION = "prompt_construction"
    LLM_GENERATION = "llm_generation"
    POSTPROCESSING = "postprocessing"
    TOTAL = "total"


class QueryType(Enum):
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class StageLatency:
    """Latency measurement for a single pipeline stage."""

    stage: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class QuestionLatencyResult:
    """Complete latency result for one question."""

    question_id: str
    question: str
    query_type: str
    latencies: List[StageLatency] = field(default_factory=list)
    total_latency_ms: float = 0.0
    token_count: int = 0
    tokens_per_second: float = 0.0
    throughput: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "query_type": self.query_type,
            "latencies": [l.to_dict() for l in self.latencies],
            "total_latency_ms": self.total_latency_ms,
            "token_count": self.token_count,
            "tokens_per_second": self.tokens_per_second,
            "throughput": self.throughput,
            "timestamp": self.timestamp,
        }


@dataclass
class LatencyStatistics:
    """Statistical summary of latency measurements."""

    mean: float
    median: float
    min: float
    max: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass
class SystemSnapshot:
    """Snapshot of system resource usage."""

    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_percent: float
    peak_memory_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_available_mb: float = 0.0
    gpu_memory_percent: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_available_mb": self.memory_available_mb,
            "memory_percent": self.memory_percent,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_available_mb": self.gpu_memory_available_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
        }


# -----------------------------------------------------------------------------
# Stage Timer (Context Manager)
# -----------------------------------------------------------------------------
class StageTimer:
    """
    Context manager for measuring stage latencies.
    Supports nesting and automatic metadata collection.
    """

    def __init__(self):
        self._stages: List[StageLatency] = []
        self._start_stack: List[Tuple[str, float]] = []  # (stage_name, start_time)

    @contextmanager
    def measure(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to measure the duration of a stage.
        Usage:
            with timer.measure("duckdb", {"query": sql}):
                # do work
        """
        start_time = time.perf_counter()
        self._start_stack.append((stage_name, start_time))
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000.0
            self._start_stack.pop()
            # Ensure duration is non-negative
            duration_ms = max(0.0, duration_ms)
            stage_latency = StageLatency(
                stage=stage_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                metadata=metadata or {},
            )
            self._stages.append(stage_latency)

    def get_latencies(self) -> List[StageLatency]:
        """Return all measured stage latencies."""
        return self._stages.copy()

    def reset(self) -> None:
        """Clear all stored latencies."""
        self._stages.clear()
        self._start_stack.clear()

    def get_total_duration(self) -> float:
        """Calculate total duration from the sum of all stage durations."""
        return sum(s.duration_ms for s in self._stages)


# -----------------------------------------------------------------------------
# Pipeline Profiler
# -----------------------------------------------------------------------------
class PipelineProfiler:
    """
    Instruments the RAG pipeline to measure each stage.
    Uses the actual pipeline components and wraps their calls.
    """

    def __init__(self):
        # Initialize all pipeline components with error handling
        try:
            self.intent_router = IntentRouter()
            self.retrieval_planner = RetrievalPlanBuilder()
            self.hybrid_retriever = HybridRetriever()
            self.context_compressor = ContextCompressor()
            self.structured_agent = StructuredAgent()
            self.semantic_agent = SemanticAgent()
            self.hybrid_agent = HybridAgent()
            self.answer_generator = AnswerGenerator()
            # QwenClient is used inside answer_generator, so we don't need separate instance
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAG components: {e}") from e

    def profile_question(self, question: str, query_type: str) -> Tuple[QuestionLatencyResult, StageTimer]:
        """
        Run the full pipeline and measure each stage.
        Returns the QuestionLatencyResult and the StageTimer containing all measurements.
        """
        timer = StageTimer()
        latencies = []

        question_id = f"q_{hash(question)}"  # temporary; will be replaced

        try:
            # Stage 1: Intent Router
            with timer.measure(Stage.INTENT_ROUTER.value):
                intent = self.intent_router.route(question)

            # Stage 2: Retrieval Plan
            with timer.measure(Stage.RETRIEVAL_PLAN.value):
                plan = self.retrieval_planner.build_plan(question, intent)

            # Stage 3: Retrieval (DuckDB, Qdrant, or Hybrid)
            if query_type == "structured":
                with timer.measure(Stage.DUCKDB_RETRIEVAL.value):
                    structured_results = self.hybrid_retriever.retrieve_structured(question, plan)
                retrieved_chunks = structured_results.get("chunks", [])
            elif query_type == "semantic":
                with timer.measure(Stage.QDRANT_RETRIEVAL.value):
                    semantic_results = self.hybrid_retriever.retrieve_semantic(question, plan)
                retrieved_chunks = semantic_results.get("chunks", [])
            else:  # hybrid
                with timer.measure(Stage.HYBRID_RETRIEVAL.value):
                    hybrid_results = self.hybrid_retriever.retrieve(question, plan)
                retrieved_chunks = hybrid_results.get("chunks", [])

            # Stage 4: Context Compression
            with timer.measure(Stage.CONTEXT_COMPRESSION.value):
                compressed_chunks = self.context_compressor.compress(retrieved_chunks, max_tokens=2000)

            # Stage 5: Agent processing (based on query type)
            if query_type == "structured":
                with timer.measure(Stage.STRUCTURED_AGENT.value):
                    agent_output = self.structured_agent.process(question, compressed_chunks)
            elif query_type == "semantic":
                with timer.measure(Stage.SEMANTIC_AGENT.value):
                    agent_output = self.semantic_agent.process(question, compressed_chunks)
            else:
                with timer.measure(Stage.HYBRID_AGENT.value):
                    agent_output = self.hybrid_agent.process(question, compressed_chunks)

            # Stage 6: Prompt Construction
            with timer.measure(Stage.PROMPT_CONSTRUCTION.value):
                prompt = self.answer_generator.build_prompt(question, agent_output, query_type)

            # Stage 7: LLM Generation
            with timer.measure(Stage.LLM_GENERATION.value):
                generated_answer, token_count = self.answer_generator.generate_with_stats(prompt)

            # Stage 8: Postprocessing
            with timer.measure(Stage.POSTPROCESSING.value):
                final_answer = self.answer_generator.postprocess(generated_answer)

        except Exception as e:
            logger.error(f"Error profiling question: {e}")
            # Still return what we have; final answer may be empty
            # We'll set token_count = 0 if failed
            token_count = 0
            # Re-raise if you want to propagate, but we'll handle gracefully
            # For this module, we'll continue and return partial data.

        # Compute total duration (sum of stages)
        total_ms = timer.get_total_duration()

        # Compute tokens per second
        tokens_per_sec = 0.0
        # Find LLM generation duration
        llm_duration_ms = None
        for s in timer.get_latencies():
            if s.stage == Stage.LLM_GENERATION.value:
                llm_duration_ms = s.duration_ms
                break
        if llm_duration_ms and llm_duration_ms > 0:
            tokens_per_sec = token_count / (llm_duration_ms / 1000.0)

        result = QuestionLatencyResult(
            question_id=question_id,
            question=question,
            query_type=query_type,
            latencies=timer.get_latencies(),
            total_latency_ms=total_ms,
            token_count=token_count,
            tokens_per_second=tokens_per_sec,
            throughput=0.0,  # will be computed in batch
        )

        return result, timer

    def profile_batch(self, questions: List[Dict[str, Any]], workers: int = 4) -> List[QuestionLatencyResult]:
        """
        Profile multiple questions in parallel using ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Limit workers to CPU count
        cpu_count = psutil.cpu_count() or 1
        workers = min(workers, cpu_count * 2)
        if workers < 1:
            workers = 1

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_q = {}
            for q in questions:
                question = q.get("question", "")
                query_type = q.get("query_type", "hybrid")
                future = executor.submit(self.profile_question, question, query_type)
                future_to_q[future] = q

            for future in tqdm(as_completed(future_to_q), total=len(questions), desc="Profiling"):
                try:
                    result, _ = future.result()
                    # Set proper question_id from original
                    result.question_id = str(future_to_q[future].get("id", result.question_id))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error profiling question: {e}")
        return results


# -----------------------------------------------------------------------------
# Resource Monitor
# -----------------------------------------------------------------------------
class ResourceMonitor:
    """
    Monitors system resources (CPU, memory, GPU) during evaluation.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.snapshots: List[SystemSnapshot] = []
        self._process = psutil.Process()
        self._peak_memory_mb = 0.0
        self._running = False

    def start(self) -> None:
        """Start monitoring."""
        self._running = True
        self.snapshots.clear()
        self._peak_memory_mb = 0.0

    def stop(self) -> None:
        self._running = False

    @contextmanager
    def track(self):
        """Context manager that samples resources during the block."""
        self.start()
        try:
            yield
        finally:
            self.stop()
            # Take final snapshot
            self.snapshots.append(self.snapshot())

    def snapshot(self) -> SystemSnapshot:
        """Take a snapshot of current system resources."""
        try:
            cpu = psutil.cpu_percent(interval=0.0)
        except Exception:
            cpu = 0.0

        try:
            mem = psutil.virtual_memory()
            mem_used_mb = mem.used / (1024 * 1024)
            mem_avail_mb = mem.available / (1024 * 1024)
            mem_percent = mem.percent
        except Exception:
            mem_used_mb = 0.0
            mem_avail_mb = 0.0
            mem_percent = 0.0

        try:
            proc_mem = self._process.memory_info().rss / (1024 * 1024)
            if proc_mem > self._peak_memory_mb:
                self._peak_memory_mb = proc_mem
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
            proc_mem = 0.0

        gpu_mem_used = 0.0
        gpu_mem_avail = 0.0
        gpu_mem_percent = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_mem_used = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_mem_avail = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                gpu_mem_percent = (gpu_mem_used / gpu_mem_avail) * 100 if gpu_mem_avail > 0 else 0.0
            except Exception:
                pass

        return SystemSnapshot(
            cpu_percent=cpu,
            memory_used_mb=mem_used_mb,
            memory_available_mb=mem_avail_mb,
            memory_percent=mem_percent,
            peak_memory_mb=self._peak_memory_mb,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_available_mb=gpu_mem_avail,
            gpu_memory_percent=gpu_mem_percent,
        )

    def get_peak_memory_mb(self) -> float:
        return self._peak_memory_mb

    def get_average_cpu(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.cpu_percent for s in self.snapshots])

    def get_average_memory(self) -> float:
        if not self.snapshots:
            return 0.0
        return np.mean([s.memory_percent for s in self.snapshots])


# -----------------------------------------------------------------------------
# Latency Analyzer
# -----------------------------------------------------------------------------
class LatencyAnalyzer:
    """
    Computes statistical metrics from latency results.
    """

    @staticmethod
    def compute_statistics(values: List[float]) -> LatencyStatistics:
        if not values:
            return LatencyStatistics(mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0,
                                     p50=0.0, p90=0.0, p95=0.0, p99=0.0)
        arr = np.array(values)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        std = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        return LatencyStatistics(
            mean=mean,
            median=median,
            min=min_val,
            max=max_val,
            std=std,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
        )

    @staticmethod
    def compute_throughput(total_questions: int, total_time_sec: float) -> float:
        return total_questions / total_time_sec if total_time_sec > 0 else 0.0

    @staticmethod
    def compute_tokens_per_second(total_tokens: int, total_generation_time_sec: float) -> float:
        return total_tokens / total_generation_time_sec if total_generation_time_sec > 0 else 0.0


# -----------------------------------------------------------------------------
# LatencyMetrics (Main Orchestrator)
# -----------------------------------------------------------------------------
class LatencyMetrics:
    """
    Main orchestrator for latency evaluation.
    Loads benchmark, runs pipeline, profiles stages, computes statistics,
    and generates reports.
    """

    def __init__(
        self,
        benchmark_path: Path,
        output_dir: Path,
        parallel: bool = True,
        workers: int = 4,
        sample_size: Optional[int] = None,
        verbose: bool = False,
    ):
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        self.parallel = parallel
        self.workers = workers
        self.sample_size = sample_size
        self.verbose = verbose

        self.profiler = PipelineProfiler()
        self.resource_monitor = ResourceMonitor()
        self.analyzer = LatencyAnalyzer()

        self.results: List[QuestionLatencyResult] = []
        self.summary: Dict[str, Any] = {}
        self.system_snapshots: List[SystemSnapshot] = []

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load benchmark questions from JSON."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")

        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Benchmark file must contain a list of questions.")

        logger.info(f"Loaded {len(data)} benchmark questions from {self.benchmark_path}")

        if self.sample_size is not None and self.sample_size < len(data):
            import random
            random.seed(42)
            data = random.sample(data, self.sample_size)
            logger.info(f"Sampled {len(data)} questions for evaluation.")

        return data

    def evaluate_question(self, question_data: Dict[str, Any]) -> QuestionLatencyResult:
        """Profile a single question."""
        question = question_data.get("question", "")
        query_type = question_data.get("query_type", "hybrid")
        result, timer = self.profiler.profile_question(question, query_type)
        result.question_id = str(question_data.get("id", result.question_id))
        return result

    def evaluate_batch(self, questions: List[Dict[str, Any]]) -> List[QuestionLatencyResult]:
        """Evaluate multiple questions, optionally in parallel."""
        if self.parallel and len(questions) > 1:
            logger.info(f"Profiling {len(questions)} questions in parallel with {self.workers} workers.")
            results = self.profiler.profile_batch(questions, workers=self.workers)
        else:
            logger.info("Profiling questions sequentially.")
            results = []
            for q in tqdm(questions, desc="Profiling"):
                try:
                    result = self.evaluate_question(q)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error profiling question {q.get('id', 'unknown')}: {e}")
        return results

    def run(self) -> None:
        """Run the full latency evaluation pipeline."""
        start_time = time.perf_counter()
        logger.info("Starting latency evaluation.")

        # Load benchmark
        questions = self.load_benchmark()
        if not questions:
            logger.warning("No questions to evaluate.")
            return

        # Start resource monitoring
        with self.resource_monitor.track():
            self.results = self.evaluate_batch(questions)

        total_time_sec = time.perf_counter() - start_time
        total_questions = len(self.results)

        # Compute statistics
        total_latencies = [r.total_latency_ms for r in self.results if r.total_latency_ms >= 0]
        if not total_latencies:
            logger.warning("No valid latency results; summary will have zeros.")
        total_stats = self.analyzer.compute_statistics(total_latencies)

        # Stage-specific latencies
        stage_latencies: Dict[str, List[float]] = {}
        for result in self.results:
            for lat in result.latencies:
                if lat.duration_ms >= 0:
                    stage_latencies.setdefault(lat.stage, []).append(lat.duration_ms)

        stage_stats = {}
        for stage, vals in stage_latencies.items():
            stage_stats[stage] = self.analyzer.compute_statistics(vals).to_dict()

        # Throughput
        throughput = self.analyzer.compute_throughput(total_questions, total_time_sec)

        # Tokens per second (total tokens / total LLM generation time)
        total_tokens = sum(r.token_count for r in self.results)
        total_llm_time_sec = 0.0
        for result in self.results:
            for lat in result.latencies:
                if lat.stage == Stage.LLM_GENERATION.value:
                    total_llm_time_sec += lat.duration_ms / 1000.0
        tokens_per_sec = self.analyzer.compute_tokens_per_second(total_tokens, total_llm_time_sec)

        # Resource summary
        peak_memory_mb = self.resource_monitor.get_peak_memory_mb()
        avg_cpu = self.resource_monitor.get_average_cpu()
        avg_mem = self.resource_monitor.get_average_memory()

        # Build summary
        self.summary = {
            "total_questions": total_questions,
            "total_time_sec": total_time_sec,
            "total_latency_ms_stats": total_stats.to_dict(),
            "stage_stats": stage_stats,
            "throughput_qps": throughput,
            "tokens_per_second": tokens_per_sec,
            "total_tokens": total_tokens,
            "peak_memory_mb": peak_memory_mb,
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_mem,
            "timestamp": datetime.now().isoformat(),
        }

        # Save reports
        self.save_report()

        # Print summary
        self.print_summary()

    def save_report(self) -> None:
        """Save latency reports to JSON and CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full report (including all results)
        report = {
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }
        report_path = self.output_dir / "latency_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Full latency report saved to {report_path}")

        # Summary only
        summary_path = self.output_dir / "latency_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Latency summary saved to {summary_path}")

        # CSV export
        csv_path = self.output_dir / "latency_report.csv"
        self.export_csv(csv_path)
        logger.info(f"CSV report saved to {csv_path}")

    def export_csv(self, csv_path: Path) -> None:
        """Export per-question latency results to CSV."""
        if not self.results:
            return
        rows = []
        for r in self.results:
            row = {
                "question_id": r.question_id,
                "question": r.question,
                "query_type": r.query_type,
                "total_latency_ms": r.total_latency_ms,
                "token_count": r.token_count,
                "tokens_per_second": r.tokens_per_second,
            }
            for lat in r.latencies:
                row[f"stage_{lat.stage}_ms"] = lat.duration_ms
            rows.append(row)

        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                for field in fieldnames:
                    if field not in row:
                        row[field] = ""
                writer.writerow(row)

    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        s = self.summary
        logger.info("=" * 60)
        logger.info("LATENCY EVALUATION SUMMARY")
        logger.info(f"Questions evaluated: {s['total_questions']}")
        logger.info(f"Total evaluation time: {s['total_time_sec']:.2f}s")
        logger.info(f"Throughput: {s['throughput_qps']:.3f} questions/sec")
        logger.info(f"Tokens per second: {s['tokens_per_second']:.1f} tokens/sec")
        logger.info(f"Total tokens generated: {s['total_tokens']}")
        logger.info(f"Peak memory usage: {s['peak_memory_mb']:.1f} MB")
        logger.info(f"Average CPU usage: {s['average_cpu_percent']:.1f}%")
        logger.info(f"Average memory usage: {s['average_memory_percent']:.1f}%")
        logger.info("\nTotal Latency Statistics:")
        stats = s['total_latency_ms_stats']
        logger.info(f"  Mean:   {stats['mean']:.2f} ms")
        logger.info(f"  Median: {stats['median']:.2f} ms")
        logger.info(f"  P95:    {stats['p95']:.2f} ms")
        logger.info(f"  P99:    {stats['p99']:.2f} ms")
        logger.info(f"  Min:    {stats['min']:.2f} ms")
        logger.info(f"  Max:    {stats['max']:.2f} ms")

        stage_stats = s.get('stage_stats', {})
        if stage_stats:
            avg_stage_times = {}
            for stage, stats in stage_stats.items():
                avg_stage_times[stage] = stats['mean']
            if avg_stage_times:
                slowest = max(avg_stage_times, key=avg_stage_times.get)
                logger.info(f"\nSlowest stage: {slowest} ({avg_stage_times[slowest]:.2f} ms average)")
        logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latency of RAG pipeline.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("evaluation/benchmark_questions.json"),
        help="Path to benchmark_questions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports"),
        help="Directory to save reports",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel evaluation (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (capped by CPU count)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for evaluation (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    evaluator = LatencyMetrics(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        parallel=args.parallel,
        workers=args.workers,
        sample_size=args.sample_size,
        verbose=args.verbose,
    )

    try:
        evaluator.run()
    except Exception as e:
        logger.error(f"Latency evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()