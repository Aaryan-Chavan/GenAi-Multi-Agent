#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation_runner.py

Orchestrates the complete RAG evaluation pipeline:
- Benchmark generation
- Accuracy evaluation
- Latency evaluation
- LLM judge evaluation
- Report aggregation and final summary

Supports parallel execution, resume, checkpointing, and dry-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Import existing evaluation modules
# -----------------------------------------------------------------------------
try:
    from evaluation.benchmark_generator import BenchmarkGenerator, BenchmarkConfig
    from evaluation.accuracy_metrics import AccuracyMetrics
    from evaluation.latency_metrics import LatencyMetrics
    from evaluation.llm_judge import LLMJudge
except ImportError as e:
    raise ImportError(
        "Required evaluation modules not found. Please ensure the project structure "
        "is correct and all modules are available."
    ) from e

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "evaluation_runner.log"

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
class EvaluationState(Enum):
    """States of the evaluation pipeline."""
    INITIALIZED = "initialized"
    BENCHMARK_READY = "benchmark_ready"
    ACCURACY_DONE = "accuracy_done"
    LATENCY_DONE = "latency_done"
    JUDGE_DONE = "judge_done"
    COMPLETED = "completed"
    FAILED = "failed"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class EvaluationConfig:
    """Configuration for the evaluation runner."""

    # Paths
    benchmark_path: Path = Path("evaluation/benchmark_questions.json")
    accuracy_output: Path = Path("evaluation/reports/accuracy_report.json")
    latency_output: Path = Path("evaluation/reports/latency_report.json")
    judge_output: Path = Path("evaluation/reports/llm_judge_report.json")
    final_output: Path = Path("evaluation/reports/final_report.json")
    summary_output: Path = Path("evaluation/reports/evaluation_summary.json")
    dashboard_output: Path = Path("evaluation/reports/evaluation_dashboard.csv")
    state_file: Path = Path("evaluation/.eval_state.json")

    # Execution flags
    generate_benchmark: bool = True
    enable_accuracy: bool = True
    enable_latency: bool = True
    enable_judge: bool = True
    parallel: bool = True
    workers: int = 4
    sample_size: Optional[int] = None
    seed: int = 42
    resume: bool = False
    dry_run: bool = False
    verbose: bool = False

    # Scoring weights
    accuracy_weight: float = 0.50
    latency_weight: float = 0.20
    judge_weight: float = 0.30

    # Additional
    force_overwrite: bool = False


@dataclass
class EvaluationResult:
    """Complete result of the evaluation run."""

    questions_evaluated: int
    accuracy_score: float
    latency_score: float
    judge_score: float
    hallucination_rate: float
    average_latency_ms: float
    overall_score: float
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions_evaluated": self.questions_evaluated,
            "accuracy_score": self.accuracy_score,
            "latency_score": self.latency_score,
            "judge_score": self.judge_score,
            "hallucination_rate": self.hallucination_rate,
            "average_latency_ms": self.average_latency_ms,
            "overall_score": self.overall_score,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Report Aggregator
# -----------------------------------------------------------------------------
class ReportAggregator:
    """
    Loads and aggregates reports from accuracy, latency, and judge evaluations.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.accuracy_report: Dict[str, Any] = {}
        self.latency_report: Dict[str, Any] = {}
        self.judge_report: Dict[str, Any] = {}

    def load_all(self) -> None:
        """Load all available reports."""
        if self.config.enable_accuracy:
            self.accuracy_report = self._load_report(self.config.accuracy_output)
        if self.config.enable_latency:
            self.latency_report = self._load_report(self.config.latency_output)
        if self.config.enable_judge:
            self.judge_report = self._load_report(self.config.judge_output)

    def _load_report(self, path: Path) -> Dict[str, Any]:
        """Load a JSON report."""
        if not path.exists():
            logger.warning(f"Report not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {path}: {e}")
            return {}

    def aggregate(self) -> EvaluationResult:
        """
        Compute aggregate metrics from loaded reports.
        Returns an EvaluationResult.
        """
        # Extract key metrics
        accuracy_score = 0.0
        hallucination_rate = 0.0
        latency_score = 0.0
        avg_latency_ms = 0.0
        judge_score = 0.0
        questions = 0

        # Accuracy report
        if self.accuracy_report:
            summary = self.accuracy_report.get("summary", {})
            questions = summary.get("total", 0)
            accuracy_score = summary.get("average_overall_score", 0.0) * 100  # scale to 0-100
        # Latency report
        if self.latency_report:
            summary = self.latency_report.get("summary", {})
            avg_latency_ms = summary.get("total_latency_ms_stats", {}).get("mean", 0.0)
            # Compute latency score: assume 100ms is ideal, 1000ms is bad; map to 0-100
            if avg_latency_ms > 0:
                latency_score = max(0.0, 100.0 - (avg_latency_ms - 100.0) / 10.0)
                latency_score = min(100.0, latency_score)
            else:
                latency_score = 0.0
            questions = summary.get("total_questions", 0)
        # Judge report
        if self.judge_report:
            summary = self.judge_report.get("summary", {})
            judge_score = summary.get("average_overall_score", 0.0) * 10  # scale to 0-100
            hallucination_rate = summary.get("hallucination_rate", 0.0)
            questions = summary.get("total", 0)

        # Compute overall weighted score
        overall_score = (
            self.config.accuracy_weight * accuracy_score +
            self.config.latency_weight * latency_score +
            self.config.judge_weight * judge_score
        )

        # Use the max questions count
        questions = max(questions, 0)

        return EvaluationResult(
            questions_evaluated=questions,
            accuracy_score=accuracy_score,
            latency_score=latency_score,
            judge_score=judge_score,
            hallucination_rate=hallucination_rate,
            average_latency_ms=avg_latency_ms,
            overall_score=overall_score,
            duration_seconds=0.0,  # will be filled by runner
            metadata={
                "accuracy_report": self.config.accuracy_output.name,
                "latency_report": self.config.latency_output.name,
                "judge_report": self.config.judge_output.name,
            },
        )


# -----------------------------------------------------------------------------
# Evaluation Pipeline
# -----------------------------------------------------------------------------
class EvaluationPipeline:
    """
    Orchestrates the execution of each evaluation module.
    Supports checkpointing and resume.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.state = EvaluationState.INITIALIZED
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[EvaluationResult] = None

        # Ensure output directories exist
        self.config.accuracy_output.parent.mkdir(parents=True, exist_ok=True)
        self.config.latency_output.parent.mkdir(parents=True, exist_ok=True)
        self.config.judge_output.parent.mkdir(parents=True, exist_ok=True)
        self.config.final_output.parent.mkdir(parents=True, exist_ok=True)

        # Load state if resuming
        if self.config.resume:
            self._load_state()

    def _save_state(self) -> None:
        """Save current state to file."""
        state_data = {
            "state": self.state.value,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "benchmark_path": str(self.config.benchmark_path),
                "sample_size": self.config.sample_size,
                "seed": self.config.seed,
                "parallel": self.config.parallel,
                "workers": self.config.workers,
            },
        }
        try:
            with open(self.config.state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
            logger.debug(f"State saved: {self.state.value}")
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load state from file if exists."""
        if self.config.state_file.exists():
            try:
                with open(self.config.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state_str = data.get("state", "initialized")
                self.state = EvaluationState(state_str)
                logger.info(f"Resuming from state: {self.state.value}")
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}")
                self.state = EvaluationState.INITIALIZED

    def _should_run_step(self, step_state: EvaluationState) -> bool:
        """Determine if a step should run based on current state and resume flag."""
        if self.config.dry_run:
            return False
        if not self.config.resume:
            return True
        # Resume: run if current state is before the step state
        order = {
            EvaluationState.INITIALIZED: 0,
            EvaluationState.BENCHMARK_READY: 1,
            EvaluationState.ACCURACY_DONE: 2,
            EvaluationState.LATENCY_DONE: 3,
            EvaluationState.JUDGE_DONE: 4,
            EvaluationState.COMPLETED: 5,
        }
        return order.get(self.state, 0) < order.get(step_state, 0)

    def generate_benchmark(self) -> None:
        """Run benchmark generation if needed."""
        if not self.config.generate_benchmark:
            logger.info("Skipping benchmark generation (disabled).")
            return
        if self.config.benchmark_path.exists() and not self.config.force_overwrite:
            logger.info("Benchmark already exists. Skipping generation.")
            self.state = EvaluationState.BENCHMARK_READY
            self._save_state()
            return
        if not self._should_run_step(EvaluationState.BENCHMARK_READY):
            logger.info("Skipping benchmark generation due to resume state.")
            return

        logger.info("Generating benchmark...")
        try:
            bench_config = BenchmarkConfig(
                structured_count=100,
                semantic_count=100,
                hybrid_count=100,
                random_seed=self.config.seed,
                sample_size=self.config.sample_size or 10000,
                output_path=self.config.benchmark_path,
                verbose=self.config.verbose,
            )
            generator = BenchmarkGenerator(bench_config)
            generator.run()
            self.state = EvaluationState.BENCHMARK_READY
            self._save_state()
            logger.info("Benchmark generation completed.")
        except Exception as e:
            logger.error(f"Benchmark generation failed: {e}")
            self.state = EvaluationState.FAILED
            self._save_state()
            raise

    def run_accuracy(self) -> None:
        """Run accuracy evaluation."""
        if not self.config.enable_accuracy:
            logger.info("Skipping accuracy evaluation (disabled).")
            return
        if self.config.accuracy_output.exists() and not self.config.force_overwrite:
            logger.info("Accuracy report already exists. Skipping.")
            self.state = EvaluationState.ACCURACY_DONE
            self._save_state()
            return
        if not self._should_run_step(EvaluationState.ACCURACY_DONE):
            logger.info("Skipping accuracy evaluation due to resume state.")
            return

        logger.info("Running accuracy evaluation...")
        try:
            evaluator = AccuracyMetrics(
                benchmark_path=self.config.benchmark_path,
                output_dir=self.config.accuracy_output.parent,
                parallel=self.config.parallel,
                max_workers=self.config.workers,
                sample_size=self.config.sample_size,
                verbose=self.config.verbose,
            )
            evaluator.run()
            self.state = EvaluationState.ACCURACY_DONE
            self._save_state()
            logger.info("Accuracy evaluation completed.")
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            self.state = EvaluationState.FAILED
            self._save_state()
            raise

    def run_latency(self) -> None:
        """Run latency evaluation."""
        if not self.config.enable_latency:
            logger.info("Skipping latency evaluation (disabled).")
            return
        if self.config.latency_output.exists() and not self.config.force_overwrite:
            logger.info("Latency report already exists. Skipping.")
            self.state = EvaluationState.LATENCY_DONE
            self._save_state()
            return
        if not self._should_run_step(EvaluationState.LATENCY_DONE):
            logger.info("Skipping latency evaluation due to resume state.")
            return

        logger.info("Running latency evaluation...")
        try:
            evaluator = LatencyMetrics(
                benchmark_path=self.config.benchmark_path,
                output_dir=self.config.latency_output.parent,
                parallel=self.config.parallel,
                workers=self.config.workers,
                sample_size=self.config.sample_size,
                verbose=self.config.verbose,
            )
            evaluator.run()
            self.state = EvaluationState.LATENCY_DONE
            self._save_state()
            logger.info("Latency evaluation completed.")
        except Exception as e:
            logger.error(f"Latency evaluation failed: {e}")
            self.state = EvaluationState.FAILED
            self._save_state()
            raise

    def run_judge(self) -> None:
        """Run LLM judge evaluation."""
        if not self.config.enable_judge:
            logger.info("Skipping LLM judge evaluation (disabled).")
            return
        if self.config.judge_output.exists() and not self.config.force_overwrite:
            logger.info("Judge report already exists. Skipping.")
            self.state = EvaluationState.JUDGE_DONE
            self._save_state()
            return
        if not self._should_run_step(EvaluationState.JUDGE_DONE):
            logger.info("Skipping LLM judge evaluation due to resume state.")
            return

        logger.info("Running LLM judge evaluation...")
        try:
            judge = LLMJudge(
                benchmark_path=self.config.benchmark_path,
                output_dir=self.config.judge_output.parent,
                parallel=self.config.parallel,
                workers=self.config.workers,
                sample_size=self.config.sample_size,
                verbose=self.config.verbose,
            )
            judge.run()
            self.state = EvaluationState.JUDGE_DONE
            self._save_state()
            logger.info("LLM judge evaluation completed.")
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            self.state = EvaluationState.FAILED
            self._save_state()
            raise

    def run(self) -> EvaluationResult:
        """
        Run the complete pipeline.
        Returns the final aggregated EvaluationResult.
        """
        if self.config.dry_run:
            logger.info("Dry run mode: skipping actual execution.")
            # Return a dummy result
            return EvaluationResult(
                questions_evaluated=0,
                accuracy_score=0.0,
                latency_score=0.0,
                judge_score=0.0,
                hallucination_rate=0.0,
                average_latency_ms=0.0,
                overall_score=0.0,
                duration_seconds=0.0,
            )

        self.start_time = time.perf_counter()
        logger.info("Starting evaluation pipeline.")

        try:
            # Step 1: Generate benchmark
            self.generate_benchmark()

            # Step 2: Accuracy
            self.run_accuracy()

            # Step 3: Latency
            self.run_latency()

            # Step 4: Judge
            self.run_judge()

            # Step 5: Aggregate reports
            logger.info("Aggregating reports...")
            aggregator = ReportAggregator(self.config)
            aggregator.load_all()
            self.result = aggregator.aggregate()
            self.result.duration_seconds = time.perf_counter() - self.start_time

            self.state = EvaluationState.COMPLETED
            self._save_state()
            logger.info("Pipeline completed successfully.")
            return self.result

        except Exception as e:
            self.state = EvaluationState.FAILED
            self._save_state()
            logger.error(f"Pipeline failed: {e}")
            raise


# -----------------------------------------------------------------------------
# Evaluation Runner (Main Orchestrator)
# -----------------------------------------------------------------------------
class EvaluationRunner:
    """
    Main orchestrator that sets up configuration, runs the pipeline,
    and generates the final report and dashboard.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.pipeline = EvaluationPipeline(config)
        self.result: Optional[EvaluationResult] = None

    def run(self) -> None:
        """Execute the full evaluation and generate outputs."""
        # Run pipeline
        self.result = self.pipeline.run()

        # Generate final reports
        self._generate_final_report()
        self._generate_summary()
        self._generate_dashboard()

        # Print summary to console
        self._print_summary()

    def _generate_final_report(self) -> None:
        """Write the final aggregated report."""
        if self.result is None:
            logger.error("No result available; cannot generate final report.")
            return
        report = {
            "summary": self.result.to_dict(),
            "details": {
                "accuracy_report": str(self.config.accuracy_output),
                "latency_report": str(self.config.latency_output),
                "judge_report": str(self.config.judge_output),
                "config": {
                    "sample_size": self.config.sample_size,
                    "seed": self.config.seed,
                    "parallel": self.config.parallel,
                    "workers": self.config.workers,
                },
            },
        }
        try:
            with open(self.config.final_output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Final report saved to {self.config.final_output}")
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")

    def _generate_summary(self) -> None:
        """Write a concise summary JSON."""
        if self.result is None:
            logger.error("No result available; cannot generate summary.")
            return
        try:
            with open(self.config.summary_output, "w", encoding="utf-8") as f:
                json.dump(self.result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Summary saved to {self.config.summary_output}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def _generate_dashboard(self) -> None:
        """Export a CSV dashboard with key metrics."""
        if self.result is None:
            logger.error("No result available; cannot generate dashboard.")
            return
        data = {
            "metric": [
                "questions_evaluated",
                "overall_score",
                "accuracy_score",
                "latency_score",
                "judge_score",
                "hallucination_rate",
                "average_latency_ms",
                "duration_seconds",
            ],
            "value": [
                self.result.questions_evaluated,
                self.result.overall_score,
                self.result.accuracy_score,
                self.result.latency_score,
                self.result.judge_score,
                self.result.hallucination_rate,
                self.result.average_latency_ms,
                self.result.duration_seconds,
            ],
        }
        try:
            df = pd.DataFrame(data)
            df.to_csv(self.config.dashboard_output, index=False)
            logger.info(f"Dashboard saved to {self.config.dashboard_output}")
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")

    def _print_summary(self) -> None:
        """Print human-readable summary to console."""
        if self.result is None:
            logger.warning("No result to display.")
            return
        r = self.result
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Questions evaluated: {r.questions_evaluated}")
        logger.info(f"Overall score: {r.overall_score:.2f}%")
        logger.info(f"Accuracy score: {r.accuracy_score:.2f}%")
        logger.info(f"Latency score: {r.latency_score:.2f}%")
        logger.info(f"Judge score: {r.judge_score:.2f}%")
        logger.info(f"Hallucination rate: {r.hallucination_rate:.3f}")
        logger.info(f"Average latency: {r.average_latency_ms:.2f} ms")
        logger.info(f"Total duration: {r.duration_seconds:.2f} s")
        logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Complete RAG evaluation orchestrator.")
    parser.add_argument(
        "--generate-benchmark",
        action="store_true",
        default=True,
        help="Generate benchmark questions",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        default=True,
        help="Run accuracy evaluation",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        default=True,
        help="Run latency evaluation",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=True,
        help="Run LLM judge evaluation",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of questions to evaluate (sample)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate run without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    config = EvaluationConfig(
        generate_benchmark=args.generate_benchmark,
        enable_accuracy=args.accuracy,
        enable_latency=args.latency,
        enable_judge=args.judge,
        parallel=args.parallel,
        workers=args.workers,
        sample_size=args.sample_size,
        resume=args.resume,
        force_overwrite=args.force,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    runner = EvaluationRunner(config)
    try:
        runner.run()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()