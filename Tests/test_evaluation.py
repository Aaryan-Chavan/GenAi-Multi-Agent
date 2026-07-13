#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_evaluation.py

Comprehensive test suite for the Hybrid RAG Evaluation Framework.
Tests benchmark generation, accuracy, latency, LLM judge, evaluation runner,
and report validation using pytest, mocking, and fixtures.

Coverage target: 90%+
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

# -----------------------------------------------------------------------------
# Import modules under test
# -----------------------------------------------------------------------------
try:
    from Evaluation.benchmark_generator import (
        BenchmarkGenerator,
        BenchmarkConfig,
        BenchmarkRecord,
        DatasetProfile,
        SchemaInspector,
        StructuredQuestionGenerator,
        SemanticQuestionGenerator,
        HybridQuestionGenerator,
    )
    from evaluation.accuracy_metrics import (
        AccuracyMetrics,
        StructuredAccuracyEvaluator,
        SemanticAccuracyEvaluator,
        RetrievalEvaluator,
        HallucinationEvaluator,
        HybridEvaluator,
        MetricResult,
        EvaluationRecord,
    )
    from evaluation.latency_metrics import (
        LatencyMetrics,
        StageTimer,
        PipelineProfiler,
        LatencyAnalyzer,
        ResourceMonitor,
        StageLatency,
        QuestionLatencyResult,
    )
    from evaluation.llm_judge import (
        LLMJudge,
        PromptBuilder,
        ResponseParser,
        AnswerJudge,
        CompositeJudge,
        JudgeScore,
        JudgementResult,
    )
    from evaluation.evaluation_runner import (
        EvaluationRunner,
        EvaluationConfig,
        EvaluationPipeline,
        ReportAggregator,
        EvaluationResult,
        EvaluationState,
    )
except ImportError as e:
    print(f"Import error: {e}. Ensure evaluation modules are in PYTHONPATH.", file=sys.stderr)
    raise

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_benchmark_data() -> list[dict]:
    """Sample benchmark questions for testing."""
    return [
        {
            "id": "1",
            "question": "What is the average rating of product X?",
            "query_type": "structured",
            "expected_answer": 4.5,
            "expected_keywords": ["rating", "average"],
            "ground_truth": {"sql": "SELECT AVG(rating) FROM products WHERE name='X'"},
        },
        {
            "id": "2",
            "question": "Summarize the pros and cons of product Y.",
            "query_type": "semantic",
            "expected_answer": "Good battery, poor display.",
            "expected_keywords": ["battery", "display", "pros", "cons"],
            "ground_truth": {},
        },
        {
            "id": "3",
            "question": "What is the highest rated product with positive reviews?",
            "query_type": "hybrid",
            "expected_answer": "Product Z",
            "expected_keywords": ["highest rated", "positive"],
            "ground_truth": {},
        },
    ]


@pytest.fixture
def sample_context() -> list[str]:
    return [
        "Product X has an average rating of 4.5 stars.",
        "Product Y has good battery life but a poor display.",
        "Product Z is the highest rated with many positive reviews.",
    ]


@pytest.fixture
def sample_answer() -> str:
    return "Product X has an average rating of 4.5."


@pytest.fixture
def sample_accuracy_report(tmp_path: Path) -> Path:
    report = {
        "summary": {
            "total": 3,
            "average_overall_score": 0.92,
            "type_counts": {"structured": 1, "semantic": 1, "hybrid": 1},
            "category_averages": {"structured": 0.9, "semantic": 0.85, "retrieval": 0.8, "hallucination": 0.9},
        },
        "results": [
            {"question_id": "1", "overall_score": 0.95},
            {"question_id": "2", "overall_score": 0.88},
            {"question_id": "3", "overall_score": 0.93},
        ],
    }
    path = tmp_path / "accuracy_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def sample_latency_report(tmp_path: Path) -> Path:
    report = {
        "summary": {
            "total_questions": 3,
            "total_latency_ms_stats": {"mean": 850, "median": 820, "p95": 1200},
            "stage_stats": {"llm_generation": {"mean": 600}},
            "throughput_qps": 2.5,
            "tokens_per_second": 30.0,
        },
        "results": [
            {"question_id": "1", "total_latency_ms": 800},
            {"question_id": "2", "total_latency_ms": 820},
            {"question_id": "3", "total_latency_ms": 900},
        ],
    }
    path = tmp_path / "latency_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def sample_judge_report(tmp_path: Path) -> Path:
    report = {
        "summary": {
            "total": 3,
            "average_overall_score": 0.88,
            "hallucination_rate": 0.05,
            "pass_rate": 0.9,
            "average_confidence": 0.85,
        },
        "results": [
            {"question_id": "1", "overall_score": 0.9},
            {"question_id": "2", "overall_score": 0.85},
            {"question_id": "3", "overall_score": 0.89},
        ],
    }
    path = tmp_path / "llm_judge_report.json"
    with open(path, "w") as f:
        json.dump(report, f)
    return path


@pytest.fixture
def temp_benchmark_file(tmp_path, sample_benchmark_data) -> Path:
    path = tmp_path / "benchmark_questions.json"
    with open(path, "w") as f:
        json.dump(sample_benchmark_data, f)
    return path


@pytest.fixture
def mock_qwen_client():
    with patch("evaluation.llm_judge.QwenClient") as mock:
        client = MagicMock()
        client.generate.return_value = json.dumps({
            "score": 8,
            "confidence": 0.9,
            "reason": "Good answer.",
            "hallucination": False,
        })
        mock.return_value = client
        yield client


@pytest.fixture
def mock_hybrid_retriever():
    with patch("evaluation.accuracy_metrics.HybridRetriever") as mock_hr, \
         patch("evaluation.latency_metrics.HybridRetriever") as mock_hr_lat:
        retriever = MagicMock()
        retriever.retrieve.return_value = {"chunks": ["context1", "context2"]}
        retriever.retrieve_structured.return_value = {"chunks": ["struct_context"]}
        retriever.retrieve_semantic.return_value = {"chunks": ["semantic_context"]}
        mock_hr.return_value = retriever
        mock_hr_lat.return_value = retriever
        yield retriever


@pytest.fixture
def mock_answer_generator():
    with patch("evaluation.accuracy_metrics.AnswerGenerator") as mock_ag_acc, \
         patch("evaluation.latency_metrics.AnswerGenerator") as mock_ag_lat:
        generator = MagicMock()
        generator.generate.return_value = "Generated answer."
        generator.generate_with_stats.return_value = ("Generated answer.", 50)
        generator.build_prompt.return_value = "prompt"
        generator.postprocess.return_value = "final answer"
        mock_ag_acc.return_value = generator
        mock_ag_lat.return_value = generator
        yield generator


@pytest.fixture
def mock_llm_judge_components():
    with patch("evaluation.llm_judge.AnswerJudge") as mock_aj:
        judge_instance = MagicMock()
        from evaluation.llm_judge import JudgeScore
        def make_score(name, score, passed=True, confidence=0.9, reason=""):
            return JudgeScore(name, score, passed, confidence, reason)

        judge_instance.score_accuracy.return_value = make_score("accuracy", 8.0)
        judge_instance.score_completeness.return_value = make_score("completeness", 7.5)
        judge_instance.score_relevance.return_value = make_score("relevance", 9.0)
        judge_instance.score_faithfulness.return_value = make_score("faithfulness", 8.5)
        judge_instance.score_reasoning.return_value = make_score("reasoning", 8.0)
        judge_instance.detect_hallucination.return_value = make_score("hallucination", 10.0, passed=True, confidence=0.95)
        mock_aj.return_value = judge_instance
        yield judge_instance


# -----------------------------------------------------------------------------
# Tests: Benchmark Generation
# -----------------------------------------------------------------------------

class TestBenchmarkGeneration:
    """Test benchmark generation module."""

    def test_benchmark_config_defaults(self):
        config = BenchmarkConfig()
        assert config.structured_count == 100
        assert config.semantic_count == 100
        assert config.hybrid_count == 100
        assert config.random_seed == 42

    def test_benchmark_generator_creates_output(self, tmp_path):
        config = BenchmarkConfig(
            structured_count=2,
            semantic_count=2,
            hybrid_count=2,
            random_seed=42,
            sample_size=10,
            output_path=tmp_path / "test_benchmark.json",
            data_dir=tmp_path,  # provide dummy data_dir
        )
        with patch("evaluation.benchmark_generator.SchemaInspector") as mock_inspector:
            # Mock dataset profile
            profile = DatasetProfile(
                structured_table="structured",
                unstructured_table=None,
                row_count=10,
                columns={"rating": "rating", "review_text": "review_text"},
                text_columns=["review_text"],
                numeric_columns=["rating"],
            )
            mock_inspector.return_value.inspect.return_value = profile

            with patch("duckdb.connect") as mock_connect:
                con = MagicMock()
                mock_connect.return_value = con
                con.execute.return_value.fetchall.return_value = [(4.5, "category")]
                con.execute.return_value.fetchone.return_value = [4.5]

                generator = BenchmarkGenerator(config)
                generator.run()
                assert config.output_path.exists()
                with open(config.output_path) as f:
                    data = json.load(f)
                    assert len(data) == 6  # 2+2+2

    def test_benchmark_record_serialization(self):
        record = BenchmarkRecord(
            id="test",
            question="Test?",
            query_type="structured",
            difficulty="easy",
            intent="aggregation",
            retrieval_type="sql",
            expected_answer=42,
            expected_keywords=["test"],
        )
        d = record.to_dict()
        assert d["id"] == "test"
        assert d["expected_answer"] == 42

    def test_schema_inspector_maps_columns(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("product_id,rating,review_text\n1,4.5,Good\n2,3.0,Okay\n")
        inspector = SchemaInspector(csv_path)
        profile = inspector.inspect()
        assert profile.row_count == 2
        assert "rating" in profile.columns.values()
        assert "review_text" in profile.text_columns

    def test_benchmark_handles_empty_dataset(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("product_id,rating\n")
        inspector = SchemaInspector(csv_path)
        with pytest.raises(ValueError, match="Dataset is empty"):
            inspector.inspect()

    def test_benchmark_generator_handles_missing_mapping(self, tmp_path):
        config = BenchmarkConfig(
            output_path=tmp_path / "out.json",
            data_dir=tmp_path,
        )
        with patch("evaluation.benchmark_generator.SchemaInspector") as mock_inspector:
            profile = DatasetProfile(row_count=1, columns={"rating": "rating"}, structured_table="t")
            mock_inspector.return_value.inspect.return_value = profile
            with patch("duckdb.connect"):
                gen = BenchmarkGenerator(config)
                gen.run()
                assert config.output_path.exists()


# -----------------------------------------------------------------------------
# Tests: Accuracy Metrics
# -----------------------------------------------------------------------------

class TestAccuracyMetrics:
    """Test accuracy evaluation module."""

    def test_structured_evaluator_exact_match(self):
        evaluator = StructuredAccuracyEvaluator()
        result = evaluator.exact_match("hello", "hello")
        assert result.score == 1.0
        assert result.passed is True
        result2 = evaluator.exact_match("hello", "world")
        assert result2.score == 0.0

    def test_structured_evaluator_relative_error(self):
        evaluator = StructuredAccuracyEvaluator()
        result = evaluator.relative_error(10.0, 9.5)
        assert 0.9 < result.score < 1.0
        result = evaluator.relative_error(0.0, 0.0)
        assert result.score == 1.0
        result = evaluator.relative_error(0.0, 5.0)
        assert result.score == 0.0

    def test_structured_evaluator_evaluate(self):
        evaluator = StructuredAccuracyEvaluator()
        results = evaluator.evaluate(expected=5.0, generated=4.8)
        assert len(results) > 1
        assert any(r.metric_name == "numeric_error" for r in results)

        results = evaluator.evaluate(expected=["a", "b"], generated=["a", "c"])
        topk = next(r for r in results if r.metric_name == "top_k_match")
        assert topk.score == 0.5

    def test_semantic_evaluator_embedding(self, tmp_path):
        with patch("evaluation.accuracy_metrics.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_st.return_value = mock_model
            evaluator = SemanticAccuracyEvaluator(model_name="test-model")
            # Cache test
            emb_sim = evaluator._embedding_similarity("text1", "text2")
            assert isinstance(emb_sim, float)
            # BLEU
            results = evaluator.evaluate("hello world", "hello planet")
            bleu = next(r for r in results if r.metric_name == "bleu")
            assert bleu.score >= 0

    def test_retrieval_evaluator(self):
        evaluator = RetrievalEvaluator(k=2)
        retrieved = ["chunk1", "chunk2", "chunk3"]
        relevant = ["chunk2", "chunk4"]
        results = evaluator.evaluate(retrieved, relevant, "answer")
        p_at_k = next(r for r in results if r.metric_name == "precision_at_k")
        assert p_at_k.score == 0.5
        recall = next(r for r in results if r.metric_name == "recall_at_k")
        assert recall.score == 0.5

    def test_hallucination_evaluator(self):
        evaluator = HallucinationEvaluator()
        context = ["The sky is blue.", "The grass is green."]
        answer = "The sky is blue and the grass is green."
        results = evaluator.evaluate(answer, context)
        support = next(r for r in results if r.metric_name == "support_score")
        assert support.score > 0.5
        results = evaluator.evaluate("", context)
        support = next(r for r in results if r.metric_name == "support_score")
        assert support.score == 0.0

    def test_hybrid_evaluator_weighted(self):
        evaluator = HybridEvaluator()
        s1 = MetricResult("acc", 0.9, True)
        s2 = MetricResult("sem", 0.8, True)
        r1 = MetricResult("ret", 0.7, True)
        h1 = MetricResult("hall", 0.95, True)
        result = evaluator.evaluate([s1], [s2], [r1], [h1])
        assert result.score > 0.7

    def test_accuracy_metrics_runner(self, tmp_path, temp_benchmark_file, mock_hybrid_retriever, mock_answer_generator):
        evaluator = AccuracyMetrics(
            benchmark_path=temp_benchmark_file,
            output_dir=tmp_path / "reports",
            sample_size=1,
            parallel=False,
            model_name="test-model",  # Use a dummy model name to avoid loading
        )
        # Patch the SentenceTransformer inside SemanticAccuracyEvaluator to avoid real load
        with patch("evaluation.accuracy_metrics.SentenceTransformer") as mock_st:
            mock_st.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
            evaluator.run()
            report_path = tmp_path / "reports" / "accuracy_report.json"
            assert report_path.exists()


# -----------------------------------------------------------------------------
# Tests: Latency Metrics
# -----------------------------------------------------------------------------

class TestLatencyMetrics:
    """Test latency measurement module."""

    def test_stage_timer(self):
        timer = StageTimer()
        with timer.measure("test"):
            import time
            time.sleep(0.001)
        latencies = timer.get_latencies()
        assert len(latencies) == 1
        assert latencies[0].stage == "test"
        assert latencies[0].duration_ms > 0

    def test_stage_timer_nested(self):
        timer = StageTimer()
        with timer.measure("outer"):
            with timer.measure("inner"):
                pass
        assert len(timer.get_latencies()) == 2
        assert timer.get_total_duration() > 0

    def test_latency_analyzer(self):
        values = [100, 200, 300, 400, 500]
        stats = LatencyAnalyzer.compute_statistics(values)
        assert stats.mean == 300.0
        assert stats.p50 == 300.0

    def test_resource_monitor(self):
        monitor = ResourceMonitor()
        with monitor.track():
            import time
            time.sleep(0.01)
        assert len(monitor.snapshots) > 0
        assert monitor.get_peak_memory_mb() > 0

    def test_pipeline_profiler(self, tmp_path):
        with patch("evaluation.latency_metrics.IntentRouter") as mock_router, \
             patch("evaluation.latency_metrics.RetrievalPlanBuilder") as mock_planner, \
             patch("evaluation.latency_metrics.HybridRetriever") as mock_retriever, \
             patch("evaluation.latency_metrics.ContextCompressor") as mock_compressor, \
             patch("evaluation.latency_metrics.StructuredAgent") as mock_sagent, \
             patch("evaluation.latency_metrics.SemanticAgent") as mock_magent, \
             patch("evaluation.latency_metrics.HybridAgent") as mock_hagent, \
             patch("evaluation.latency_metrics.AnswerGenerator") as mock_ag, \
             patch("evaluation.latency_metrics.QwenClient") as mock_qc:
            mock_router.return_value.route.return_value = "structured"
            mock_planner.return_value.build_plan.return_value = {}
            mock_retriever.return_value.retrieve.return_value = {"chunks": ["ctx"]}
            mock_retriever.return_value.retrieve_structured.return_value = {"chunks": ["ctx"]}
            mock_compressor.return_value.compress.return_value = ["ctx_comp"]
            mock_sagent.return_value.process.return_value = "agent_out"
            mock_magent.return_value.process.return_value = "agent_out"
            mock_hagent.return_value.process.return_value = "agent_out"
            mock_ag.return_value.build_prompt.return_value = "prompt"
            mock_ag.return_value.generate_with_stats.return_value = ("answer", 10)
            mock_ag.return_value.postprocess.return_value = "final"

            profiler = PipelineProfiler()
            result, timer = profiler.profile_question("test q", "structured")
            assert result.total_latency_ms > 0
            assert result.token_count == 10
            assert len(result.latencies) > 0

    def test_latency_metrics_runner(self, tmp_path, temp_benchmark_file):
        with patch("evaluation.latency_metrics.PipelineProfiler") as mock_profiler:
            mock_result = QuestionLatencyResult(
                question_id="1",
                question="test",
                query_type="structured",
                latencies=[StageLatency("test", 0, 1, 100, {})],
                total_latency_ms=100,
                token_count=10,
                tokens_per_second=100,
            )
            mock_profiler.return_value.profile_question.return_value = (mock_result, MagicMock())
            mock_profiler.return_value.profile_batch.return_value = [mock_result]

            evaluator = LatencyMetrics(
                benchmark_path=temp_benchmark_file,
                output_dir=tmp_path / "reports",
                sample_size=2,
                parallel=False,
            )
            evaluator.run()
            report_path = tmp_path / "reports" / "latency_report.json"
            assert report_path.exists()


# -----------------------------------------------------------------------------
# Tests: LLM Judge
# -----------------------------------------------------------------------------

class TestLLMJudge:
    """Test LLM-based judge module."""

    def test_prompt_builder(self):
        builder = PromptBuilder()
        prompt = builder.build_accuracy_prompt("Q", "C", "A", "GT")
        assert "Question: Q" in prompt
        assert "Retrieved Context: C" in prompt

    def test_response_parser(self):
        parser = ResponseParser()
        text = '{"score": 8, "confidence": 0.9, "reason": "good"}'
        data = parser.parse_json(text)
        assert data["score"] == 8
        text2 = '{"score": 8, "confidence": 0.9, "reason": "good"'
        data2 = parser.parse_json(text2)
        assert data2.get("score") == 8
        text3 = '```json\n{"score": 9}\n```'
        data3 = parser.parse_json(text3)
        assert data3["score"] == 9

    def test_answer_judge(self, mock_qwen_client):
        builder = PromptBuilder()
        parser = ResponseParser()
        judge = AnswerJudge(mock_qwen_client, builder, parser, retries=0)
        score = judge.score_accuracy("Q", "C", "A", "GT")
        assert score.score == 8.0
        assert score.passed is True

    def test_answer_judge_hallucination(self, mock_qwen_client):
        builder = PromptBuilder()
        parser = ResponseParser()
        mock_qwen_client.generate.return_value = json.dumps({
            "hallucination": True,
            "confidence": 0.8,
            "reason": "Hallucination detected."
        })
        judge = AnswerJudge(mock_qwen_client, builder, parser, retries=0)
        score = judge.detect_hallucination("Q", "C", "A", "GT")
        assert score.score == 0.0
        assert score.passed is False

    def test_composite_judge(self):
        scores = [
            JudgeScore("accuracy", 8.0, True, 0.9, ""),
            JudgeScore("completeness", 7.0, True, 0.8, ""),
            JudgeScore("relevance", 9.0, True, 0.9, ""),
            JudgeScore("faithfulness", 8.5, True, 0.85, ""),
            JudgeScore("reasoning", 7.5, True, 0.8, ""),
        ]
        judge = CompositeJudge()
        overall, verdict = judge.aggregate(scores)
        assert 7.5 < overall < 8.5
        assert verdict == "pass"

    def test_llm_judge_runner(self, tmp_path, temp_benchmark_file):
        with patch("evaluation.llm_judge.QwenClient") as mock_qc, \
             patch("evaluation.llm_judge.AnswerJudge") as mock_aj:
            mock_aj_instance = MagicMock()
            mock_aj.return_value = mock_aj_instance
            mock_aj_instance.score_accuracy.return_value = JudgeScore("accuracy", 8.0, True, 0.9, "")
            mock_aj_instance.score_completeness.return_value = JudgeScore("completeness", 7.0, True, 0.8, "")
            mock_aj_instance.score_relevance.return_value = JudgeScore("relevance", 9.0, True, 0.9, "")
            mock_aj_instance.score_faithfulness.return_value = JudgeScore("faithfulness", 8.0, True, 0.8, "")
            mock_aj_instance.score_reasoning.return_value = JudgeScore("reasoning", 7.0, True, 0.8, "")
            mock_aj_instance.detect_hallucination.return_value = JudgeScore("hallucination", 10.0, True, 0.9, "")

            judge = LLMJudge(
                benchmark_path=temp_benchmark_file,
                output_dir=tmp_path / "reports",
                sample_size=2,
                parallel=False,
                retries=0,
            )
            judge.run()
            report_path = tmp_path / "reports" / "llm_judge_report.json"
            assert report_path.exists()


# -----------------------------------------------------------------------------
# Tests: Evaluation Runner
# -----------------------------------------------------------------------------

class TestEvaluationRunner:
    """Test the evaluation orchestrator."""

    def test_evaluation_config_defaults(self):
        config = EvaluationConfig()
        assert config.generate_benchmark is True
        assert config.enable_accuracy is True
        assert config.enable_latency is True
        assert config.enable_judge is True

    def test_evaluation_pipeline_steps(self, tmp_path, temp_benchmark_file):
        config = EvaluationConfig(
            benchmark_path=temp_benchmark_file,
            accuracy_output=tmp_path / "accuracy_report.json",
            latency_output=tmp_path / "latency_report.json",
            judge_output=tmp_path / "llm_judge_report.json",
            state_file=tmp_path / ".state.json",
            generate_benchmark=False,
            enable_accuracy=True,
            enable_latency=True,
            enable_judge=True,
            resume=False,
        )
        with patch("evaluation.evaluation_runner.AccuracyMetrics") as mock_acc, \
             patch("evaluation.evaluation_runner.LatencyMetrics") as mock_lat, \
             patch("evaluation.evaluation_runner.LLMJudge") as mock_judge:
            mock_acc.return_value.run.return_value = None
            mock_lat.return_value.run.return_value = None
            mock_judge.return_value.run.return_value = None

            pipeline = EvaluationPipeline(config)
            result = pipeline.run()
            assert pipeline.state == EvaluationState.COMPLETED
            assert result is not None

    def test_evaluation_pipeline_resume(self, tmp_path):
        state_file = tmp_path / ".state.json"
        state_data = {"state": "accuracy_done", "timestamp": "now"}
        with open(state_file, "w") as f:
            json.dump(state_data, f)
        config = EvaluationConfig(
            benchmark_path=tmp_path / "bench.json",
            state_file=state_file,
            resume=True,
            generate_benchmark=True,
            enable_accuracy=True,
            enable_latency=True,
            enable_judge=True,
        )
        with patch("evaluation.evaluation_runner.AccuracyMetrics") as mock_acc, \
             patch("evaluation.evaluation_runner.LatencyMetrics") as mock_lat, \
             patch("evaluation.evaluation_runner.LLMJudge") as mock_judge:
            pipeline = EvaluationPipeline(config)
            # We'll spy on the run_accuracy method
            pipeline.run_accuracy = MagicMock()
            pipeline.run_latency = MagicMock()
            pipeline.run_judge = MagicMock()
            pipeline.generate_benchmark = MagicMock()
            pipeline.run()
            # Accuracy should not be called because state is already at accuracy_done
            pipeline.run_accuracy.assert_not_called()
            # Latency and judge should be called
            pipeline.run_latency.assert_called()
            pipeline.run_judge.assert_called()

    def test_report_aggregator(self, tmp_path, sample_accuracy_report, sample_latency_report, sample_judge_report):
        config = EvaluationConfig(
            accuracy_output=sample_accuracy_report,
            latency_output=sample_latency_report,
            judge_output=sample_judge_report,
        )
        aggregator = ReportAggregator(config)
        aggregator.load_all()
        result = aggregator.aggregate()
        assert result.questions_evaluated == 3
        assert result.accuracy_score > 0
        assert result.latency_score > 0
        assert result.judge_score > 0
        assert result.hallucination_rate == 0.05

    def test_evaluation_runner_integration(self, tmp_path, temp_benchmark_file):
        config = EvaluationConfig(
            benchmark_path=temp_benchmark_file,
            accuracy_output=tmp_path / "accuracy_report.json",
            latency_output=tmp_path / "latency_report.json",
            judge_output=tmp_path / "llm_judge_report.json",
            final_output=tmp_path / "final_report.json",
            summary_output=tmp_path / "summary.json",
            dashboard_output=tmp_path / "dashboard.csv",
            generate_benchmark=False,
            enable_accuracy=True,
            enable_latency=True,
            enable_judge=True,
            parallel=False,
            workers=1,
        )
        with patch("evaluation.evaluation_runner.AccuracyMetrics") as mock_acc, \
             patch("evaluation.evaluation_runner.LatencyMetrics") as mock_lat, \
             patch("evaluation.evaluation_runner.LLMJudge") as mock_judge:
            # Mock run() to produce report files
            def create_acc():
                with open(config.accuracy_output, "w") as f:
                    json.dump({"summary": {"total": 3, "average_overall_score": 0.9}}, f)
            def create_lat():
                with open(config.latency_output, "w") as f:
                    json.dump({"summary": {"total_questions": 3, "total_latency_ms_stats": {"mean": 800}}}, f)
            def create_judge():
                with open(config.judge_output, "w") as f:
                    json.dump({"summary": {"total": 3, "average_overall_score": 0.85, "hallucination_rate": 0.02}}, f)
            mock_acc.return_value.run.side_effect = create_acc
            mock_lat.return_value.run.side_effect = create_lat
            mock_judge.return_value.run.side_effect = create_judge

            runner = EvaluationRunner(config)
            runner.run()
            assert config.final_output.exists()
            assert config.summary_output.exists()
            assert config.dashboard_output.exists()


# -----------------------------------------------------------------------------
# Tests: Report Validation
# -----------------------------------------------------------------------------

class TestReportValidation:
    """Test report schema and validation."""

    def test_final_report_schema(self, tmp_path):
        result = EvaluationResult(
            questions_evaluated=10,
            accuracy_score=85.0,
            latency_score=70.0,
            judge_score=90.0,
            hallucination_rate=0.02,
            average_latency_ms=500,
            overall_score=82.5,
            duration_seconds=120.5,
        )
        report = {"summary": result.to_dict(), "details": {}}
        with open(tmp_path / "report.json", "w") as f:
            json.dump(report, f)
        with open(tmp_path / "report.json") as f:
            data = json.load(f)
            assert "summary" in data
            assert "questions_evaluated" in data["summary"]

    def test_summary_values(self):
        result = EvaluationResult(
            questions_evaluated=5,
            accuracy_score=90.0,
            latency_score=80.0,
            judge_score=95.0,
            hallucination_rate=0.01,
            average_latency_ms=300,
            overall_score=88.0,
            duration_seconds=60.0,
        )
        assert result.overall_score == 88.0

    def test_dashboard_export(self, tmp_path):
        result = EvaluationResult(
            questions_evaluated=3,
            accuracy_score=90.0,
            latency_score=80.0,
            judge_score=95.0,
            hallucination_rate=0.01,
            average_latency_ms=300,
            overall_score=88.0,
            duration_seconds=60.0,
        )
        data = {
            "metric": ["questions_evaluated", "overall_score"],
            "value": [result.questions_evaluated, result.overall_score],
        }
        df = pd.DataFrame(data)
        path = tmp_path / "dashboard.csv"
        df.to_csv(path, index=False)
        assert path.exists()
        df2 = pd.read_csv(path)
        assert df2.iloc[0]["value"] == 3


# -----------------------------------------------------------------------------
# Tests: Edge Cases and Failure Recovery
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and failure scenarios."""

    def test_accuracy_metrics_missing_benchmark(self, tmp_path):
        evaluator = AccuracyMetrics(
            benchmark_path=tmp_path / "missing.json",
            output_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            evaluator.load_benchmark()

    def test_accuracy_metrics_malformed_benchmark(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not a json array")
        evaluator = AccuracyMetrics(benchmark_path=path, output_dir=tmp_path)
        with pytest.raises(json.JSONDecodeError):
            evaluator.load_benchmark()

    def test_latency_metrics_empty_benchmark(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        evaluator = LatencyMetrics(benchmark_path=path, output_dir=tmp_path)
        evaluator.run()
        report_path = tmp_path / "latency_report.json"
        assert report_path.exists()

    def test_llm_judge_response_parser_repair(self):
        parser = ResponseParser()
        text = '{"score": 8, "confidence": 0.9, "reason": "good"'
        data = parser.parse_json(text)
        assert data.get("score") == 8
        text2 = '{"score": 8, "confidence": 0.9, "reason": "good" extra}'
        data2 = parser.parse_json(text2)
        assert data2.get("score") == 8

    def test_runner_dry_run(self, tmp_path):
        config = EvaluationConfig(
            benchmark_path=tmp_path / "bench.json",
            dry_run=True,
        )
        runner = EvaluationRunner(config)
        with patch("evaluation.evaluation_runner.EvaluationPipeline") as mock_pipeline:
            runner.run()
            mock_pipeline.return_value.run.assert_not_called()


# -----------------------------------------------------------------------------
# Main entry for pytest
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])