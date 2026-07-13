#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
accuracy_metrics.py

Complete evaluation framework for Hybrid RAG system.
Evaluates structured, semantic, retrieval, and hallucination metrics
against benchmark_questions.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Core ML / NLP libraries (with graceful fallback)
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    # Ensure punkt tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except ImportError:
    raise ImportError("nltk is required. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("rouge-score is required. Install with: pip install rouge-score")

# -----------------------------------------------------------------------------
# Project-specific imports (RAG pipeline components)
# -----------------------------------------------------------------------------
try:
    from retrieval.hybrid_retriever import HybridRetriever
    from llm.answer_generator import AnswerGenerator
    from agents.intent_router import IntentRouter
    from agents.retrieval_plan_builder import RetrievalPlanBuilder
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
LOG_FILE = LOG_DIR / "accuracy_metrics.log"

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
class MetricType(Enum):
    """Enumeration of all metric types."""
    EXACT_MATCH = "exact_match"
    NUMERIC_ERROR = "numeric_error"
    RELATIVE_ERROR = "relative_error"
    AGGREGATION_ACCURACY = "aggregation_accuracy"
    TOP_K_MATCH = "top_k_match"
    RANGE_ACCURACY = "range_accuracy"
    DISTRIBUTION_SIMILARITY = "distribution_similarity"
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    BERTSCORE = "bertscore"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    KEYWORD_RECALL = "keyword_recall"
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    MRR = "mrr"
    HIT_RATE = "hit_rate"
    CONTEXT_COVERAGE = "context_coverage"
    SUPPORT_SCORE = "support_score"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    FAITHFULNESS = "faithfulness"
    HYBRID_WEIGHTED = "hybrid_weighted"


class QuestionType(Enum):
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    metric_name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationRecord:
    """Complete evaluation record for one benchmark question."""

    question_id: str
    question: str
    query_type: str
    generated_answer: str
    expected_answer: Any
    retrieved_context: List[str]
    metrics: List[MetricResult]
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "query_type": self.query_type,
            "generated_answer": self.generated_answer,
            "expected_answer": self.expected_answer,
            "retrieved_context": self.retrieved_context,
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_score": self.overall_score,
        }


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""
    import re
    text = re.sub(r"\s+", " ", text.lower().strip())
    return text


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words using NLTK word tokenizer, with fallback."""
    if not text:
        return []
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    except (LookupError, ImportError):
        # Fallback: simple split
        return text.lower().split()


def safe_float(value: Any) -> Optional[float]:
    """Safely convert to float, return None if fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# 1. Structured Accuracy Evaluator
# -----------------------------------------------------------------------------
class StructuredAccuracyEvaluator:
    """
    Evaluates structured (SQL-like) answers.
    Supports exact match, numeric error, relative error, aggregation,
    top-K, range, and distribution metrics.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def evaluate(
        self,
        expected: Any,
        generated: Any,
        expected_type: Optional[str] = None,
    ) -> List[MetricResult]:
        """
        Evaluate structured answer and return list of MetricResult.
        The expected_type can hint at the type of question.
        """
        results = []
        # Determine expected and generated types
        if expected is None or generated is None:
            results.append(
                MetricResult(
                    metric_name="exact_match",
                    score=0.0,
                    passed=False,
                    details={"reason": "Missing expected or generated answer"},
                )
            )
            return results

        # Exact match (for strings, numbers, lists)
        exact = self.exact_match(expected, generated)
        results.append(exact)

        # Numeric error (if both are numeric)
        num_exp = safe_float(expected)
        num_gen = safe_float(generated)
        if num_exp is not None and num_gen is not None:
            # Relative error
            rel_err = self.relative_error(num_exp, num_gen)
            results.append(rel_err)
            # Absolute error (as score, e.g., 1 - abs_err / (abs(exp)+1))
            abs_err = abs(num_exp - num_gen)
            abs_score = max(0.0, 1.0 - abs_err / (abs(num_exp) + 1.0))
            results.append(
                MetricResult(
                    metric_name="numeric_error",
                    score=abs_score,
                    passed=abs_err < 1.0,
                    details={"absolute_error": abs_err},
                )
            )

        # Aggregation accuracy: if expected is a scalar and generated is scalar
        if isinstance(expected, (int, float)) and isinstance(generated, (int, float)):
            agg_acc = 1.0 if abs(expected - generated) < self.tolerance else 0.0
            results.append(
                MetricResult(
                    metric_name="aggregation_accuracy",
                    score=agg_acc,
                    passed=agg_acc > 0.5,
                )
            )

        # Top-K match: if expected is a list and generated is list, compute overlap
        if isinstance(expected, list) and isinstance(generated, list):
            exp_set = set(str(e).strip().lower() for e in expected)
            gen_set = set(str(g).strip().lower() for g in generated)
            if exp_set and gen_set:
                overlap = len(exp_set & gen_set) / len(exp_set)
            else:
                overlap = 0.0
            results.append(
                MetricResult(
                    metric_name="top_k_match",
                    score=overlap,
                    passed=overlap > 0.5,
                    details={"overlap": overlap},
                )
            )

        # Range accuracy: if expected is a tuple (min, max) and generated is a value
        if isinstance(expected, tuple) and len(expected) == 2:
            try:
                lo, hi = expected
                if lo <= generated <= hi:
                    results.append(
                        MetricResult(
                            metric_name="range_accuracy",
                            score=1.0,
                            passed=True,
                            details={"range": (lo, hi), "value": generated},
                        )
                    )
                else:
                    results.append(
                        MetricResult(
                            metric_name="range_accuracy",
                            score=0.0,
                            passed=False,
                            details={"range": (lo, hi), "value": generated},
                        )
                    )
            except Exception:
                pass

        # Distribution similarity (if both are dicts)
        if isinstance(expected, dict) and isinstance(generated, dict):
            keys = set(expected.keys()) | set(generated.keys())
            if keys:
                exp_vals = np.array([expected.get(k, 0) for k in keys])
                gen_vals = np.array([generated.get(k, 0) for k in keys])
                if np.linalg.norm(exp_vals) > 0 and np.linalg.norm(gen_vals) > 0:
                    sim = np.dot(exp_vals, gen_vals) / (np.linalg.norm(exp_vals) * np.linalg.norm(gen_vals))
                else:
                    sim = 0.0
                results.append(
                    MetricResult(
                        metric_name="distribution_similarity",
                        score=float(sim),
                        passed=sim > 0.5,
                        details={"cosine_similarity": float(sim)},
                    )
                )

        return results

    def exact_match(self, expected: Any, generated: Any) -> MetricResult:
        """Check exact match after converting to string and normalizing."""
        exp_str = normalize_text(str(expected))
        gen_str = normalize_text(str(generated))
        match = exp_str == gen_str
        return MetricResult(
            metric_name="exact_match",
            score=1.0 if match else 0.0,
            passed=match,
            details={"expected_normalized": exp_str, "generated_normalized": gen_str},
        )

    def relative_error(self, expected: float, generated: float) -> MetricResult:
        """Compute relative error and return as a metric."""
        if abs(expected) < self.tolerance:
            rel_err = 0.0 if abs(generated) < self.tolerance else 1.0
        else:
            rel_err = abs(expected - generated) / abs(expected)
        score = max(0.0, 1.0 - rel_err)
        return MetricResult(
            metric_name="relative_error",
            score=score,
            passed=rel_err < 0.1,
            details={"relative_error": rel_err},
        )


# -----------------------------------------------------------------------------
# 2. Semantic Accuracy Evaluator
# -----------------------------------------------------------------------------
class SemanticAccuracyEvaluator:
    """
    Evaluates semantic (text-based) answers using BLEU, ROUGE, BERTScore,
    embedding similarity, and keyword metrics.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedder: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}
        self._batch_size = 32

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                self._embedder = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
        return self._embedder

    def evaluate(
        self,
        expected: str,
        generated: str,
        expected_keywords: Optional[List[str]] = None,
    ) -> List[MetricResult]:
        """
        Evaluate semantic similarity.
        expected and generated are strings.
        expected_keywords is optional list of keywords from ground truth.
        """
        results = []
        if not expected or not generated:
            # If either is empty, return zero scores
            results.append(
                MetricResult(
                    metric_name="bleu",
                    score=0.0,
                    passed=False,
                    details={"reason": "Empty reference or hypothesis"},
                )
            )
            results.append(
                MetricResult(
                    metric_name="rouge_l",
                    score=0.0,
                    passed=False,
                    details={"reason": "Empty reference or hypothesis"},
                )
            )
            results.append(
                MetricResult(
                    metric_name="embedding_similarity",
                    score=0.0,
                    passed=False,
                    details={"reason": "Empty reference or hypothesis"},
                )
            )
            return results

        # BLEU
        bleu_score = self._compute_bleu(expected, generated)
        results.append(
            MetricResult(
                metric_name="bleu",
                score=bleu_score,
                passed=bleu_score > 0.1,
                details={"bleu": bleu_score},
            )
        )

        # ROUGE-L
        rouge_l = self._compute_rouge_l(expected, generated)
        results.append(
            MetricResult(
                metric_name="rouge_l",
                score=rouge_l,
                passed=rouge_l > 0.2,
                details={"rougeL_f1": rouge_l},
            )
        )

        # Embedding similarity (cosine)
        emb_sim = self._embedding_similarity(expected, generated)
        results.append(
            MetricResult(
                metric_name="embedding_similarity",
                score=emb_sim,
                passed=emb_sim > 0.6,
                details={"cosine_similarity": emb_sim},
            )
        )

        # BERTScore placeholder: we use embedding similarity as proxy
        results.append(
            MetricResult(
                metric_name="bertscore",
                score=emb_sim,  # proxy
                passed=emb_sim > 0.6,
                details={"bertscore_proxy": "embedding_similarity"},
            )
        )

        # Keyword recall and precision if keywords provided
        if expected_keywords:
            exp_keywords = [normalize_text(kw) for kw in expected_keywords if kw]
            gen_keywords = self._extract_keywords(generated, top_n=len(exp_keywords) * 2)
            if exp_keywords:
                recalled = sum(1 for kw in exp_keywords if any(kw in gk for gk in gen_keywords))
                recall = recalled / len(exp_keywords)
            else:
                recall = 0.0
            results.append(
                MetricResult(
                    metric_name="keyword_recall",
                    score=recall,
                    passed=recall > 0.5,
                    details={"recall": recall, "expected_keywords": exp_keywords, "generated_keywords": gen_keywords},
                )
            )

        return results

    def _compute_bleu(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score."""
        ref_tokens = tokenize_text(reference)
        hyp_tokens = tokenize_text(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return 0.0
        smoothing = SmoothingFunction().method4
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Compute ROUGE-L F1 score."""
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return scores["rougeL"].fmeasure
        except Exception:
            return 0.0

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between embeddings of two texts."""
        if not text1 or not text2:
            return 0.0
        # Use cache if available
        if text1 in self._cache and text2 in self._cache:
            emb1 = self._cache[text1]
            emb2 = self._cache[text2]
        else:
            # Batch encode to reduce overhead
            texts_to_encode = []
            keys = []
            if text1 not in self._cache:
                texts_to_encode.append(text1)
                keys.append(text1)
            if text2 not in self._cache:
                texts_to_encode.append(text2)
                keys.append(text2)
            if texts_to_encode:
                embeddings = self.embedder.encode(texts_to_encode, batch_size=self._batch_size, show_progress_bar=False)
                for key, emb in zip(keys, embeddings):
                    self._cache[key] = emb
            emb1 = self._cache[text1]
            emb2 = self._cache[text2]
        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(sim)

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords using simple frequency."""
        from collections import Counter
        words = tokenize_text(text)
        stopwords = {"a", "an", "the", "and", "or", "but", "if", "because", "as",
                     "until", "while", "of", "at", "by", "for", "with", "without",
                     "via", "during", "in", "on", "to", "from", "into", "through",
                     "although", "whereas", "etc", "e.g", "i", "you", "he", "she",
                     "it", "we", "they", "me", "him", "her", "us", "them", "my",
                     "your", "his", "its", "our", "their"}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        counter = Counter(words)
        return [word for word, _ in counter.most_common(top_n)]


# -----------------------------------------------------------------------------
# 3. Retrieval Evaluator
# -----------------------------------------------------------------------------
class RetrievalEvaluator:
    """
    Evaluates retrieval quality: Precision@k, Recall@k, MRR, Hit Rate,
    and context coverage.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(
        self,
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        answer: Optional[str] = None,
    ) -> List[MetricResult]:
        """
        Evaluate retrieval metrics.
        retrieved_chunks: list of chunk texts retrieved.
        relevant_chunks: list of ground truth relevant chunk texts (or IDs).
        answer: the generated answer to compute context coverage.
        """
        results = []

        # Convert to sets for overlap
        retrieved_set = set(normalize_text(c) for c in retrieved_chunks)
        relevant_set = set(normalize_text(c) for c in relevant_chunks)

        # Precision@k
        if retrieved_chunks and relevant_set:
            top_k = retrieved_chunks[:self.k]
            top_k_set = set(normalize_text(c) for c in top_k)
            precision = len(top_k_set & relevant_set) / len(top_k_set) if top_k_set else 0.0
        else:
            precision = 0.0
        results.append(
            MetricResult(
                metric_name="precision_at_k",
                score=precision,
                passed=precision > 0.5,
                details={"k": self.k, "precision": precision},
            )
        )

        # Recall@k
        if relevant_set and retrieved_chunks:
            top_k_set = set(normalize_text(c) for c in retrieved_chunks[:self.k])
            recall = len(top_k_set & relevant_set) / len(relevant_set) if relevant_set else 0.0
        else:
            recall = 0.0
        results.append(
            MetricResult(
                metric_name="recall_at_k",
                score=recall,
                passed=recall > 0.5,
                details={"k": self.k, "recall": recall},
            )
        )

        # MRR (Mean Reciprocal Rank): find first relevant
        mrr = 0.0
        for i, chunk in enumerate(retrieved_chunks, start=1):
            if normalize_text(chunk) in relevant_set:
                mrr = 1.0 / i
                break
        results.append(
            MetricResult(
                metric_name="mrr",
                score=mrr,
                passed=mrr > 0.5,
                details={"mrr": mrr},
            )
        )

        # Hit Rate: at least one relevant in top-k
        hit = 1.0 if any(normalize_text(c) in relevant_set for c in retrieved_chunks[:self.k]) else 0.0
        results.append(
            MetricResult(
                metric_name="hit_rate",
                score=hit,
                passed=hit > 0.5,
                details={"hit": bool(hit)},
            )
        )

        # Context Coverage: overlap between retrieved context and answer
        if answer and retrieved_chunks:
            ans_tokens = set(tokenize_text(answer))
            ctx_tokens = set()
            for chunk in retrieved_chunks:
                ctx_tokens.update(tokenize_text(chunk))
            if ans_tokens:
                overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
            else:
                overlap = 0.0
            results.append(
                MetricResult(
                    metric_name="context_coverage",
                    score=overlap,
                    passed=overlap > 0.5,
                    details={"coverage": overlap},
                )
            )
        else:
            results.append(
                MetricResult(
                    metric_name="context_coverage",
                    score=0.0,
                    passed=False,
                    details={"reason": "Missing answer or retrieved context"},
                )
            )

        return results


# -----------------------------------------------------------------------------
# 4. Hallucination Evaluator
# -----------------------------------------------------------------------------
class HallucinationEvaluator:
    """
    Detects hallucinations by comparing generated answer to retrieved context.
    Uses claim extraction and support checking via lexical overlap or entailment.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        generated_answer: str,
        retrieved_context: List[str],
    ) -> List[MetricResult]:
        """
        Evaluate hallucination metrics.
        Returns support score, unsupported claims, faithfulness.
        """
        results = []

        if not generated_answer or not retrieved_context:
            results.append(
                MetricResult(
                    metric_name="support_score",
                    score=0.0,
                    passed=False,
                    details={"reason": "Missing answer or context"},
                )
            )
            return results

        context_text = " ".join(retrieved_context)

        # Extract claims from answer (simple: split into sentences)
        try:
            sentences = nltk.sent_tokenize(generated_answer)
        except LookupError:
            sentences = generated_answer.split(". ")
        if not sentences:
            sentences = [generated_answer]

        support_ratios = []
        for sent in sentences:
            sent_tokens = set(tokenize_text(sent))
            ctx_tokens = set(tokenize_text(context_text))
            if sent_tokens:
                overlap = len(sent_tokens & ctx_tokens) / len(sent_tokens)
            else:
                overlap = 0.0
            support_ratios.append(overlap)

        support_score = np.mean(support_ratios) if support_ratios else 0.0
        support_score = float(support_score)
        results.append(
            MetricResult(
                metric_name="support_score",
                score=support_score,
                passed=support_score >= self.threshold,
                details={"average_support": support_score},
            )
        )

        # Unsupported claims: number of sentences with overlap < threshold
        unsupported_count = sum(1 for r in support_ratios if r < self.threshold)
        total_sentences = len(support_ratios)
        unsupported_ratio = unsupported_count / total_sentences if total_sentences > 0 else 0.0
        unsupported_score = 1.0 - unsupported_ratio  # higher is better
        results.append(
            MetricResult(
                metric_name="unsupported_claims",
                score=unsupported_score,
                passed=unsupported_ratio < 0.3,
                details={"unsupported_count": unsupported_count, "total_sentences": total_sentences},
            )
        )

        # Faithfulness: overall score = support_score * (1 - unsupported_ratio)
        faithfulness = support_score * (1.0 - unsupported_ratio)
        results.append(
            MetricResult(
                metric_name="faithfulness",
                score=faithfulness,
                passed=faithfulness >= 0.5,
                details={"faithfulness": faithfulness},
            )
        )

        return results


# -----------------------------------------------------------------------------
# 5. Hybrid Evaluator
# -----------------------------------------------------------------------------
class HybridEvaluator:
    """
    Aggregates all metrics into a single weighted hybrid score.
    """

    DEFAULT_WEIGHTS = {
        "structured": 0.35,
        "semantic": 0.30,
        "retrieval": 0.20,
        "hallucination": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def evaluate(
        self,
        structured_metrics: List[MetricResult],
        semantic_metrics: List[MetricResult],
        retrieval_metrics: List[MetricResult],
        hallucination_metrics: List[MetricResult],
    ) -> MetricResult:
        """Combine metrics using weighted average."""
        def average_score(metrics: List[MetricResult]) -> float:
            if not metrics:
                return 0.0
            return sum(m.score for m in metrics) / len(metrics)

        structured_avg = average_score(structured_metrics)
        semantic_avg = average_score(semantic_metrics)
        retrieval_avg = average_score(retrieval_metrics)
        hallucination_avg = average_score(hallucination_metrics)

        weighted_score = (
            self.weights.get("structured", 0.0) * structured_avg +
            self.weights.get("semantic", 0.0) * semantic_avg +
            self.weights.get("retrieval", 0.0) * retrieval_avg +
            self.weights.get("hallucination", 0.0) * hallucination_avg
        )

        return MetricResult(
            metric_name="hybrid_weighted",
            score=weighted_score,
            passed=weighted_score >= 0.6,
            details={
                "structured_avg": structured_avg,
                "semantic_avg": semantic_avg,
                "retrieval_avg": retrieval_avg,
                "hallucination_avg": hallucination_avg,
                "weights": self.weights,
            },
        )


# -----------------------------------------------------------------------------
# 6. Main Orchestrator: AccuracyMetrics
# -----------------------------------------------------------------------------
class AccuracyMetrics:
    """
    Main orchestrator for evaluating the RAG pipeline against benchmark.
    Loads benchmark, runs pipeline to get answers and contexts,
    evaluates each question, and generates reports.
    """

    def __init__(
        self,
        benchmark_path: Path,
        output_dir: Path,
        parallel: bool = True,
        max_workers: int = 4,
        sample_size: Optional[int] = None,
        verbose: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        self.parallel = parallel
        # Limit workers to CPU count
        cpu_count = os.cpu_count() or 1
        self.max_workers = min(max_workers, cpu_count * 2)  # allow some oversubscription
        if self.max_workers < 1:
            self.max_workers = 1
        self.sample_size = sample_size
        self.verbose = verbose

        # Initialize evaluators
        self.structured_evaluator = StructuredAccuracyEvaluator()
        self.semantic_evaluator = SemanticAccuracyEvaluator(model_name=model_name)
        self.retrieval_evaluator = RetrievalEvaluator(k=5)
        self.hallucination_evaluator = HallucinationEvaluator()
        self.hybrid_evaluator = HybridEvaluator()

        # RAG components (lazy initialization)
        self._retriever: Optional[HybridRetriever] = None
        self._answer_generator: Optional[AnswerGenerator] = None

        # Results storage
        self.records: List[EvaluationRecord] = []
        self.summary: Dict[str, Any] = {}

        # Setup logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def _init_rag_components(self) -> None:
        """Lazy initialization of RAG components."""
        if self._retriever is None:
            logger.info("Initializing HybridRetriever...")
            try:
                self._retriever = HybridRetriever()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize HybridRetriever: {e}") from e
        if self._answer_generator is None:
            logger.info("Initializing AnswerGenerator...")
            try:
                self._answer_generator = AnswerGenerator()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize AnswerGenerator: {e}") from e

    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load benchmark questions from JSON file."""
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

    def evaluate_question(self, question_data: Dict[str, Any]) -> EvaluationRecord:
        """Evaluate a single question by running the RAG pipeline and computing metrics."""
        question_id = question_data.get("id", "unknown")
        question = question_data.get("question", "")
        query_type = question_data.get("query_type", "structured")
        expected_answer = question_data.get("expected_answer")
        expected_keywords = question_data.get("expected_keywords", [])
        ground_truth = question_data.get("ground_truth", {})

        # Run RAG pipeline to get retrieved context and generated answer
        self._init_rag_components()
        retrieved_chunks = []
        generated_answer = ""
        try:
            # Retrieve context
            # The retriever may need a plan, but we'll call a simple retrieve method
            # We'll assume retrieve(question, query_type) returns dict with 'chunks'
            retrieval_result = self._retriever.retrieve(question, query_type=query_type)
            retrieved_chunks = retrieval_result.get("chunks", [])
            # Generate answer
            generated_answer = self._answer_generator.generate(
                question=question,
                context=retrieved_chunks,
                query_type=query_type,
            )
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            # Fallback to empty values
            retrieved_chunks = []
            generated_answer = ""

        # Evaluate structured metrics (if applicable)
        structured_metrics = []
        if query_type in ("structured", "hybrid"):
            structured_metrics = self.structured_evaluator.evaluate(
                expected=expected_answer,
                generated=generated_answer,
            )

        # Semantic metrics
        semantic_metrics = []
        if query_type in ("semantic", "hybrid"):
            semantic_metrics = self.semantic_evaluator.evaluate(
                expected=str(expected_answer) if expected_answer is not None else "",
                generated=generated_answer,
                expected_keywords=expected_keywords,
            )

        # Retrieval metrics
        retrieval_metrics = []
        # Ground truth relevant chunks might be in ground_truth under "relevant_chunks"
        relevant_chunks = ground_truth.get("relevant_chunks", [])
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=relevant_chunks,
            answer=generated_answer,
        )

        # Hallucination metrics
        hallucination_metrics = self.hallucination_evaluator.evaluate(
            generated_answer=generated_answer,
            retrieved_context=retrieved_chunks,
        )

        # Hybrid weighted score
        hybrid_metric = self.hybrid_evaluator.evaluate(
            structured_metrics=structured_metrics,
            semantic_metrics=semantic_metrics,
            retrieval_metrics=retrieval_metrics,
            hallucination_metrics=hallucination_metrics,
        )

        # Combine all metrics
        all_metrics = structured_metrics + semantic_metrics + retrieval_metrics + hallucination_metrics + [hybrid_metric]

        # Compute overall score (use hybrid as overall)
        overall_score = hybrid_metric.score

        return EvaluationRecord(
            question_id=question_id,
            question=question,
            query_type=query_type,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            retrieved_context=retrieved_chunks,
            metrics=all_metrics,
            overall_score=overall_score,
        )

    def evaluate_batch(self, questions: List[Dict[str, Any]]) -> List[EvaluationRecord]:
        """Evaluate multiple questions, optionally in parallel."""
        records = []
        if self.parallel and len(questions) > 1:
            logger.info(f"Evaluating {len(questions)} questions in parallel with {self.max_workers} workers.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_q = {executor.submit(self.evaluate_question, q): q for q in questions}
                for future in tqdm(as_completed(future_to_q), total=len(questions), desc="Evaluating"):
                    try:
                        record = future.result()
                        records.append(record)
                    except Exception as e:
                        logger.error(f"Error in parallel evaluation: {e}")
        else:
            logger.info("Evaluating questions sequentially.")
            for q in tqdm(questions, desc="Evaluating"):
                try:
                    record = self.evaluate_question(q)
                    records.append(record)
                except Exception as e:
                    logger.error(f"Error evaluating question {q.get('id', 'unknown')}: {e}")
        return records

    def generate_report(self, records: List[EvaluationRecord]) -> Dict[str, Any]:
        """Generate summary report from evaluation records."""
        if not records:
            return {"summary": {"total": 0}, "results": []}

        total = len(records)
        overall_scores = [r.overall_score for r in records]
        avg_overall = np.mean(overall_scores) if overall_scores else 0.0

        type_counts = {}
        for r in records:
            type_counts[r.query_type] = type_counts.get(r.query_type, 0) + 1

        # Average scores per metric category
        metric_categories = {
            "structured": ["exact_match", "numeric_error", "relative_error", "aggregation_accuracy", "top_k_match",
                           "range_accuracy", "distribution_similarity"],
            "semantic": ["bleu", "rouge_l", "bertscore", "embedding_similarity", "keyword_recall"],
            "retrieval": ["precision_at_k", "recall_at_k", "mrr", "hit_rate", "context_coverage"],
            "hallucination": ["support_score", "unsupported_claims", "faithfulness"],
        }
        category_avgs = {}
        for cat, names in metric_categories.items():
            scores = []
            for r in records:
                for m in r.metrics:
                    if m.metric_name in names:
                        scores.append(m.score)
            if scores:
                category_avgs[cat] = np.mean(scores)
            else:
                category_avgs[cat] = 0.0

        summary = {
            "total": total,
            "average_overall_score": float(avg_overall),
            "type_counts": type_counts,
            "category_averages": category_avgs,
            "timestamp": datetime.now().isoformat(),
        }

        results = [r.to_dict() for r in records]

        return {"summary": summary, "results": results}

    def save_report(self, report: Dict[str, Any]) -> None:
        """Save report to JSON files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full report
        report_path = self.output_dir / "accuracy_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Full report saved to {report_path}")

        # Summary only
        summary_path = self.output_dir / "accuracy_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report["summary"], f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")

    def run(self) -> None:
        """Run the full evaluation pipeline."""
        start_time = time.perf_counter()
        logger.info("Starting accuracy evaluation.")

        # Load benchmark
        questions = self.load_benchmark()
        if not questions:
            logger.warning("No questions to evaluate.")
            return

        # Evaluate
        self.records = self.evaluate_batch(questions)

        # Generate report
        report = self.generate_report(self.records)
        self.summary = report["summary"]

        # Save report
        self.save_report(report)

        elapsed = time.perf_counter() - start_time
        logger.info(f"Evaluation completed in {elapsed:.2f}s.")
        logger.info(f"Processed {len(self.records)} questions.")
        logger.info(f"Overall average score: {self.summary.get('average_overall_score', 0):.3f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline accuracy.")
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
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    evaluator = AccuracyMetrics(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        parallel=args.parallel,
        max_workers=args.workers,
        sample_size=args.sample_size,
        verbose=args.verbose,
        model_name=args.model,
    )

    try:
        evaluator.run()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()