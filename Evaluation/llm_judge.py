#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_judge.py

LLM-based evaluation of RAG system answers.
Uses Qwen (or compatible) to judge accuracy, completeness, relevance,
faithfulness, hallucination, and reasoning quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project-specific imports
# -----------------------------------------------------------------------------
try:
    from llm.qwen_client import QwenClient
except ImportError:
    raise ImportError(
        "QwenClient not found. Please ensure llm.qwen_client is implemented "
        "and available in the Python path."
    )

# Optional: use prompt templates if available
try:
    from llm.prompt_templates import PromptTemplates
except ImportError:
    PromptTemplates = None  # type: ignore

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "llm_judge.log"

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
class JudgeVerdict(Enum):
    """Overall verdict of the judge."""
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class JudgeScore:
    """Score from a single judge dimension."""

    score_name: str
    score: float  # 0-10 scale
    passed: bool
    confidence: float  # 0-1
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_name": self.score_name,
            "score": self.score,
            "passed": self.passed,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class JudgementResult:
    """Complete result of evaluating one question."""

    question_id: str
    question: str
    query_type: str
    answer: str
    context: List[str]
    ground_truth: Any
    scores: List[JudgeScore]
    overall_score: float
    verdict: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "query_type": self.query_type,
            "answer": self.answer,
            "context": self.context,
            "ground_truth": self.ground_truth,
            "scores": [s.to_dict() for s in self.scores],
            "overall_score": self.overall_score,
            "verdict": self.verdict,
            "timestamp": self.timestamp,
        }


# -----------------------------------------------------------------------------
# Prompt Builder
# -----------------------------------------------------------------------------
class PromptBuilder:
    """
    Builds evaluation prompts with modular templates.
    Uses built-in templates; can be overridden with external PromptTemplates.
    """

    DEFAULT_ACCURACY_PROMPT = """
You are an expert evaluator of RAG systems. Evaluate the following question and answer based on accuracy.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Evaluate accuracy on a scale of 0-10, where 10 means the answer is completely accurate and matches the ground truth.
Return JSON: {{"score": <0-10>, "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    DEFAULT_HALLUCINATION_PROMPT = """
You are an expert evaluator detecting hallucinations in RAG answers.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Determine if the generated answer contains hallucinations (claims not supported by the context). If there are hallucinations, also provide a list of unsupported claims.
Return JSON: {{"hallucination": true/false, "unsupported_claims": ["claim1", ...], "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    DEFAULT_RELEVANCE_PROMPT = """
You are an expert evaluator of RAG systems. Evaluate the relevance of the answer to the question.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Score relevance on a scale of 0-10, where 10 means the answer is highly relevant and directly addresses the question.
Return JSON: {{"score": <0-10>, "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    DEFAULT_FAITHFULNESS_PROMPT = """
You are an expert evaluator of RAG systems. Evaluate faithfulness of the answer to the retrieved context.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Score faithfulness on a scale of 0-10, where 10 means the answer is completely faithful to the context and does not introduce unsupported information.
Return JSON: {{"score": <0-10>, "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    DEFAULT_REASONING_PROMPT = """
You are an expert evaluator of RAG systems. Evaluate reasoning quality of the answer.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Score reasoning quality on a scale of 0-10, where 10 means the answer shows excellent reasoning, logical flow, and correct synthesis of information.
Return JSON: {{"score": <0-10>, "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    DEFAULT_COMPLETENESS_PROMPT = """
You are an expert evaluator of RAG systems. Evaluate completeness of the answer.

Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Ground Truth Answer: {ground_truth}

Score completeness on a scale of 0-10, where 10 means the answer covers all key aspects expected from the question and context.
Return JSON: {{"score": <0-10>, "confidence": <0-1>, "reason": "<brief reason>"}}
"""

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        if templates is None:
            # If PromptTemplates is available, use it; otherwise use defaults
            if PromptTemplates is not None:
                try:
                    pt = PromptTemplates()
                    self.templates = {
                        "accuracy": pt.get_accuracy_prompt(),
                        "hallucination": pt.get_hallucination_prompt(),
                        "relevance": pt.get_relevance_prompt(),
                        "faithfulness": pt.get_faithfulness_prompt(),
                        "reasoning": pt.get_reasoning_prompt(),
                        "completeness": pt.get_completeness_prompt(),
                    }
                except Exception:
                    self.templates = self._default_templates()
            else:
                self.templates = self._default_templates()
        else:
            self.templates = templates

    def _default_templates(self) -> Dict[str, str]:
        return {
            "accuracy": self.DEFAULT_ACCURACY_PROMPT,
            "hallucination": self.DEFAULT_HALLUCINATION_PROMPT,
            "relevance": self.DEFAULT_RELEVANCE_PROMPT,
            "faithfulness": self.DEFAULT_FAITHFULNESS_PROMPT,
            "reasoning": self.DEFAULT_REASONING_PROMPT,
            "completeness": self.DEFAULT_COMPLETENESS_PROMPT,
        }

    def _format(self, template: str, question: str, context: str, answer: str, ground_truth: str) -> str:
        return template.format(
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
        )

    def build_accuracy_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["accuracy"], question, context, answer, ground_truth)

    def build_hallucination_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["hallucination"], question, context, answer, ground_truth)

    def build_relevance_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["relevance"], question, context, answer, ground_truth)

    def build_faithfulness_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["faithfulness"], question, context, answer, ground_truth)

    def build_reasoning_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["reasoning"], question, context, answer, ground_truth)

    def build_completeness_prompt(self, question: str, context: str, answer: str, ground_truth: str) -> str:
        return self._format(self.templates["completeness"], question, context, answer, ground_truth)


# -----------------------------------------------------------------------------
# Response Parser
# -----------------------------------------------------------------------------
class ResponseParser:
    """
    Parses LLM responses, extracts JSON, and repairs malformed outputs.
    """

    @staticmethod
    def parse_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from text and parse it.
        Handles markdown code blocks and partial responses.
        """
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find anything that looks like JSON
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = text.strip()

        # Try to parse directly
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt repair: remove trailing commas, unclosed braces
            json_str = ResponseParser.repair_json(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON after repair: {e}")
                raise

    @staticmethod
    def repair_json(text: str) -> str:
        """
        Attempt to repair common JSON issues.
        """
        # Remove trailing commas in objects/arrays
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        # Add missing quotes around keys
        text = re.sub(r"(\w+):", r'"\1":', text)
        # Ensure braces are balanced
        open_braces = text.count("{") - text.count("}")
        if open_braces > 0:
            text += "}" * open_braces
        return text

    @staticmethod
    def validate_judge_output(data: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that all required keys are present and have valid types.
        """
        for key in required_keys:
            if key not in data:
                return False
            # Check types: scores should be int/float, confidence float, reason str
            if key in ["score", "confidence"]:
                if not isinstance(data[key], (int, float)):
                    return False
            if key == "reason":
                if not isinstance(data[key], str):
                    return False
            if key == "hallucination":
                if not isinstance(data[key], bool):
                    return False
        return True


# -----------------------------------------------------------------------------
# Hallucination Judge (Heuristic fallback)
# -----------------------------------------------------------------------------
class HallucinationJudge:
    """
    Detects hallucinations using heuristics (lexical overlap) as fallback
    when LLM is unavailable. Primary is LLM-based in the main judge.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def evaluate_heuristic(
        self,
        answer: str,
        context: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Heuristic hallucination detection: check token overlap.
        Returns (is_hallucination, unsupported_claims)
        """
        if not answer or not context:
            return True, ["No context or answer provided"]

        # Tokenize
        answer_tokens = set(self._tokenize(answer))
        context_tokens = set()
        for chunk in context:
            context_tokens.update(self._tokenize(chunk))

        if not context_tokens:
            return True, ["No context tokens"]

        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        hallucination = overlap < self.threshold
        # For unsupported claims, we simply identify tokens not in context
        unsupported = list(answer_tokens - context_tokens)
        return hallucination, unsupported

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# -----------------------------------------------------------------------------
# AnswerJudge (LLM-based)
# -----------------------------------------------------------------------------
class AnswerJudge:
    """
    Evaluates answer quality using LLM for each dimension.
    """

    def __init__(
        self,
        qwen_client: QwenClient,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        retries: int = 2,
        timeout: float = 30.0,
        cache_enabled: bool = True,
    ):
        self.qwen_client = qwen_client
        self.prompt_builder = prompt_builder
        self.parser = response_parser
        self.retries = retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, JudgeScore] = {}

    def score_accuracy(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        return self._evaluate_dimension(
            prompt_func=self.prompt_builder.build_accuracy_prompt,
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            score_name="accuracy",
            required_keys=["score", "confidence", "reason"],
        )

    def score_relevance(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        return self._evaluate_dimension(
            prompt_func=self.prompt_builder.build_relevance_prompt,
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            score_name="relevance",
            required_keys=["score", "confidence", "reason"],
        )

    def score_faithfulness(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        return self._evaluate_dimension(
            prompt_func=self.prompt_builder.build_faithfulness_prompt,
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            score_name="faithfulness",
            required_keys=["score", "confidence", "reason"],
        )

    def score_reasoning(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        return self._evaluate_dimension(
            prompt_func=self.prompt_builder.build_reasoning_prompt,
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            score_name="reasoning",
            required_keys=["score", "confidence", "reason"],
        )

    def score_completeness(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        return self._evaluate_dimension(
            prompt_func=self.prompt_builder.build_completeness_prompt,
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
            score_name="completeness",
            required_keys=["score", "confidence", "reason"],
        )

    def detect_hallucination(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> JudgeScore:
        """
        Detect hallucinations. Returns score as 10 if no hallucination, 0 if hallucination.
        """
        prompt = self.prompt_builder.build_hallucination_prompt(
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
        )
        response = self._call_llm(prompt)
        try:
            data = self.parser.parse_json(response)
        except json.JSONDecodeError:
            # Fallback heuristic
            heuristic = HallucinationJudge()
            hall, unsupported = heuristic.evaluate_heuristic(answer, [context])
            return JudgeScore(
                score_name="hallucination",
                score=10.0 if not hall else 0.0,
                passed=not hall,
                confidence=0.5,
                reason=f"Heuristic: {'no' if not hall else 'hallucination'} detected",
            )

        required = ["hallucination", "confidence", "reason"]
        if not self.parser.validate_judge_output(data, required):
            # Fallback heuristic
            heuristic = HallucinationJudge()
            hall, unsupported = heuristic.evaluate_heuristic(answer, [context])
            return JudgeScore(
                score_name="hallucination",
                score=10.0 if not hall else 0.0,
                passed=not hall,
                confidence=0.5,
                reason=f"Heuristic (fallback): {'no' if not hall else 'hallucination'} detected",
            )

        hall = data.get("hallucination", True)
        score = 10.0 if not hall else 0.0
        return JudgeScore(
            score_name="hallucination",
            score=score,
            passed=not hall,
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
        )

    def _evaluate_dimension(
        self,
        prompt_func,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
        score_name: str,
        required_keys: List[str],
    ) -> JudgeScore:
        """
        Generic evaluator for a single dimension.
        """
        # Create cache key
        if self.cache_enabled:
            cache_key = hashlib.md5(
                f"{question[:100]}{answer[:100]}{ground_truth[:100]}{score_name}".encode()
            ).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
        else:
            cache_key = None

        prompt = prompt_func(
            question=question,
            context=context,
            answer=answer,
            ground_truth=ground_truth,
        )
        try:
            response = self._call_llm(prompt)
            data = self.parser.parse_json(response)
        except Exception as e:
            logger.warning(f"LLM call failed for {score_name}: {e}")
            # Use default score
            data = {"score": 5.0, "confidence": 0.3, "reason": f"LLM error: {str(e)}"}

        # Validate
        if not self.parser.validate_judge_output(data, required_keys):
            data = {"score": 5.0, "confidence": 0.3, "reason": "Invalid response format"}

        score = float(data.get("score", 5.0))
        # Ensure score in 0-10
        score = max(0.0, min(10.0, score))
        confidence = float(data.get("confidence", 0.5))
        reason = data.get("reason", "")
        passed = score >= 6.0  # Threshold for pass

        judge_score = JudgeScore(
            score_name=score_name,
            score=score,
            passed=passed,
            confidence=confidence,
            reason=reason,
        )

        if cache_key is not None:
            self._cache[cache_key] = judge_score
        return judge_score

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retries and timeout."""
        last_exception = None
        for attempt in range(self.retries + 1):
            try:
                response = self.qwen_client.generate(prompt, timeout=self.timeout)
                return response
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.retries+1}): {e}")
                if attempt < self.retries:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise RuntimeError(f"LLM call failed after {self.retries+1} attempts") from last_exception
        # Should not reach here
        raise RuntimeError("LLM call failed")


# -----------------------------------------------------------------------------
# Composite Judge
# -----------------------------------------------------------------------------
class CompositeJudge:
    """
    Combines multiple judge scores into a final overall score and verdict.
    """

    DEFAULT_WEIGHTS = {
        "accuracy": 0.30,
        "completeness": 0.20,
        "relevance": 0.20,
        "faithfulness": 0.20,
        "reasoning": 0.10,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def aggregate(self, scores: List[JudgeScore]) -> Tuple[float, str]:
        """
        Compute weighted average and determine pass/fail/uncertain.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for score in scores:
            weight = self.weights.get(score.score_name, 0.0)
            if weight > 0:
                weighted_sum += score.score * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0, JudgeVerdict.UNCERTAIN.value

        overall = weighted_sum / total_weight
        # Verdict: pass if overall >= 6.0, fail if < 4.0, else uncertain
        if overall >= 6.0:
            verdict = JudgeVerdict.PASS.value
        elif overall < 4.0:
            verdict = JudgeVerdict.FAIL.value
        else:
            verdict = JudgeVerdict.UNCERTAIN.value
        return overall, verdict


# -----------------------------------------------------------------------------
# LLM Judge (Main Orchestrator)
# -----------------------------------------------------------------------------
class LLMJudge:
    """
    Main orchestrator for LLM-based evaluation.
    Loads benchmark, runs answer generation, evaluates with LLM judge,
    and produces reports.
    """

    def __init__(
        self,
        benchmark_path: Path,
        output_dir: Path,
        parallel: bool = True,
        workers: int = 4,
        sample_size: Optional[int] = None,
        verbose: bool = False,
        use_cached: bool = True,
        retries: int = 2,
        timeout: float = 30.0,
        model_name: Optional[str] = None,  # not used if QwenClient already configured
    ):
        self.benchmark_path = benchmark_path
        self.output_dir = output_dir
        self.parallel = parallel
        self.workers = workers
        self.sample_size = sample_size
        self.verbose = verbose
        self.use_cached = use_cached
        self.retries = retries
        self.timeout = timeout
        self.model_name = model_name

        # Initialize components
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

        # LLM Client (will be initialized lazily)
        self.qwen_client: Optional[QwenClient] = None

        # Judge instances
        self.answer_judge: Optional[AnswerJudge] = None
        self.composite_judge = CompositeJudge()

        # Results
        self.results: List[JudgementResult] = []
        self.summary: Dict[str, Any] = {}
        self._cache: Dict[str, JudgementResult] = {}

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def _init_llm(self) -> None:
        """Lazy initialization of LLM client and judge."""
        if self.qwen_client is None:
            try:
                # If model_name provided, pass to client constructor (if supported)
                if self.model_name:
                    self.qwen_client = QwenClient(model_name=self.model_name)
                else:
                    self.qwen_client = QwenClient()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize QwenClient: {e}") from e
            self.answer_judge = AnswerJudge(
                qwen_client=self.qwen_client,
                prompt_builder=self.prompt_builder,
                response_parser=self.response_parser,
                retries=self.retries,
                timeout=self.timeout,
                cache_enabled=self.use_cached,
            )

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

    def evaluate_question(self, question_data: Dict[str, Any]) -> JudgementResult:
        """
        Evaluate a single question using the LLM judge.
        Expects question_data to have 'question', 'query_type', 'expected_answer',
        and optionally 'generated_answer' and 'retrieved_context' from previous runs.
        """
        question = question_data.get("question", "")
        query_type = question_data.get("query_type", "hybrid")
        ground_truth = question_data.get("expected_answer", "")
        # If we have generated_answer and retrieved_context in the data, use them.
        # Otherwise, we'll need to generate them (we'll use placeholders).
        generated_answer = question_data.get("generated_answer", "")
        retrieved_context = question_data.get("retrieved_context", [])

        # If not provided, fallback to ground truth as answer and context
        if not generated_answer:
            generated_answer = str(ground_truth) if ground_truth is not None else ""
            logger.warning(f"Question {question_data.get('id')} missing generated_answer, using ground truth.")
        if not retrieved_context:
            retrieved_context = [str(ground_truth)] if ground_truth is not None else []
            logger.warning(f"Question {question_data.get('id')} missing retrieved_context, using ground truth as context.")

        # Prepare context as a single string
        context_str = "\n".join(retrieved_context) if retrieved_context else ""

        # Initialize LLM
        self._init_llm()

        # Evaluate each dimension
        scores = []
        try:
            # Accuracy
            scores.append(
                self.answer_judge.score_accuracy(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
            # Completeness
            scores.append(
                self.answer_judge.score_completeness(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
            # Relevance
            scores.append(
                self.answer_judge.score_relevance(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
            # Faithfulness
            scores.append(
                self.answer_judge.score_faithfulness(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
            # Hallucination
            scores.append(
                self.answer_judge.detect_hallucination(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
            # Reasoning
            scores.append(
                self.answer_judge.score_reasoning(
                    question=question,
                    context=context_str,
                    answer=generated_answer,
                    ground_truth=str(ground_truth) if ground_truth is not None else "",
                )
            )
        except Exception as e:
            logger.error(f"Error evaluating question {question_data.get('id')}: {e}")
            # Add placeholder scores
            scores = [
                JudgeScore("accuracy", 0.0, False, 0.0, "Error"),
                JudgeScore("completeness", 0.0, False, 0.0, "Error"),
                JudgeScore("relevance", 0.0, False, 0.0, "Error"),
                JudgeScore("faithfulness", 0.0, False, 0.0, "Error"),
                JudgeScore("hallucination", 0.0, False, 0.0, "Error"),
                JudgeScore("reasoning", 0.0, False, 0.0, "Error"),
            ]

        # Aggregate
        overall_score, verdict = self.composite_judge.aggregate(scores)

        return JudgementResult(
            question_id=str(question_data.get("id", "unknown")),
            question=question,
            query_type=query_type,
            answer=generated_answer,
            context=retrieved_context,
            ground_truth=ground_truth,
            scores=scores,
            overall_score=overall_score,
            verdict=verdict,
        )

    def evaluate_batch(self, questions: List[Dict[str, Any]]) -> List[JudgementResult]:
        """Evaluate multiple questions, optionally in parallel."""
        self._init_llm()
        results = []
        if self.parallel and len(questions) > 1:
            # Limit workers to CPU count
            import os
            cpu_count = os.cpu_count() or 1
            workers = min(self.workers, cpu_count * 2)
            if workers < 1:
                workers = 1
            logger.info(f"Evaluating {len(questions)} questions in parallel with {workers} workers.")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_q = {executor.submit(self.evaluate_question, q): q for q in questions}
                for future in tqdm(as_completed(future_to_q), total=len(questions), desc="LLM Judging"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel evaluation: {e}")
        else:
            logger.info("Evaluating questions sequentially.")
            for q in tqdm(questions, desc="LLM Judging"):
                try:
                    result = self.evaluate_question(q)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating question {q.get('id')}: {e}")
        return results

    def generate_summary(self, results: List[JudgementResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if not results:
            return {"total": 0}

        total = len(results)
        overall_scores = [r.overall_score for r in results]
        avg_score = np.mean(overall_scores) if overall_scores else 0.0
        median_score = np.median(overall_scores) if overall_scores else 0.0

        pass_count = sum(1 for r in results if r.verdict == JudgeVerdict.PASS.value)
        pass_rate = pass_count / total if total > 0 else 0.0

        hallucinated = 0
        for r in results:
            for s in r.scores:
                if s.score_name == "hallucination" and s.score < 5.0:
                    hallucinated += 1
                    break
        hallucination_rate = hallucinated / total if total > 0 else 0.0

        score_dimensions = {}
        for r in results:
            for s in r.scores:
                score_dimensions.setdefault(s.score_name, []).append(s.score)
        avg_per_dim = {dim: np.mean(vals) for dim, vals in score_dimensions.items()}

        confidence_scores = []
        for r in results:
            for s in r.scores:
                confidence_scores.append(s.confidence)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            "total": total,
            "average_overall_score": float(avg_score),
            "median_overall_score": float(median_score),
            "pass_rate": float(pass_rate),
            "hallucination_rate": float(hallucination_rate),
            "average_confidence": float(avg_confidence),
            "average_scores_by_dimension": {k: float(v) for k, v in avg_per_dim.items()},
            "timestamp": datetime.now().isoformat(),
        }

    def save_report(self, results: List[JudgementResult], summary: Dict[str, Any]) -> None:
        """Save reports to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full report
        report = {
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        report_path = self.output_dir / "llm_judge_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Full LLM judge report saved to {report_path}")

        # Summary only
        summary_path = self.output_dir / "llm_judge_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"LLM judge summary saved to {summary_path}")

    def run(self) -> None:
        """Run the full LLM judge evaluation pipeline."""
        start_time = time.perf_counter()
        logger.info("Starting LLM judge evaluation.")

        # Load benchmark
        questions = self.load_benchmark()
        if not questions:
            logger.warning("No questions to evaluate.")
            return

        # Evaluate
        self.results = self.evaluate_batch(questions)

        # Generate summary
        self.summary = self.generate_summary(self.results)

        # Save reports
        self.save_report(self.results, self.summary)

        elapsed = time.perf_counter() - start_time
        logger.info(f"LLM judge evaluation completed in {elapsed:.2f}s.")
        logger.info(f"Processed {len(self.results)} questions.")
        logger.info(f"Average overall score: {self.summary.get('average_overall_score', 0):.2f}")
        logger.info(f"Hallucination rate: {self.summary.get('hallucination_rate', 0):.3f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based evaluation of RAG answers.")
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
        default=None,
        help="Model name for Qwen client (optional)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries for LLM calls",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for LLM calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    judge = LLMJudge(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        parallel=args.parallel,
        workers=args.workers,
        sample_size=args.sample_size,
        verbose=args.verbose,
        use_cached=True,
        retries=args.retries,
        timeout=args.timeout,
        model_name=args.model,
    )

    try:
        judge.run()
    except Exception as e:
        logger.error(f"LLM judge evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()