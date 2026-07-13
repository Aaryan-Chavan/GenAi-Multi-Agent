"""
llm/qwen_client.py
==================
Enterprise-grade central inference engine for the multi-agent RAG system.

Wraps a HuggingFace causal language model (default: Qwen3-14B-Instruct) with:
  - Thread-safe singleton lifecycle management
  - Structured logging & custom exceptions
  - LRU prompt cache with hit/miss statistics
  - Automatic context budget management & graceful truncation
  - Streaming generation (TextIteratorStreamer)
  - Batch generation
  - Async-compatible APIs (asyncio executor bridge)
  - Configurable retry manager with OOM recovery
  - Model warmup at startup
  - Token budget utilities for RAG pipelines
  - Response validation layer
  - ConversationMemory helper
  - Per-request UUID tracking
  - Extended performance metrics
  - Prompt SHA-256 hashing (never logs raw prompts in plaintext)
  - Safety / jailbreak detection hook (extensible via subclass)
  - Automatic periodic GPU memory optimisation
  - Lifecycle hooks (before/after generation, on_error, …)
  - Diagnostics export (JSON / CSV / YAML / Markdown)
  - Telemetry collector
  - Model version report in diagnostics
  - RAG integration convenience helpers
  - Benchmark helpers with warmup and standard-deviation reporting
  - Health monitoring with four-tier classification
  - Context manager & cleanup/reload support
  - Redis-backed result cache (RedisResultCache) with transparent
    fallback to in-memory when Redis is unavailable or not configured
  - Pluggable ``ModelLoader`` strategy (see ``ModelLoader`` /
    ``HFAutoModelLoader`` below) isolating the one place a concrete model
    family is chosen, so swapping the local backend (a different HF causal
    LM family, a quantised loader, etc.) never requires touching this
    class or any of its callers

Verified callers in this codebase: ``answer_generator.py`` constructs a
``QwenClient`` and calls ``generate(prompt=...)``, and optionally
``hash_prompt()`` / ``health_check()`` / ``diagnostics()``.
``hybrid_agent.py`` never imports this module directly — it only talks to
``answer_generator.py``. ``structured_agent.py`` does NOT use this client;
it makes its own independent HTTP calls to a local Ollama server for
NL-to-SQL generation, entirely bypassing this file.

Architecture note: this client performs LOCAL in-process inference via
`transformers` (`model.generate()` on GPU/CPU) — there is no network
socket anywhere in the request path. Retry/backoff logic here exists for
CUDA OOM recovery and result-cache-backend failures, not for HTTP
transport, since there is no HTTP transport to retry.

Public API stability: all public methods, return types, and exception
classes are preserved across revisions. Internal helpers (prefixed ``_``)
are not part of the stable contract.

Author : Senior AI Infrastructure Engineer
Python : 3.11+
Style  : Google-style docstrings, PEP 8, PEP 257
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import asyncio
import bisect
import csv
import gc
import hashlib
import io
import json
import logging
import os
import platform
import queue
import statistics
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Generator, Iterator, Optional, Union, AsyncIterator
import copy
# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    __version__ as _transformers_version,
)

# Optional third-party deps used only by the extended diagnostics / metrics
# surface. Both are soft dependencies: their absence degrades gracefully
# (CPU/RAM metrics report as unavailable; YAML export falls back to a
# minimal built-in dumper) rather than raising an ImportError at module
# load time, since neither is required for core inference.
try:
    import psutil  # type: ignore[import]

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

try:
    import yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

# Redis is a soft dependency: the RedisResultCache backend requires it, but
# the rest of the module falls back to InMemoryResultCache when it is absent.
try:
    import redis  # type: ignore[import]
    from redis.exceptions import RedisError  # type: ignore[import]

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment,misc]
    _REDIS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
from Config import settings

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(_handler)


# ===========================================================================
# Custom Exceptions
# ===========================================================================


class ModelLoadError(RuntimeError):
    """Raised when the causal LM cannot be loaded from disk or the HuggingFace Hub.

    Common causes: model path not found, insufficient disk space, network
    timeout during download, or incompatible transformers version.
    """


class TokenizerLoadError(RuntimeError):
    """Raised when the tokenizer cannot be initialised.

    Common causes: tokenizer path not found, missing ``tokenizer.json``,
    or incompatible tokenizer class for the model family.
    """


class GenerationError(RuntimeError):
    """Raised when text generation fails for any reason other than CUDA OOM.

    Common causes: invalid generation config, model forward-pass failure,
    or an unexpected exception inside ``model.generate()``.
    """


class ContextLengthError(ValueError):
    """Raised when the tokenised prompt meets or exceeds the model's context window.

    Callers should reduce prompt size or truncate retrieved context before
    retrying. Use :meth:`QwenClient.context_budget_report` for diagnosis.
    """


class GPUOutOfMemoryError(RuntimeError):
    """Raised when a CUDA OOM event occurs and cannot be recovered.

    The client attempts a single cache-clear and retry before raising this
    exception. Callers may reduce ``max_new_tokens`` or the prompt size and
    retry.
    """


class InvalidPromptError(ValueError):
    """Raised when the caller supplies a prompt that fails basic validation.

    Triggers include: empty string, whitespace-only string, or a non-string
    value passed where a prompt is expected.
    """


class ResponseValidationError(ValueError):
    """Raised when the generated response fails quality validation.

    Not currently raised by default; available for callers that wish to treat
    validation failures as hard errors rather than logged warnings.
    """


class SafetyError(ValueError):
    """Raised when the prompt or response triggers the safety layer.

    Triggers include: prompt length exceeding ``settings.MAX_PROMPT_CHARS``,
    presence of null bytes, or a match against ``settings.JAILBREAK_FRAGMENTS``.
    """


class WarmupError(RuntimeError):
    """Raised when the model warmup sequence fails.

    Warmup failures do not prevent subsequent :meth:`QwenClient.generate`
    calls; only :attr:`QwenClient._warmup_complete` will remain ``False``.
    """


class InvalidGenerationParameterError(ValueError):
    """Raised by :meth:`QwenClient.validate_generation_parameters` when a
    sampling parameter (temperature, top_p, top_k, repetition_penalty, or
    max_new_tokens) falls outside its valid range or has the wrong type.
    """


class StreamCancelledError(RuntimeError):
    """Raised (and immediately swallowed by the caller as a clean stop) when
    a streaming generation is cancelled via its ``cancel_event``.
    """


class CacheBackendError(RuntimeError):
    """Raised when a :class:`ResultCacheBackend` implementation fails on
    ``get``/``put``/``delete``. Caching failures never abort generation;
    callers catch this internally and treat it as a cache miss.
    """


# ===========================================================================
# Enumerations
# ===========================================================================


class FinishReason(str, Enum):
    """Enumeration of possible generation stop reasons."""

    LENGTH = "length"
    EOS = "eos"
    UNKNOWN = "unknown"


class ValidationStatus(str, Enum):
    """Outcome codes returned by the response validation layer."""

    VALID = "valid"
    EMPTY = "empty"
    REPEATED = "repeated"
    PUNCTUATION_ONLY = "punctuation_only"
    HALLUCINATED_PREFIX = "hallucinated_prefix"
    MALFORMED_UNICODE = "malformed_unicode"
    OVERSIZED = "oversized"


class HookEvent(str, Enum):
    """Lifecycle hook event identifiers."""

    BEFORE_TOKENIZE = "before_tokenize"
    AFTER_TOKENIZE = "after_tokenize"
    BEFORE_GENERATION = "before_generation"
    AFTER_GENERATION = "after_generation"
    ON_ERROR = "on_error"
    BEFORE_RELOAD = "before_reload"
    AFTER_RELOAD = "after_reload"
    BEFORE_DECODE = "before_decode"
    AFTER_DECODE = "after_decode"
    ON_STREAM_TOKEN = "on_stream_token"
    ON_STREAM_CANCEL = "on_stream_cancel"
    ON_CACHE_HIT = "on_cache_hit"
    ON_CACHE_MISS = "on_cache_miss"


class HealthStatus(str, Enum):
    """Four-tier health classification, ordered from best to worst.

    Supersedes the original binary ``healthy``/``degraded`` status string
    used by :meth:`QwenClient.health_check`. The original two values are
    retained as members so existing string comparisons
    (``health["status"] == "healthy"``) continue to work unchanged; this
    enum only adds the two intermediate/extreme tiers requested by item 10
    of the production hardening spec (warning, critical).
    """

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ===========================================================================
# Dataclasses
# ===========================================================================


@dataclass
class GenerationResult:
    """Immutable result object returned by :meth:`QwenClient.generate`.

    Attributes:
        text: The decoded model output (assistant turn only).
        prompt_tokens: Number of tokens in the input prompt.
        generated_tokens: Number of tokens produced by the model.
        total_tokens: ``prompt_tokens + generated_tokens``.
        latency: Wall-clock seconds from tokenisation start to decode end.
        tokens_per_second: Throughput in generated tokens per second.
        finish_reason: One of the :class:`FinishReason` values.
        model: Model name / identifier string.
        device: Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        gpu_memory_used_mb: GPU memory allocated at result time (MiB).
        timestamp: ISO-8601 UTC timestamp of the generation.
        request_id: UUID assigned to this request.
        cache_hit: Whether the tokenisation result was served from cache.
        tokenize_latency: Seconds spent on tokenisation.
        generate_latency: Seconds spent in ``model.generate()``.
        decode_latency: Seconds spent on token decoding.
        gpu_memory_delta_mb: GPU memory change during generation (MiB).
        retry_count: Number of retries performed.
        validation_status: :class:`ValidationStatus` of the response.
        caller_metadata: Arbitrary dict supplied by the caller.
        thread_id: OS thread identifier of the calling thread.
        queue_wait: Seconds spent waiting for the generation lock before
            this request began processing. Previously computed inside
            :meth:`QwenClient.generate` but never attached to the result;
            now surfaced here (defaults to ``0.0`` for any code path that
            doesn't supply it, preserving backward compatibility).
        result_cache_hit: Whether this entire result (not just the
            tokenisation) was served from the result cache rather than a
            fresh inference call. Distinct from ``cache_hit``, which refers
            only to the tokenisation cache.
    """

    text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    latency: float
    tokens_per_second: float
    finish_reason: str
    model: str
    device: str
    gpu_memory_used_mb: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cache_hit: bool = False
    tokenize_latency: float = 0.0
    generate_latency: float = 0.0
    decode_latency: float = 0.0
    gpu_memory_delta_mb: float = 0.0
    retry_count: int = 0
    validation_status: str = ValidationStatus.VALID
    caller_metadata: dict[str, Any] = field(default_factory=dict)
    thread_id: int = field(default_factory=threading.get_ident)
    queue_wait: float = 0.0
    result_cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return a pretty-printed JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Cumulative, thread-safe performance counters for the inference engine.

    Attributes:
        total_requests: Total calls to :meth:`QwenClient.generate`.
        successful_requests: Calls that completed without exception.
        failed_requests: Calls that raised an exception.
        total_prompt_tokens: Aggregate prompt token count.
        total_generated_tokens: Aggregate generated token count.
        total_latency: Sum of per-request total latencies (seconds).
        average_latency: Rolling mean total latency (seconds).
        average_tokens_per_second: Rolling mean throughput.
        peak_gpu_memory: Maximum GPU allocation seen (MiB).
        longest_generation: Highest ``generated_tokens`` in a single call.
        shortest_generation: Lowest ``generated_tokens`` in a single call.
        uptime: Seconds since the client was instantiated.
        last_request_time: ISO-8601 UTC timestamp of the last request.
        total_tokenize_latency: Aggregate tokenisation time (seconds).
        total_generate_latency: Aggregate ``model.generate()`` time (seconds).
        total_decode_latency: Aggregate decode time (seconds).
        cache_hits: Number of tokenisation cache hits.
        cache_misses: Number of tokenisation cache misses.
        oom_count: Number of CUDA OOM events recovered.
        retry_count: Total retry attempts across all requests.
        streaming_count: Number of streaming generation calls.
        batch_count: Number of batch generation calls.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0
    average_tokens_per_second: float = 0.0
    peak_gpu_memory: float = 0.0
    longest_generation: int = 0
    shortest_generation: int = 0
    uptime: float = 0.0
    last_request_time: str = ""
    total_tokenize_latency: float = 0.0
    total_generate_latency: float = 0.0
    total_decode_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    oom_count: int = 0
    retry_count: int = 0
    streaming_count: int = 0
    batch_count: int = 0

    _start_time: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self, result: GenerationResult) -> None:
        """Update counters after a successful generation.

        Args:
            result: The :class:`GenerationResult` produced by the engine.
        """
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_prompt_tokens += result.prompt_tokens
            self.total_generated_tokens += result.generated_tokens
            self.total_latency += result.latency
            self.total_tokenize_latency += result.tokenize_latency
            self.total_generate_latency += result.generate_latency
            self.total_decode_latency += result.decode_latency
            self.average_latency = self.total_latency / self.successful_requests
            self.average_tokens_per_second = (
                self.total_generated_tokens / self.total_latency
                if self.total_latency > 0
                else 0.0
            )
            self.peak_gpu_memory = max(self.peak_gpu_memory, result.gpu_memory_used_mb)
            self.longest_generation = max(
                self.longest_generation, result.generated_tokens
            )
            if self.shortest_generation == 0:
                self.shortest_generation = result.generated_tokens
            else:
                self.shortest_generation = min(
                    self.shortest_generation, result.generated_tokens
                )
            self.last_request_time = result.timestamp
            self.retry_count += result.retry_count
            self.uptime = time.monotonic() - self._start_time
            if result.cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_failure(self) -> None:
        """Increment the failure counter and refresh uptime."""
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.uptime = time.monotonic() - self._start_time
            self.last_request_time = datetime.now(timezone.utc).isoformat()

    def record_oom(self) -> None:
        """Increment the OOM counter."""
        with self._lock:
            self.oom_count += 1

    def record_streaming(self) -> None:
        """Increment the streaming call counter."""
        with self._lock:
            self.streaming_count += 1

    def record_batch(self) -> None:
        """Increment the batch call counter."""
        with self._lock:
            self.batch_count += 1

    def reset(self) -> None:
        """Reset all counters and restart the uptime clock."""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.total_prompt_tokens = 0
            self.total_generated_tokens = 0
            self.total_latency = 0.0
            self.average_latency = 0.0
            self.average_tokens_per_second = 0.0
            self.peak_gpu_memory = 0.0
            self.longest_generation = 0
            self.shortest_generation = 0
            self.uptime = 0.0
            self.last_request_time = ""
            self.total_tokenize_latency = 0.0
            self.total_generate_latency = 0.0
            self.total_decode_latency = 0.0
            self.cache_hits = 0
            self.cache_misses = 0
            self.oom_count = 0
            self.retry_count = 0
            self.streaming_count = 0
            self.batch_count = 0
            self._start_time = time.monotonic()

    def summary(self) -> str:
        """Return a human-readable one-liner summary of key metrics."""
        return (
            f"requests={self.total_requests} "
            f"ok={self.successful_requests} "
            f"fail={self.failed_requests} "
            f"avg_latency={self.average_latency:.3f}s "
            f"avg_tps={self.average_tokens_per_second:.1f} "
            f"peak_gpu={self.peak_gpu_memory:.1f}MiB "
            f"cache_hits={self.cache_hits} "
            f"oom={self.oom_count}"
        )

    def to_dict(self):
        data = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                copy.deepcopy(v)  # validate safe serializability
                data[k] = v
            except Exception:
                data[k] = str(v)  # fallback for locks, etc.
        return data


# ---------------------------------------------------------------------------


class AdvancedMetrics:
    """Extended metrics collector covering percentile latency, system
    resource usage, and request-rate tracking (spec item 8).

    Deliberately kept separate from :class:`PerformanceMetrics` rather than
    bolted onto it: ``PerformanceMetrics`` is part of the original public,
    backward-compatible surface (it is returned as-is from
    ``QwenClient._metrics`` and serialised in ``diagnostics()``), and adding
    unrelated fields to an existing dataclass that callers may already be
    constructing or comparing against risks subtle breakage. This class is
    purely additive: a new collector, exposed as ``QwenClient._advanced_metrics``,
    that existing code which only knows about ``PerformanceMetrics`` will
    never observe or be affected by.

    Thread-safe. GPU/CPU/RAM readings are best-effort: GPU utilisation
    requires ``pynvml`` (via ``torch.cuda.utilization`` when available) and
    CPU/RAM readings require the optional ``psutil`` dependency; both
    degrade to ``None`` rather than raising when unavailable.

    Args:
        device: Torch device used for GPU utilisation queries.
        latency_window: Number of recent latency samples retained for
            percentile computation.
    """

    def __init__(
        self,
        device: Optional["torch.device"] = None,
        latency_window: int = 2000,
    ) -> None:
        self._device = device
        self._lock = threading.Lock()
        self._latency = _LatencyPercentileTracker(maxlen=latency_window)
        self._active_requests = 0
        self._completed_requests = 0
        self._generation_failures = 0
        self._oom_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_queue_wait = 0.0
        self._queue_wait_samples = 0
        self._request_timestamps: deque[float] = deque(maxlen=10_000)
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def request_started(self) -> None:
        """Mark a request as having entered the active/in-flight state."""
        with self._lock:
            self._active_requests += 1

    def request_finished(
        self,
        *,
        latency: float,
        success: bool,
        queue_wait: float = 0.0,
        cache_hit: bool = False,
        oom: bool = False,
    ) -> None:
        """Record the outcome of a completed request.

        Args:
            latency: Total wall-clock latency in seconds.
            success: Whether the request completed successfully.
            queue_wait: Seconds spent waiting for the generation lock.
            cache_hit: Whether the result was served from the result cache
                or the tokenisation cache.
            oom: Whether this request hit a (recovered) CUDA OOM.
        """
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._completed_requests += 1
            self._request_timestamps.append(time.monotonic())
            self._total_queue_wait += queue_wait
            self._queue_wait_samples += 1
            if not success:
                self._generation_failures += 1
            if oom:
                self._oom_count += 1
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
        self._latency.add(latency)

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def requests_per_minute(self) -> float:
        """Return the request rate over the last 60 seconds.

        Returns:
            Number of ``request_finished`` calls observed in the trailing
            60-second window.
        """
        cutoff = time.monotonic() - 60.0
        with self._lock:
            return float(sum(1 for t in self._request_timestamps if t >= cutoff))

    def cache_hit_ratio(self) -> float:
        """Return the cache hit ratio in ``[0.0, 1.0]``."""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            return self._cache_hits / total if total > 0 else 0.0

    def cache_miss_ratio(self) -> float:
        """Return the cache miss ratio in ``[0.0, 1.0]``."""
        return 1.0 - self.cache_hit_ratio()

    def average_queue_wait(self) -> float:
        """Return the mean queue-wait time in seconds across all requests."""
        with self._lock:
            if self._queue_wait_samples == 0:
                return 0.0
            return self._total_queue_wait / self._queue_wait_samples

    def gpu_utilization_percent(self) -> Optional[float]:
        """Return current GPU utilisation as a percentage, if obtainable.

        Returns:
            Float percentage, or ``None`` if CUDA is unavailable or the
            underlying query fails (e.g. ``pynvml`` not installed).
        """
        if self._device is None or not torch.cuda.is_available():
            return None
        try:
            return float(torch.cuda.utilization(self._device))
        except Exception:
            return None

    def cpu_percent(self) -> Optional[float]:
        """Return process-wide CPU utilisation percentage, if ``psutil`` is
        available; otherwise ``None``."""
        if not _PSUTIL_AVAILABLE:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def ram_usage_mb(self) -> Optional[float]:
        """Return resident set size (RSS) of the current process in MiB, if
        ``psutil`` is available; otherwise ``None``."""
        if not _PSUTIL_AVAILABLE:
            return None
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        except Exception:
            return None

    def reset(self) -> None:
        """Reset all counters, samples, and the latency window."""
        with self._lock:
            self._active_requests = 0
            self._completed_requests = 0
            self._generation_failures = 0
            self._oom_count = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_queue_wait = 0.0
            self._queue_wait_samples = 0
            self._request_timestamps.clear()
            self._start_time = time.monotonic()
        self._latency.reset()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all advanced metrics.

        Returns:
            Dict with latency percentiles, system resource usage, request
            rate, cache ratios, and failure/OOM counters.
        """
        with self._lock:
            active = self._active_requests
            completed = self._completed_requests
            failures = self._generation_failures
            oom = self._oom_count
        return {
            "average_latency_s": round(self._latency.mean(), 4),
            "p95_latency_s": round(self._latency.percentile(95), 4),
            "p99_latency_s": round(self._latency.percentile(99), 4),
            "gpu_utilization_percent": self.gpu_utilization_percent(),
            "cpu_percent": self.cpu_percent(),
            "ram_usage_mb": (
                round(v, 2) if (v := self.ram_usage_mb()) is not None else None
            ),
            "tokens_per_second": None,  # populated by caller from PerformanceMetrics
            "requests_per_minute": self.requests_per_minute(),
            "cache_hit_ratio": round(self.cache_hit_ratio(), 4),
            "cache_miss_ratio": round(self.cache_miss_ratio(), 4),
            "active_requests": active,
            "completed_requests": completed,
            "average_queue_wait_s": round(self.average_queue_wait(), 4),
            "generation_failures": failures,
            "oom_count": oom,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }


# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Static metadata captured at model load time.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        tokenizer_name: HuggingFace tokenizer identifier or local path.
        device: Torch device string used for inference.
        dtype: String representation of the model's parameter dtype.
        vocab_size: Size of the tokenizer vocabulary.
        hidden_size: Width of the model's hidden dimension.
        num_layers: Number of transformer decoder layers.
        context_length: Maximum sequence length supported by the model.
        parameter_count: Total number of model parameters.
        quantized: Whether quantisation (e.g. bitsandbytes) is active.
        loaded_time: ISO-8601 UTC timestamp of successful model load.
        model_revision: Git revision / commit SHA of the model checkpoint.
        transformers_version: Version of the ``transformers`` library.
        torch_version: Version of PyTorch.
        cuda_version: CUDA toolkit version string, or ``"N/A"``.
        python_version: CPython version string.
        gpu_name: GPU display name, or ``"N/A"``.
        gpu_driver: CUDA driver version string, or ``"N/A"``.
        hf_config: Snapshot of the model's HuggingFace config dict.
    """

    model_name: str = ""
    tokenizer_name: str = ""
    device: str = ""
    dtype: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    context_length: int = 0
    parameter_count: int = 0
    quantized: bool = False
    loaded_time: str = ""
    model_revision: str = ""
    transformers_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    python_version: str = ""
    gpu_name: str = ""
    gpu_driver: str = ""
    hf_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------


@dataclass
class ContextBudget:
    """Token budget breakdown for a single RAG request.

    Attributes:
        context_length: Model's total context window (tokens).
        system_tokens: Tokens consumed by the system prompt.
        history_tokens: Tokens consumed by conversation history.
        question_tokens: Tokens consumed by the user question.
        context_tokens: Tokens consumed by retrieved context / documents.
        reserved_generation: Tokens reserved for the assistant answer.
        available: Remaining tokens unaccounted for.
        over_budget: Whether the total exceeds the context window.
    """

    context_length: int = 0
    system_tokens: int = 0
    history_tokens: int = 0
    question_tokens: int = 0
    context_tokens: int = 0
    reserved_generation: int = 0
    available: int = 0
    over_budget: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """Per-request tracking envelope.

    Attributes:
        request_id: UUID string.
        timestamp: ISO-8601 UTC start timestamp.
        prompt_hash: SHA-256 hex digest of the formatted prompt.
        caller_metadata: Arbitrary dict supplied by the caller.
        thread_id: OS thread identifier.
        queue_wait: Seconds spent waiting for the generation lock.
        result: The completed :class:`GenerationResult`, if successful.
        error: Exception message if the request failed.
    """

    request_id: str
    timestamp: str
    prompt_hash: str
    caller_metadata: dict[str, Any]
    thread_id: int
    queue_wait: float = 0.0
    result: Optional[GenerationResult] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        d = asdict(self)
        if self.result is not None:
            d["result"] = self.result.to_dict()
        return d


# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Output of the response validation layer.

    Attributes:
        status: A :class:`ValidationStatus` code.
        valid: Shorthand boolean (``True`` when status is ``VALID``).
        message: Human-readable description of the failure, if any.
    """

    status: str
    valid: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------


@dataclass
class CachedGeneration:
    """Envelope stored by a :class:`ResultCacheBackend` for one prompt.

    This is distinct from the tokenisation cache used internally by
    :class:`_PromptCache`: that cache stores tensors to skip re-tokenising
    an identical formatted prompt, while :class:`CachedGeneration` stores a
    complete, previously computed :class:`GenerationResult` so that an
    identical request can skip inference entirely.

    Attributes:
        prompt_hash: SHA-256 hex digest of the formatted prompt (the cache
            key, also stored inside the entry for export/debugging).
        result: The cached :class:`GenerationResult`.
        created_at: Unix timestamp (``time.time()``) when the entry was
            inserted, used for TTL expiry.
        gen_signature: Hash of the generation parameters (temperature,
            top_p, etc.) that produced ``result``, so that a cache hit only
            occurs when both the prompt *and* the sampling parameters match.
    """

    prompt_hash: str
    result: GenerationResult
    created_at: float
    gen_signature: str = ""

    def is_expired(self, ttl_seconds: float) -> bool:
        """Return ``True`` if this entry is older than ``ttl_seconds``.

        Args:
            ttl_seconds: Time-to-live in seconds. A value ``<= 0`` means
                "never expires".

        Returns:
            Boolean expiry status.
        """
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "prompt_hash": self.prompt_hash,
            "result": self.result.to_dict(),
            "created_at": self.created_at,
            "gen_signature": self.gen_signature,
        }


# ---------------------------------------------------------------------------


class _LatencyPercentileTracker:
    """Thread-safe rolling window of latency samples with percentile queries.

    Keeps the most recent ``maxlen`` samples in a :class:`collections.deque`
    and computes percentiles on demand via :func:`statistics.quantiles`-style
    interpolation. A bounded window is used (rather than retaining every
    sample for the process lifetime) so memory stays constant under
    sustained high request volume.

    Args:
        maxlen: Maximum number of recent samples to retain.
    """

    def __init__(self, maxlen: int = 2000) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        """Record a new latency sample (seconds).

        Args:
            value: Latency sample to record.
        """
        with self._lock:
            self._samples.append(value)

    def percentile(self, p: float) -> float:
        """Return the ``p``-th percentile of the current window.

        Args:
            p: Percentile in the range ``[0, 100]``.

        Returns:
            The interpolated percentile value, or ``0.0`` if no samples
            have been recorded yet.
        """
        with self._lock:
            if not self._samples:
                return 0.0
            ordered = sorted(self._samples)
        if len(ordered) == 1:
            return ordered[0]
        k = (len(ordered) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(ordered) - 1)
        if f == c:
            return ordered[f]
        return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

    def mean(self) -> float:
        """Return the mean of the current window, or ``0.0`` if empty."""
        with self._lock:
            if not self._samples:
                return 0.0
            return statistics.mean(self._samples)

    def reset(self) -> None:
        """Clear all recorded samples."""
        with self._lock:
            self._samples.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


# ===========================================================================
# LRU Prompt Cache
# ===========================================================================


class _PromptCache:
    """Thread-safe LRU cache for tokenised prompts.

    Stores ``(input_ids_cpu, attention_mask_cpu)`` tensors on CPU so that
    GPU memory is not consumed by the cache.  Tensors are moved to the target
    device on retrieval.

    Args:
        maxsize: Maximum number of entries before LRU eviction.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = max(1, maxsize)
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str) -> str:
        """Return the SHA-256 hex digest of ``prompt`` (UTF-8 encoded).

        Args:
            prompt: Raw prompt string.

        Returns:
            64-character hex string.
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(
        self, prompt: str, device: torch.device
    ) -> Optional[dict[str, torch.Tensor]]:
        """Retrieve a cached tokenisation result, moving tensors to ``device``.

        Args:
            prompt: The formatted prompt string used as the cache key.
            device: Target device for the returned tensors.

        Returns:
            A dict of tensors on ``device``, or ``None`` on a cache miss.
        """
        key = self._hash(prompt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return {k: v.to(device) for k, v in self._cache[key].items()}
            self._misses += 1
            return None

    def put(
        self, prompt: str, tensors: dict[str, torch.Tensor]
    ) -> None:
        """Store a tokenisation result in the cache (stored on CPU).

        Args:
            prompt: The formatted prompt string used as the cache key.
            tensors: Dict of tensors to cache (e.g. ``input_ids``,
                ``attention_mask``).
        """
        key = self._hash(prompt)
        cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = cpu_tensors
                return
            self._cache[key] = cpu_tensors
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Evict all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        """Cumulative cache hit count."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Cumulative cache miss count."""
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate in the range [0.0, 1.0]."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics as a plain dictionary."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total if total > 0 else 0.0, 4),
            }


# ===========================================================================
# Result Cache (full GenerationResult cache, pluggable backend)
# ===========================================================================


class ResultCacheBackend(ABC):
    """Abstract storage backend for cached :class:`GenerationResult` objects.

    This is the seam item 4 of the production hardening spec asks for: an
    in-memory implementation ships by default, and a Redis-backed (or any
    other) implementation can be dropped in later by implementing this
    interface and passing it to ``QwenClient`` — see
    :meth:`QwenClient.set_result_cache_backend`. No call site in
    :class:`QwenClient` needs to change when the backend changes.

    Implementations must be thread-safe.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[CachedGeneration]:
        """Retrieve a cached entry by key, or ``None`` on a miss.

        Args:
            key: Cache key (typically a prompt hash, see
                :meth:`QwenClient.hash_prompt`).

        Returns:
            The cached :class:`CachedGeneration`, or ``None``.
        """

    @abstractmethod
    def put(self, key: str, entry: CachedGeneration) -> None:
        """Store ``entry`` under ``key``.

        Args:
            key: Cache key.
            entry: The entry to store.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` from the cache if present.

        Args:
            key: Cache key to evict.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return backend statistics (size, hits, misses, etc.)."""


class InMemoryResultCache(ResultCacheBackend):
    """Thread-safe in-process LRU cache of :class:`GenerationResult` objects.

    Default :class:`ResultCacheBackend` used by :class:`QwenClient`. Supports
    a maximum size with LRU eviction and a per-entry TTL checked lazily on
    ``get`` (no background sweeper thread, so there's nothing extra to shut
    down on :meth:`QwenClient.cleanup`).

    Args:
        maxsize: Maximum number of entries before LRU eviction.
        ttl_seconds: Default time-to-live for entries, in seconds. ``0``
            (or any value ``<= 0``) disables expiry.
    """

    def __init__(self, maxsize: int = 512, ttl_seconds: float = 3600.0) -> None:
        self._maxsize = max(1, maxsize)
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CachedGeneration] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> Optional[CachedGeneration]:
        """Retrieve a cached entry, evicting it first if expired.

        Args:
            key: Cache key.

        Returns:
            The cached entry, or ``None`` if absent or expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(self._ttl):
                del self._store[key]
                self._misses += 1
                self._expirations += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return entry

    def put(self, key: str, entry: CachedGeneration) -> None:
        """Insert or replace an entry, evicting the LRU item if over capacity.

        Args:
            key: Cache key.
            entry: Entry to store.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = entry
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> None:
        """Remove ``key`` if present (no-op otherwise).

        Args:
            key: Cache key to remove.
        """
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries and reset hit/miss/eviction counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def stats(self) -> dict[str, Any]:
        """Return size, capacity, hit/miss/eviction/expiry counters.

        Returns:
            Dict with keys ``size``, ``maxsize``, ``ttl_seconds``, ``hits``,
            ``misses``, ``hit_rate``, ``evictions``, ``expirations``.
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "in_memory",
                "size": len(self._store),
                "maxsize": self._maxsize,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }



# ===========================================================================
# Redis Result Cache Backend
# ===========================================================================


class RedisResultCache(ResultCacheBackend):
    """Redis-backed :class:`ResultCacheBackend` for :class:`QwenClient`.

    Stores serialised :class:`CachedGeneration` entries in Redis using the
    key pattern ``<key_prefix><prompt_hash>``.  TTL is enforced by Redis
    natively (via ``SETEX``) rather than by lazy expiry checks, so entries
    are collected automatically without any background sweeper.

    Connection pooling is used so that concurrent requests share a bounded
    pool of sockets instead of opening a new connection per call.

    This class requires the ``redis`` package (``pip install redis``).
    When ``redis`` is not installed the constructor raises
    :class:`ImportError` with a helpful message.

    Args:
        host: Redis server hostname. Defaults to ``"localhost"``.
        port: Redis server port. Defaults to ``6379``.
        db: Redis logical database index. Defaults to ``0``.
        password: Optional Redis AUTH password.
        ttl_seconds: Default entry TTL in seconds.  ``0`` means no
            expiry (entries persist until evicted by Redis maxmemory
            policy or explicitly deleted).  Defaults to ``3600``.
        key_prefix: String prepended to every cache key so that
            multiple applications sharing one Redis instance don't
            collide.  Defaults to ``"qwen:result:"``.
        max_connections: Size of the connection pool.  Defaults to
            ``10``.
        socket_timeout: Socket-level timeout in seconds for Redis
            operations.  Defaults to ``1.0``.
        socket_connect_timeout: Connection-establishment timeout in
            seconds.  Defaults to ``1.0``.
        ssl: Whether to connect via TLS.  Defaults to ``False``.
        decode_responses: Must remain ``False`` (we use raw bytes for
            JSON payloads).
        **redis_kwargs: Any additional keyword arguments are forwarded
            verbatim to :class:`redis.ConnectionPool`.

    Raises:
        ImportError: When the ``redis`` package is not installed.
        CacheBackendError: When the constructor cannot reach Redis
            (ping fails).  The error is *logged* but **not** re-raised
            so that the caller can decide whether a Redis connectivity
            failure should be fatal.

    Example::

        from llm.qwen_client import QwenClient, RedisResultCache

        client = QwenClient()
        redis_cache = RedisResultCache(
            host="redis.internal",
            port=6379,
            ttl_seconds=1800,
            key_prefix="rag:cache:",
        )
        client.set_result_cache_backend(redis_cache)
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: float = 3600.0,
        key_prefix: str = "qwen:result:",
        max_connections: int = 10,
        socket_timeout: float = 1.0,
        socket_connect_timeout: float = 1.0,
        ssl: bool = False,
        **redis_kwargs: Any,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "RedisResultCache requires the 'redis' package. "
                "Install it with: pip install redis"
            )

        self._ttl = int(ttl_seconds) if ttl_seconds > 0 else None
        self._prefix = key_prefix
        self._lock = threading.Lock()

        # --- counters (thread-safe via _lock) ---
        self._hits = 0
        self._misses = 0
        self._errors = 0

        # --- connection pool ---
        pool_kwargs: dict[str, Any] = dict(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,
            ssl=ssl,
            **redis_kwargs,
        )
        self._pool: "redis.ConnectionPool" = redis.ConnectionPool(**pool_kwargs)
        self._client: "redis.Redis" = redis.Redis(connection_pool=self._pool)

        # Verify connectivity; log but don't raise so callers can fall back.
        try:
            self._client.ping()
            logger.info(
                "RedisResultCache connected | host=%s port=%d db=%d ttl=%s prefix=%s",
                host,
                port,
                db,
                self._ttl,
                key_prefix,
            )
        except RedisError as exc:
            logger.error(
                "RedisResultCache: could not reach Redis at %s:%d – %s. "
                "Cache calls will raise CacheBackendError until connectivity "
                "is restored.",
                host,
                port,
                exc,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        """Return the Redis key with the configured prefix applied.

        Args:
            key: Raw cache key (prompt hash).

        Returns:
            Prefixed Redis key string.
        """
        return f"{self._prefix}{key}"

    @staticmethod
    def _serialise(entry: CachedGeneration) -> bytes:
        """Serialise a :class:`CachedGeneration` to UTF-8 JSON bytes.

        Args:
            entry: The entry to serialise.

        Returns:
            JSON-encoded bytes.
        """
        return json.dumps(entry.to_dict()).encode("utf-8")

    @staticmethod
    def _deserialise(raw: bytes) -> CachedGeneration:
        """Reconstruct a :class:`CachedGeneration` from JSON bytes.

        Args:
            raw: Raw bytes as stored in Redis.

        Returns:
            A :class:`CachedGeneration` instance.

        Raises:
            ValueError: If the payload is not valid JSON or is missing
                required keys.
        """
        data: dict[str, Any] = json.loads(raw.decode("utf-8"))
        result_data = data["result"]
        result = GenerationResult(
            text=result_data["text"],
            prompt_tokens=result_data["prompt_tokens"],
            generated_tokens=result_data["generated_tokens"],
            total_tokens=result_data["total_tokens"],
            latency=result_data["latency"],
            tokens_per_second=result_data["tokens_per_second"],
            finish_reason=result_data["finish_reason"],
            model=result_data["model"],
            device=result_data["device"],
            gpu_memory_used_mb=result_data["gpu_memory_used_mb"],
            timestamp=result_data.get("timestamp", ""),
            request_id=result_data.get("request_id", ""),
            cache_hit=result_data.get("cache_hit", False),
            tokenize_latency=result_data.get("tokenize_latency", 0.0),
            generate_latency=result_data.get("generate_latency", 0.0),
            decode_latency=result_data.get("decode_latency", 0.0),
            gpu_memory_delta_mb=result_data.get("gpu_memory_delta_mb", 0.0),
            retry_count=result_data.get("retry_count", 0),
            validation_status=result_data.get(
                "validation_status", ValidationStatus.VALID
            ),
            caller_metadata=result_data.get("caller_metadata", {}),
            thread_id=result_data.get("thread_id", 0),
            queue_wait=result_data.get("queue_wait", 0.0),
            result_cache_hit=result_data.get("result_cache_hit", False),
        )
        return CachedGeneration(
            prompt_hash=data["prompt_hash"],
            result=result,
            created_at=data["created_at"],
            gen_signature=data.get("gen_signature", ""),
        )

    # ------------------------------------------------------------------
    # ResultCacheBackend interface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[CachedGeneration]:
        """Retrieve a cached entry from Redis.

        Args:
            key: Cache key (prompt hash).

        Returns:
            The :class:`CachedGeneration` if present and not expired,
            ``None`` otherwise.

        Raises:
            CacheBackendError: On a Redis connectivity or protocol
                error.  :class:`QwenClient` catches this and treats it
                as a cache miss so generation can continue.
        """
        redis_key = self._full_key(key)
        try:
            raw: Optional[bytes] = self._client.get(redis_key)
        except RedisError as exc:
            with self._lock:
                self._errors += 1
            logger.warning("RedisResultCache.get error | key=%s | %s", key, exc)
            raise CacheBackendError(str(exc)) from exc

        if raw is None:
            with self._lock:
                self._misses += 1
            return None

        try:
            entry = self._deserialise(raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Corrupted entry – treat as a miss and evict it.
            logger.warning(
                "RedisResultCache: corrupt entry for key=%s, evicting | %s",
                key,
                exc,
            )
            try:
                self._client.delete(redis_key)
            except RedisError:
                pass
            with self._lock:
                self._misses += 1
            return None

        with self._lock:
            self._hits += 1
        return entry

    def put(self, key: str, entry: CachedGeneration) -> None:
        """Store a :class:`CachedGeneration` in Redis.

        When ``ttl_seconds > 0`` the key is stored with ``SETEX`` so Redis
        expires it automatically; otherwise ``SET`` is used and the entry
        persists indefinitely (subject to Redis eviction policies).

        Args:
            key: Cache key.
            entry: The entry to persist.

        Raises:
            CacheBackendError: On a Redis connectivity or serialisation
                error.
        """
        redis_key = self._full_key(key)
        try:
            payload = self._serialise(entry)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "RedisResultCache.put: serialisation failed | key=%s | %s", key, exc
            )
            raise CacheBackendError(str(exc)) from exc

        try:
            if self._ttl is not None and self._ttl > 0:
                self._client.setex(redis_key, self._ttl, payload)
            else:
                self._client.set(redis_key, payload)
        except RedisError as exc:
            with self._lock:
                self._errors += 1
            logger.warning("RedisResultCache.put error | key=%s | %s", key, exc)
            raise CacheBackendError(str(exc)) from exc

    def delete(self, key: str) -> None:
        """Remove ``key`` from Redis (no-op if absent).

        Args:
            key: Cache key to evict.

        Raises:
            CacheBackendError: On a Redis connectivity error.
        """
        try:
            self._client.delete(self._full_key(key))
        except RedisError as exc:
            with self._lock:
                self._errors += 1
            logger.warning("RedisResultCache.delete error | key=%s | %s", key, exc)
            raise CacheBackendError(str(exc)) from exc

    def clear(self) -> None:
        """Delete all keys matching ``<key_prefix>*`` from Redis.

        This uses ``SCAN`` + ``DEL`` in batches rather than ``FLUSHDB``
        so that only keys belonging to this cache are removed — other
        applications sharing the same Redis instance are unaffected.

        Raises:
            CacheBackendError: On a Redis connectivity error.
        """
        pattern = f"{self._prefix}*"
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(
                "RedisResultCache.clear: deleted %d key(s) matching %s",
                deleted,
                pattern,
            )
        except RedisError as exc:
            with self._lock:
                self._errors += 1
            logger.warning("RedisResultCache.clear error | %s", exc)
            raise CacheBackendError(str(exc)) from exc

        with self._lock:
            self._hits = 0
            self._misses = 0
            self._errors = 0

    def size(self) -> int:
        """Return the number of keys matching ``<key_prefix>*`` in Redis.

        Uses a ``SCAN``-based count so as not to block the server.
        Returns ``-1`` if the count cannot be obtained.

        Returns:
            Entry count, or ``-1`` on error.
        """
        pattern = f"{self._prefix}*"
        count = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except RedisError as exc:
            logger.warning("RedisResultCache.size error | %s", exc)
            return -1

    def ping(self) -> bool:
        """Return ``True`` if the Redis connection is healthy.

        Returns:
            Boolean indicating whether a PING round-trip succeeded.
        """
        try:
            return bool(self._client.ping())
        except RedisError:
            return False

    def stats(self) -> dict[str, Any]:
        """Return backend statistics including hit/miss/error counters.

        Returns:
            Dict with keys ``backend``, ``size``, ``ttl_seconds``,
            ``prefix``, ``hits``, ``misses``, ``errors``, ``hit_rate``,
            ``connected``.
        """
        with self._lock:
            hits = self._hits
            misses = self._misses
            errors = self._errors
        total = hits + misses
        return {
            "backend": "redis",
            "size": self.size(),
            "ttl_seconds": self._ttl,
            "prefix": self._prefix,
            "hits": hits,
            "misses": misses,
            "errors": errors,
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            "connected": self.ping(),
        }

    def close(self) -> None:
        """Disconnect all pooled connections gracefully.

        Call this when the application shuts down to release file
        descriptors cleanly.  After calling ``close()`` this backend
        should not be used.
        """
        try:
            self._pool.disconnect()
            logger.info("RedisResultCache: connection pool disconnected.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("RedisResultCache.close error | %s", exc)


# ===========================================================================
# ConversationMemory
# ===========================================================================


class ConversationMemory:
    """Thread-safe conversation history manager for multi-turn RAG sessions.

    Maintains a list of ``{"role": …, "content": …}`` dicts and provides
    token-budget-aware trimming.

    Args:
        max_messages: Hard cap on the number of messages retained.
        max_tokens: Soft token budget; trimming removes oldest turns first.
        tokenizer: Optional tokenizer for exact token counting.  When
            ``None`` a character-based heuristic is used instead.
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int = 4096,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _count(self, text: str) -> int:
        """Estimate token count for ``text``."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Rough heuristic: 4 chars ≈ 1 token
        return max(1, len(text) // 4)

    def _total_tokens(self) -> int:
        """Return the total estimated token count across all messages."""
        return sum(self._count(m["content"]) for m in self._messages)

    # ------------------------------------------------------------------

    def append(self, role: str, content: str) -> None:
        """Append a message and auto-trim to budget.

        Args:
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
        """
        with self._lock:
            self._messages.append({"role": role, "content": content})
            # Trim by message count
            while len(self._messages) > self._max_messages:
                self._messages.pop(0)
            # Trim by token budget (keep most recent)
            while self._total_tokens() > self._max_tokens and len(self._messages) > 1:
                self._messages.pop(0)

    def clear(self) -> None:
        """Remove all messages."""
        with self._lock:
            self._messages.clear()

    def truncate(self, n: int) -> None:
        """Keep only the most recent ``n`` messages.

        Args:
            n: Number of messages to retain.
        """
        with self._lock:
            self._messages = self._messages[-n:]

    def last_n_messages(self, n: int) -> list[dict[str, str]]:
        """Return the most recent ``n`` messages (read-only copy).

        Args:
            n: Number of messages to return.

        Returns:
            List of message dicts.
        """
        with self._lock:
            return list(self._messages[-n:])

    def trim_to_token_budget(self, budget: int) -> None:
        """Trim history until its token count fits within ``budget``.

        Args:
            budget: Maximum token budget for the history section.
        """
        with self._lock:
            while self._total_tokens() > budget and len(self._messages) > 1:
                self._messages.pop(0)

    def summarize(self) -> str:
        """Return a compact text representation of the conversation.

        Returns:
            Multi-line string with one ``role: content`` line per message.
        """
        with self._lock:
            return "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in self._messages
            )

    @property
    def messages(self) -> list[dict[str, str]]:
        """Read-only copy of the current message list."""
        with self._lock:
            return list(self._messages)

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


# ===========================================================================
# Telemetry
# ===========================================================================


class Telemetry:
    """Lightweight in-process telemetry collector.

    Accumulates request/error/latency/token/GPU/cache statistics.
    Thread-safe.  Export via :meth:`to_dict` or :meth:`to_json`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "requests": 0,
            "errors": 0,
            "total_latency_s": 0.0,
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "gpu_peak_mb": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "oom_events": 0,
            "retries": 0,
        }

    def record(self, result: GenerationResult) -> None:
        """Update telemetry from a completed :class:`GenerationResult`.

        Args:
            result: The generation result to record.
        """
        with self._lock:
            self._data["requests"] += 1
            self._data["total_latency_s"] += result.latency
            self._data["total_prompt_tokens"] += result.prompt_tokens
            self._data["total_generated_tokens"] += result.generated_tokens
            self._data["gpu_peak_mb"] = max(
                self._data["gpu_peak_mb"], result.gpu_memory_used_mb
            )
            if result.cache_hit:
                self._data["cache_hits"] += 1
            else:
                self._data["cache_misses"] += 1
            self._data["retries"] += result.retry_count

    def record_error(self) -> None:
        """Increment the error counter."""
        with self._lock:
            self._data["errors"] += 1

    def record_oom(self) -> None:
        """Increment the OOM counter."""
        with self._lock:
            self._data["oom_events"] += 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        with self._lock:
            return dict(self._data)

    def to_json(self, indent: int = 2) -> str:
        """Return a pretty-printed JSON snapshot."""
        return json.dumps(self.to_dict(), indent=indent)


# ===========================================================================
# Thread-safe Singleton metaclass
# ===========================================================================


class _SingletonMeta(type):
    """Thread-safe metaclass ensuring at most one instance per class."""

    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


# ===========================================================================
# RetryManager
# ===========================================================================


class _RetryManager:
    """Manages retry logic for recoverable generation failures.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base seconds between retries (exponential backoff).
        recoverable_errors: Exception types considered recoverable.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        recoverable_errors: tuple[type[Exception], ...] = (
            torch.cuda.OutOfMemoryError,  # type: ignore[attr-defined]
            RuntimeError,
        ),
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._recoverable = recoverable_errors

    def is_recoverable(self, exc: Exception) -> bool:
        """Return ``True`` if ``exc`` is a recoverable error type.

        Args:
            exc: The exception to test.

        Returns:
            Boolean indicating recoverability.
        """
        return isinstance(exc, self._recoverable)

    def sleep(self, attempt: int) -> None:
        """Sleep for an exponentially increasing interval.

        Args:
            attempt: Zero-based attempt index.
        """
        time.sleep(self._backoff_base * (2 ** attempt))

    @property
    def max_retries(self) -> int:
        """Maximum number of retry attempts."""
        return self._max_retries

    @property
    def backoff_base(self) -> float:
        """Base seconds between retries (before exponential scaling)."""
        return self._backoff_base


# ===========================================================================
# Streaming Cancellation Support
# ===========================================================================


class _CancellationStoppingCriteria(StoppingCriteria):
    """A :class:`~transformers.StoppingCriteria` backed by a
    :class:`threading.Event`.

    ``model.generate()`` polls every registered :class:`StoppingCriteria`
    after each generated token; returning ``True`` here halts generation on
    the *next* check, which is the only reliable way to stop a HuggingFace
    generate call mid-flight from another thread (killing the generation
    thread itself is not safe — it can leave CUDA state inconsistent).

    Args:
        cancel_event: Event that, once set, causes generation to stop.
    """

    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        """Return ``True`` (stop) once :attr:`_cancel_event` has been set."""
        return self._cancel_event.is_set()


# ===========================================================================
# Model Loading Backend (generic, swappable)
# ===========================================================================
#
# Everything QwenClient does after this point in the file — tokenisation,
# generation, decoding, streaming, batching, GPU diagnostics — operates
# purely on `self._tokenizer` / `self._model` / `self._device` / `self._dtype`
# and never assumes anything Qwen-specific about them (no Qwen chat-template
# hardcoding, no Qwen-only generation arguments). The ONLY place a concrete
# model family is ever chosen is inside `ModelLoader.load_tokenizer()` /
# `load_model()` below.
#
# This is the seam for "support other providers later without requiring
# changes in the rest of the project": to add a new local backend (a
# different HF causal LM family, a quantised/AWQ/GGUF loader, etc.), write a
# new `ModelLoader` subclass and pass it to `QwenClient(model_loader=...)`.
# No other method on QwenClient — and no caller of QwenClient — needs to
# change.


class ModelLoader(ABC):
    """Strategy interface for loading a tokenizer + causal LM pair.

    Implementations own every backend-specific detail (which `transformers`
    loader class to use, quantisation config, `trust_remote_code`, chat
    template handling, etc.). :class:`QwenClient` only ever calls
    :meth:`load_tokenizer` / :meth:`load_model` and stores whatever comes
    back — it does not care which model family produced them.
    """

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load and return a ready-to-use tokenizer.

        Raises:
            TokenizerLoadError: If the tokenizer cannot be loaded.
        """

    @abstractmethod
    def load_model(
        self, dtype: torch.dtype, device: torch.device
    ) -> PreTrainedModel:
        """Load and return a ready-to-use causal LM, in eval mode.

        Args:
            dtype: Resolved weight dtype (see ``QwenClient._resolve_dtype``).
            device: Resolved inference device (see ``QwenClient._resolve_device``).

        Raises:
            GPUOutOfMemoryError: On CUDA OOM during load.
            ModelLoadError: For any other load failure.
        """


class HFAutoModelLoader(ModelLoader):
    """Default :class:`ModelLoader`: any HuggingFace ``AutoModelForCausalLM``
    -compatible checkpoint, selected entirely by ``settings.MODEL_NAME``.

    This is intentionally generic — it is not Qwen-specific in any way, and
    already supports swapping the backend model family today by changing
    configuration alone (e.g. ``MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct``
    or ``MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3``), with zero code
    changes anywhere in this file or its callers.
    """

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer_name: str = getattr(settings, "TOKENIZER_NAME", settings.MODEL_NAME)
        logger.info("Loading tokenizer | name=%s", tokenizer_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=getattr(settings, "TRUST_REMOTE_CODE", True),
                use_fast=True,
            )
        except Exception as exc:
            logger.error(
                "Tokenizer load failed | name=%s | error=%s", tokenizer_name, exc
            )
            raise TokenizerLoadError(
                f"Failed to load tokenizer '{tokenizer_name}': {exc}. "
                "Check that the model/tokenizer name is correct, that any "
                "required credentials for a gated repo are configured, and "
                "that the host has network access to the HuggingFace Hub "
                "(or that a local path was supplied)."
            ) from exc

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug(
                "pad_token not set; defaulting to eos_token (%s)",
                tokenizer.eos_token,
            )
        logger.info(
            "Tokenizer loaded | name=%s | vocab_size=%d | fast=%s",
            tokenizer_name,
            len(tokenizer),
            tokenizer.is_fast,
        )
        return tokenizer

    def load_model(
        self, dtype: torch.dtype, device: torch.device
    ) -> PreTrainedModel:
        model_name: str = settings.MODEL_NAME
        use_device_map_auto = getattr(settings, "USE_DEVICE_MAP_AUTO", True)
        logger.info(
            "Loading model | name=%s | dtype=%s | device=%s | device_map_auto=%s",
            model_name,
            dtype,
            device,
            use_device_map_auto,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=(
                    "auto"
                    if use_device_map_auto and device.type == "cuda"
                    else None
                ),
                trust_remote_code=getattr(settings, "TRUST_REMOTE_CODE", True),
                low_cpu_mem_usage=True,
            )
            if device.type != "cuda" or not use_device_map_auto:
                model = model.to(device)

            model.eval()

            if getattr(settings, "TORCH_COMPILE", False):
                logger.info("Compiling model with torch.compile (this may take a while) …")
                model = torch.compile(model)  # type: ignore[assignment]

            return model
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(
                "GPU OOM while loading model '%s' – consider reducing MODEL_DTYPE"
                " or using a smaller model. Error: %s",
                model_name,
                exc,
            )
            raise GPUOutOfMemoryError(
                f"GPU OOM while loading model '{model_name}': {exc}. "
                "Consider using float16/bfloat16 dtype or a smaller model."
            ) from exc
        except Exception as exc:
            logger.error("Model load failed | name=%s | error=%s", model_name, exc)
            raise ModelLoadError(
                f"Failed to load model '{model_name}': {exc}"
            ) from exc


# ===========================================================================
# QwenClient
# ===========================================================================


class QwenClient(metaclass=_SingletonMeta):
    """Thread-safe singleton inference engine for HuggingFace causal LMs.

    Loads a tokenizer and model on first construction, then exposes a
    high-level :meth:`generate` API suitable for FastAPI request handlers,
    multi-agent RAG pipelines, and evaluation harnesses. Subsequent calls to
    ``QwenClient()`` return the same instance (singleton semantics), so the
    model is loaded exactly once per process regardless of how many modules
    call the constructor.

    All public configuration is read from ``config.settings`` at startup.
    Individual :meth:`generate` calls may supply per-request runtime
    overrides via keyword arguments without permanently modifying the default
    :class:`~transformers.GenerationConfig`.

    Example::

        client = QwenClient()
        with client:
            result = client.generate("Explain quantum entanglement simply.")
            print(result.text)
            print(f"Generated {result.generated_tokens} tokens in {result.latency:.2f}s")

    Attributes:
        _model: The loaded :class:`~transformers.PreTrainedModel`.
        _tokenizer: The loaded :class:`~transformers.PreTrainedTokenizerBase`.
        _gen_config: The active :class:`~transformers.GenerationConfig`.
        _model_info: :class:`ModelInfo` populated at model load time.
        _metrics: :class:`PerformanceMetrics` updated after every request.
        _device: The resolved :class:`torch.device` for inference.
        _dtype: The resolved :class:`torch.dtype` for model weights.
        _cache: :class:`_PromptCache` for tokenised prompt tensors.
        _telemetry: :class:`Telemetry` collector.
        _hooks: Dict of registered lifecycle callbacks per :class:`HookEvent`.
        _retry_manager: :class:`_RetryManager` for transient failures.
        _advanced_metrics: :class:`AdvancedMetrics` for percentile latency
            and system-resource statistics.
    """

    # ------------------------------------------------------------------
    # Construction & initialisation
    # ------------------------------------------------------------------

    def __init__(self, model_loader: Optional[ModelLoader] = None) -> None:  # noqa: D107
        """Construct (or return the existing singleton) client.

        Args:
            model_loader: Optional :class:`ModelLoader` strategy controlling
                how the tokenizer/model are loaded. Defaults to
                :class:`HFAutoModelLoader` (any ``AutoModelForCausalLM``
                -compatible checkpoint named by ``settings.MODEL_NAME``),
                which preserves the exact loading behaviour of previous
                revisions of this client. Only the *first* construction in a
                process is honoured, per singleton semantics — later
                ``QwenClient()`` calls (with or without this argument)
                return the already-initialised instance untouched. This
                argument is the extension point for future backends (a
                different model family, a quantised loader, etc.) and does
                not require any change to existing callers, which continue
                to call ``QwenClient()`` with no arguments.
        """
        if getattr(self, "_initialised", False):
            return

        self._initialised: bool = False
        self._loader: ModelLoader = model_loader or HFAutoModelLoader()
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._gen_config: Optional[GenerationConfig] = None
        self._default_gen_config_dict: dict[str, Any] = {}
        self._model_info: ModelInfo = ModelInfo()
        self._metrics: PerformanceMetrics = PerformanceMetrics()
        self._telemetry: Telemetry = Telemetry()
        self._start_time: float = time.monotonic()
        self._request_counter: int = 0
        self._auto_cleanup_counter: int = 0
        self._warmup_complete: bool = False

        # Locks
        self._generate_lock: threading.Lock = threading.Lock()
        self._reload_lock: threading.Lock = threading.Lock()
        # Guards every read/write of self._gen_config. generate() already
        # serialises full requests behind _generate_lock, but
        # update_generation_config()/reset_generation_config() previously
        # mutated _gen_config without holding any lock at all, racing
        # against the swap-in/swap-out of `active_cfg` inside generate()'s
        # critical section. This dedicated lock closes that gap without
        # changing the (already correct) locking around _generate_lock.
        self._gen_config_lock: threading.Lock = threading.Lock()

        # Lifecycle hooks: event → list of callables
        self._hooks: dict[str, list[Callable[..., None]]] = {
            e.value: [] for e in HookEvent
        }

        # LRU prompt cache
        cache_size: int = getattr(settings, "PROMPT_CACHE_SIZE", 256)
        self._cache: _PromptCache = _PromptCache(maxsize=cache_size)

        # Result cache (full GenerationResult, pluggable backend — spec #4).
        # Defaults to the in-memory backend; swap via
        # set_result_cache_backend() to plug in Redis or any other store
        # without touching generate()'s call sites.
        result_cache_size: int = getattr(settings, "RESULT_CACHE_SIZE", 512)
        result_cache_ttl: float = getattr(settings, "RESULT_CACHE_TTL_SECONDS", 3600.0)

        # Auto-select Redis backend when REDIS_HOST is configured *and* the
        # ``redis`` package is available; fall back to InMemoryResultCache
        # transparently so the system stays functional without Redis.
        redis_host: Optional[str] = getattr(settings, "REDIS_HOST", None)
        self._result_cache: ResultCacheBackend
        if redis_host and _REDIS_AVAILABLE:
            try:
                self._result_cache = RedisResultCache(
                    host=redis_host,
                    port=int(getattr(settings, "REDIS_PORT", 6379)),
                    db=int(getattr(settings, "REDIS_DB", 0)),
                    password=getattr(settings, "REDIS_PASSWORD", None) or None,
                    ttl_seconds=result_cache_ttl,
                    key_prefix=getattr(
                        settings, "REDIS_KEY_PREFIX", "qwen:result:"
                    ),
                    max_connections=int(
                        getattr(settings, "REDIS_MAX_CONNECTIONS", 10)
                    ),
                    socket_timeout=float(
                        getattr(settings, "REDIS_SOCKET_TIMEOUT", 1.0)
                    ),
                    socket_connect_timeout=float(
                        getattr(settings, "REDIS_CONNECT_TIMEOUT", 1.0)
                    ),
                    ssl=bool(getattr(settings, "REDIS_SSL", False)),
                )
                logger.info(
                    "QwenClient: using RedisResultCache | host=%s port=%s",
                    redis_host,
                    getattr(settings, "REDIS_PORT", 6379),
                )
            except Exception as _redis_init_exc:  # noqa: BLE001
                logger.warning(
                    "QwenClient: RedisResultCache init failed (%s) – "
                    "falling back to InMemoryResultCache.",
                    _redis_init_exc,
                )
                self._result_cache = InMemoryResultCache(
                    maxsize=result_cache_size, ttl_seconds=result_cache_ttl
                )
        else:
            if redis_host and not _REDIS_AVAILABLE:
                logger.warning(
                    "QwenClient: REDIS_HOST is set but the 'redis' package is "
                    "not installed – falling back to InMemoryResultCache. "
                    "Install it with: pip install redis"
                )
            self._result_cache = InMemoryResultCache(
                maxsize=result_cache_size, ttl_seconds=result_cache_ttl
            )

        self._result_cache_enabled: bool = getattr(
            settings, "RESULT_CACHE_ENABLED", True
        )

        # Retry manager. `retry_attempts`/`retry_delay`/`backoff_factor`
        # are the names used in the production-hardening spec; they map
        # onto the existing _RetryManager(max_retries=, backoff_base=)
        # constructor, which is left unchanged for backward compatibility.
        self._retry_manager: _RetryManager = _RetryManager(
            max_retries=getattr(
                settings,
                "RETRY_ATTEMPTS",
                getattr(settings, "MAX_GENERATION_RETRIES", 3),
            ),
            backoff_base=getattr(
                settings,
                "RETRY_DELAY",
                getattr(settings, "RETRY_BACKOFF_BASE", 0.5),
            ),
        )
        self._retry_backoff_factor: float = getattr(settings, "RETRY_BACKOFF_FACTOR", 2.0)

        # Active streaming generations, keyed by request_id, so that
        # cancel_stream() can signal a specific in-flight stream. Entries
        # are removed when the stream finishes, errors, or is cancelled.
        self._stream_cancel_events: dict[str, threading.Event] = {}
        self._stream_cancel_lock: threading.Lock = threading.Lock()

        # Auto-cleanup every N requests (0 = disabled)
        self._auto_cleanup_interval: int = getattr(
            settings, "AUTO_CLEANUP_INTERVAL", 50
        )

        # Device / dtype
        self._device: torch.device = self._resolve_device()
        self._dtype: torch.dtype = self._resolve_dtype()

        # Advanced metrics collector (percentiles, system resource usage,
        # request rate — spec #8). Separate from PerformanceMetrics; see
        # AdvancedMetrics docstring for the rationale.
        self._advanced_metrics: AdvancedMetrics = AdvancedMetrics(device=self._device)

        # Bounded request-history ring buffer (request tracing — spec #12).
        # RequestRecord existed in the original module as a fully-defined
        # dataclass but was never instantiated anywhere; this wires it up
        # as an opt-in, memory-bounded trace of recent requests (hash +
        # timing + outcome, never the raw prompt text) for debugging and
        # request-tracing endpoints.
        self._request_history_enabled: bool = getattr(
            settings, "REQUEST_HISTORY_ENABLED", True
        )
        history_size: int = getattr(settings, "REQUEST_HISTORY_SIZE", 200)
        self._request_history: deque[RequestRecord] = deque(maxlen=history_size)
        self._request_history_lock: threading.Lock = threading.Lock()

        logger.info(
            "QwenClient initialising | device=%s | dtype=%s | cache_size=%d"
            " | auto_cleanup_interval=%d | result_cache_enabled=%s",
            self._device,
            self._dtype,
            cache_size,
            self._auto_cleanup_interval,
            self._result_cache_enabled,
        )

        self._load_tokenizer()
        self._load_model()
        self._init_generation_config()

        # Auto-warmup
        if getattr(settings, "AUTO_WARMUP", True):
            self.warmup()

        self._initialised = True
        logger.info(
            "QwenClient ready | model=%s | device=%s | dtype=%s | warmup=%s",
            settings.MODEL_NAME,
            self._device,
            self._dtype,
            self._warmup_complete,
        )

    # ------------------------------------------------------------------
    # Device / dtype resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        """Return the best available :class:`torch.device`.

        Returns:
            Resolved :class:`torch.device`.
        """
        if torch.cuda.is_available():
            device_str = getattr(settings, "DEVICE", "cuda:0")
            device = torch.device(device_str)
            logger.info(
                "CUDA available | device=%s | name=%s",
                device,
                torch.cuda.get_device_name(device),
            )
            return device
        logger.warning("CUDA not available – falling back to CPU")
        return torch.device("cpu")

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve model weight dtype from settings.

        Returns:
            A :class:`torch.dtype` constant.
        """
        dtype_map: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        configured = getattr(settings, "MODEL_DTYPE", "").lower()
        if configured in dtype_map:
            return dtype_map[configured]
        if self._device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self) -> None:
        """Load and configure the tokenizer via :attr:`_loader`.

        Delegates to ``self._loader.load_tokenizer()`` (see
        :class:`ModelLoader`). All backend-specific loading detail lives in
        the loader; this method's job is just to store the result and fail
        loudly if loading didn't work.

        Raises:
            TokenizerLoadError: If the tokenizer cannot be loaded for any
                reason, including missing files, network errors, or
                incompatible tokenizer classes. Previous revisions of this
                method silently substituted a non-functional dummy
                tokenizer on any load failure and returned normally, which
                let the client report itself "ready" while actually being
                unable to tokenize anything correctly. That fallback has
                been removed: a tokenizer load failure is fatal, as
                ``TokenizerLoadError``'s own docstring always said it
                should be.
        """
        self._tokenizer = self._loader.load_tokenizer()

    def _load_model(self) -> None:
        """Load the causal language model via :attr:`_loader`.

        Delegates to ``self._loader.load_model(dtype, device)`` (see
        :class:`ModelLoader`), then populates :attr:`_model_info` from the
        result exactly as before.

        Raises:
            GPUOutOfMemoryError: When a CUDA OOM occurs during model
                loading. The model weights are too large for the available
                VRAM; use a smaller model, a lower precision dtype, or
                more GPU memory.
            ModelLoadError: For any other failure, including path not found,
                network timeout, or an incompatible model architecture.
                The original exception is chained via ``__cause__``.
        """
        model_name: str = settings.MODEL_NAME
        self._model = self._loader.load_model(self._dtype, self._device)
        self._populate_model_info(model_name)
        logger.info(
            "Model loaded successfully | name=%s | params=%s | context_length=%d"
            " | gpu_allocated=%.1f MiB",
            model_name,
            f"{self._model_info.parameter_count:,}",
            self._model_info.context_length,
            self.gpu_memory_allocated(),
        )

    def _populate_model_info(self, model_name: str) -> None:
        """Populate :attr:`_model_info` after a successful model load.

        Args:
            model_name: HuggingFace model identifier or local path.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "_populate_model_info() called before model/tokenizer were loaded."
            )

        cfg = self._model.config
        cuda_ver = "N/A"
        gpu_name = "N/A"
        gpu_driver = "N/A"
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "N/A"
            gpu_name = torch.cuda.get_device_name(self._device)
            try:
                gpu_driver = str(
                    torch.cuda.get_device_properties(self._device).major
                ) + "." + str(
                    torch.cuda.get_device_properties(self._device).minor
                )
            except Exception:
                gpu_driver = "N/A"

        # Capture a plain-dict snapshot of the HF config for diagnostics
        hf_cfg_dict: dict[str, Any] = {}
        try:
            hf_cfg_dict = cfg.to_diff_dict()
        except Exception:
            hf_cfg_dict = {}

        self._model_info = ModelInfo(
            model_name=model_name,
            tokenizer_name=getattr(settings, "TOKENIZER_NAME", model_name),
            device=str(self._device),
            dtype=str(self._dtype),
            vocab_size=len(self._tokenizer),
            hidden_size=getattr(cfg, "hidden_size", 0),
            num_layers=getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)),
            context_length=getattr(
                cfg,
                "max_position_embeddings",
                getattr(cfg, "max_seq_len", getattr(settings, "MAX_CONTEXT_LENGTH", 32768)),
            ),
            parameter_count=sum(p.numel() for p in self._model.parameters()),
            quantized=getattr(cfg, "quantization_config", None) is not None,
            loaded_time=datetime.now(timezone.utc).isoformat(),
            model_revision=getattr(cfg, "_commit_hash", ""),
            transformers_version=_transformers_version,
            torch_version=torch.__version__,
            cuda_version=cuda_ver,
            python_version=platform.python_version(),
            gpu_name=gpu_name,
            gpu_driver=gpu_driver,
            hf_config=hf_cfg_dict,
        )

    def _init_generation_config(self) -> None:
        """Build the default :class:`GenerationConfig` from ``Config.settings``."""
        cfg_kwargs: dict[str, Any] = {
            "max_new_tokens": getattr(settings, "MAX_NEW_TOKENS", 2048),
            "temperature": getattr(settings, "TEMPERATURE", 0.7),
            "top_p": getattr(settings, "TOP_P", 0.9),
            "top_k": getattr(settings, "TOP_K", 50),
            "repetition_penalty": getattr(settings, "REPETITION_PENALTY", 1.1),
            "do_sample": getattr(settings, "DO_SAMPLE", True),
            "pad_token_id": (
                self._tokenizer.pad_token_id if self._tokenizer else None
            ),
            "eos_token_id": (
                self._tokenizer.eos_token_id if self._tokenizer else None
            ),
        }
        self._gen_config = GenerationConfig(**cfg_kwargs)
        self._default_gen_config_dict = dict(cfg_kwargs)
        logger.debug(
            "GenerationConfig initialised | max_new_tokens=%d | temperature=%.2f"
            " | top_p=%.2f | top_k=%d | repetition_penalty=%.2f | do_sample=%s",
            self._gen_config.max_new_tokens,
            self._gen_config.temperature,
            self._gen_config.top_p,
            self._gen_config.top_k,
            self._gen_config.repetition_penalty,
            self._gen_config.do_sample,
        )

    # ------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------

    def register_hook(self, event: Union[HookEvent, str], fn: Callable[..., None]) -> None:
        """Register a callback for a lifecycle event.

        Args:
            event: A :class:`HookEvent` member or its string value.
            fn: Callable invoked with keyword arguments relevant to the event.

        Example::

            def on_gen(result, **kw):
                print(result.tokens_per_second)

            client.register_hook(HookEvent.AFTER_GENERATION, on_gen)
        """
        key = event.value if isinstance(event, HookEvent) else str(event)
        self._hooks.setdefault(key, []).append(fn)

    def _fire(self, event: HookEvent, **kwargs: Any) -> None:
        """Invoke all registered callbacks for ``event``.

        Args:
            event: The lifecycle event to fire.
            **kwargs: Keyword arguments forwarded to each callback.
        """
        for fn in self._hooks.get(event.value, []):
            try:
                fn(**kwargs)
            except Exception as exc:
                logger.warning(
                    "Hook %s raised an exception (suppressed): %s", event.value, exc
                )

    # ------------------------------------------------------------------
    # Safety Layer
    # ------------------------------------------------------------------

    def validate_prompt_safety(self, prompt: str) -> None:
        """Run safety checks on ``prompt`` before generation.

        Checks performed (in order):
          1. Empty or whitespace-only input → :class:`InvalidPromptError`.
          2. Prompt exceeding ``settings.MAX_PROMPT_CHARS`` → :class:`SafetyError`.
          3. Null bytes (``\\x00``) in the prompt → :class:`SafetyError`.
          4. Configurable jailbreak fragment match via
             :meth:`_jailbreak_detected` → :class:`SafetyError`.

        Args:
            prompt: The fully formatted prompt string (after
                :meth:`build_chat_prompt`).

        Raises:
            InvalidPromptError: If the prompt is empty or whitespace-only.
            SafetyError: If the prompt violates any safety policy.
        """
        if not prompt or not prompt.strip():
            raise InvalidPromptError("Prompt is empty or whitespace-only.")

        max_chars: int = getattr(settings, "MAX_PROMPT_CHARS", 200_000)
        if len(prompt) > max_chars:
            raise SafetyError(
                f"Prompt length ({len(prompt):,} chars) exceeds the maximum "
                f"allowed ({max_chars:,} chars). Trim the prompt or retrieved context."
            )

        if "\x00" in prompt:
            raise SafetyError(
                "Prompt contains null bytes (\\x00), which are not allowed."
            )

        if self._jailbreak_detected(prompt):
            logger.warning("Prompt blocked by jailbreak detection heuristic.")
            raise SafetyError("Prompt triggered the jailbreak detection heuristic.")

    def _jailbreak_detected(self, prompt: str) -> bool:
        """Heuristic jailbreak detection hook.

        Override this method to integrate a custom classifier.

        Args:
            prompt: The formatted prompt string.

        Returns:
            ``True`` if the prompt should be blocked.
        """
        # Minimal built-in heuristic: block known injection scaffolding.
        lowered = prompt.lower()
        blocked_fragments = getattr(settings, "JAILBREAK_FRAGMENTS", [])
        return any(frag.lower() in lowered for frag in blocked_fragments)

    def validate_response(self, text: str) -> ValidationResult:
        """Validate the decoded response text.

        Checks for:
          - Empty or whitespace-only response.
          - Responses that consist entirely of punctuation / whitespace.
          - Responses that repeat the same word or phrase excessively.
          - Hallucinated role prefixes (e.g. ``"assistant:"``).
          - Malformed unicode (surrogate characters).

        Args:
            text: The cleaned assistant response text.

        Returns:
            A :class:`ValidationResult` with ``valid=True`` or a failure
            code and human-readable message.
        """
        if not text or not text.strip():
            return ValidationResult(
                status=ValidationStatus.EMPTY,
                valid=False,
                message="Response is empty or whitespace-only.",
            )

        # Punctuation-only
        stripped = text.strip()
        if all(not c.isalnum() for c in stripped):
            return ValidationResult(
                status=ValidationStatus.PUNCTUATION_ONLY,
                valid=False,
                message="Response contains only punctuation.",
            )

        # Repeated n-gram detection (simple: most common 3-gram > 50% of all)
        words = stripped.split()
        if len(words) >= 9:
            trigrams: dict[str, int] = {}
            for i in range(len(words) - 2):
                key = " ".join(words[i : i + 3])
                trigrams[key] = trigrams.get(key, 0) + 1
            max_freq = max(trigrams.values())
            if max_freq / len(trigrams) > 0.5:
                return ValidationResult(
                    status=ValidationStatus.REPEATED,
                    valid=False,
                    message="Response appears to contain repetitive content.",
                )

        # Hallucinated role prefix
        for prefix in ("assistant:", "assistant\n", "user:", "user\n", "<|im_start|>"):
            if stripped.lower().startswith(prefix.lower()):
                return ValidationResult(
                    status=ValidationStatus.HALLUCINATED_PREFIX,
                    valid=False,
                    message=f"Response begins with a hallucinated prefix: '{prefix}'.",
                )

        # Malformed unicode (surrogate halves)
        try:
            text.encode("utf-16", errors="surrogatepass").decode("utf-16")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return ValidationResult(
                status=ValidationStatus.MALFORMED_UNICODE,
                valid=False,
                message="Response contains malformed unicode characters.",
            )

        return ValidationResult(status=ValidationStatus.VALID, valid=True)

    # ------------------------------------------------------------------
    # Generation Parameter Safety Layer (spec #14)
    # ------------------------------------------------------------------

    def validate_generation_parameters(
        self,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        """Validate sampling-parameter overrides before they reach the model.

        Every value is optional; only supplied (non-``None``) values are
        checked, making this safe for partial override dicts. Called
        automatically at the top of :meth:`generate` and
        :meth:`stream_generate`; also safe to call standalone as a
        pre-validation step (e.g. inside a FastAPI request handler).

        Args:
            max_new_tokens: Must be a positive ``int`` when supplied.
            temperature: Must be a ``float`` in ``[0.0, 5.0]`` when supplied.
                Values above ~2.0 produce near-random output; the upper
                bound exists to reject obvious mistakes, not to prevent
                intentional high-temperature sampling.
            top_p: Must be a ``float`` in ``(0.0, 1.0]`` when supplied.
                The HuggingFace convention treats ``top_p=1.0`` as
                "no nucleus filtering" (all tokens eligible).
            top_k: Must be a non-negative ``int`` when supplied. ``0``
                disables top-k filtering, matching HuggingFace semantics.
            repetition_penalty: Must be a ``float`` in ``(0.0, 10.0]``
                when supplied. Values below ``1.0`` increase repetition;
                ``1.0`` applies no penalty; values above ``1.0`` reduce it.

        Raises:
            InvalidGenerationParameterError: If any supplied value is
                outside its valid range or has an unexpected type. The
                error message names the parameter and states its valid range.
        """
        if max_new_tokens is not None:
            if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
                raise InvalidGenerationParameterError(
                    f"max_new_tokens must be an int, got {type(max_new_tokens).__name__}."
                )
            if max_new_tokens <= 0:
                raise InvalidGenerationParameterError(
                    f"max_new_tokens must be a positive integer, got {max_new_tokens}."
                )

        if temperature is not None:
            if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
                raise InvalidGenerationParameterError(
                    f"temperature must be a float, got {type(temperature).__name__}."
                )
            if not (0.0 <= temperature <= 5.0):
                raise InvalidGenerationParameterError(
                    f"temperature must be in [0.0, 5.0], got {temperature}."
                )

        if top_p is not None:
            if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
                raise InvalidGenerationParameterError(
                    f"top_p must be a float, got {type(top_p).__name__}."
                )
            if not (0.0 < top_p <= 1.0):
                raise InvalidGenerationParameterError(
                    f"top_p must be in (0.0, 1.0], got {top_p}."
                )

        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise InvalidGenerationParameterError(
                    f"top_k must be an int, got {type(top_k).__name__}."
                )
            if top_k < 0:
                raise InvalidGenerationParameterError(
                    f"top_k must be >= 0 (0 disables top-k filtering), got {top_k}."
                )

        if repetition_penalty is not None:
            if not isinstance(repetition_penalty, (int, float)) or isinstance(
                repetition_penalty, bool
            ):
                raise InvalidGenerationParameterError(
                    "repetition_penalty must be a float, got "
                    f"{type(repetition_penalty).__name__}."
                )
            if not (0.0 < repetition_penalty <= 10.0):
                raise InvalidGenerationParameterError(
                    f"repetition_penalty must be in (0.0, 10.0], got {repetition_penalty}."
                )

    # ------------------------------------------------------------------
    # Validation & sanitisation (public API — backward-compatible)
    # ------------------------------------------------------------------

    def _validate_prompt(self, prompt: str) -> None:
        """Assert that ``prompt`` is a non-empty string.

        Args:
            prompt: The raw user-supplied prompt.

        Raises:
            InvalidPromptError: If ``prompt`` is empty or not a string.
        """
        if not isinstance(prompt, str):
            raise InvalidPromptError(
                f"Prompt must be a str, got {type(prompt).__name__}"
            )
        if not prompt.strip():
            raise InvalidPromptError("Prompt must not be empty or whitespace-only.")

    def _sanitize_prompt(self, prompt: str) -> str:
        """Strip leading/trailing whitespace from ``prompt``.

        Args:
            prompt: The raw prompt string.

        Returns:
            Sanitised prompt string.
        """
        return prompt.strip()

    # ------------------------------------------------------------------
    # Prompt hashing
    # ------------------------------------------------------------------

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Return a SHA-256 hex digest of ``prompt``.

        Prompts must never be logged in plaintext.  Log this hash instead.

        Args:
            prompt: Any string.

        Returns:
            64-character hex string.
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Context budget management
    # ------------------------------------------------------------------

    def reserve_generation_tokens(
        self, prompt_tokens: int, max_new_tokens: Optional[int] = None
    ) -> int:
        """Calculate the maximum new tokens available after the prompt.

        Args:
            prompt_tokens: Number of tokens already in the prompt.
            max_new_tokens: Caller-supplied cap, or ``None`` to use the
                configured default.

        Returns:
            Adjusted ``max_new_tokens`` that fits within the context window.
        """
        if self._gen_config is None:
            raise RuntimeError("GenerationConfig is not initialised.")
        limit = self._model_info.context_length
        cap = max_new_tokens if max_new_tokens is not None else self._gen_config.max_new_tokens
        if limit:
            cap = min(cap, limit - prompt_tokens - 1)
        return max(1, cap)

    def truncate_context(
        self,
        text: str,
        max_tokens: int,
        *,
        side: str = "left",
    ) -> str:
        """Truncate ``text`` to at most ``max_tokens`` tokens.

        Args:
            text: Any text to truncate.
            max_tokens: Maximum token budget.
            side: ``"left"`` to drop the oldest content (default) or
                ``"right"`` to drop the newest.

        Returns:
            Truncated string.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        ids = self._tokenizer.encode(text)
        if len(ids) <= max_tokens:
            return text
        if side == "left":
            ids = ids[-max_tokens:]
        else:
            ids = ids[:max_tokens]
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def estimate_remaining_tokens(self, prompt_tokens: int) -> int:
        """Estimate the number of tokens still available in the context.

        Args:
            prompt_tokens: Tokens already consumed by the prompt.

        Returns:
            Remaining tokens (may be 0 or negative if over budget).
        """
        limit = self._model_info.context_length
        if not limit:
            return 0
        return limit - prompt_tokens

    def context_budget_report(
        self,
        *,
        system: str = "",
        question: str = "",
        history: Optional[list[dict[str, str]]] = None,
        context: str = "",
        reserved_generation: Optional[int] = None,
    ) -> ContextBudget:
        """Produce a detailed token-budget breakdown for a RAG request.

        Args:
            system: System prompt text.
            question: User question text.
            history: Conversation history messages.
            context: Retrieved document/chunk context.
            reserved_generation: Tokens to reserve for the answer.
                Defaults to ``settings.MAX_NEW_TOKENS``.

        Returns:
            A populated :class:`ContextBudget` dataclass.
        """
        if self._gen_config is None:
            raise RuntimeError("GenerationConfig is not initialised.")
        reserved = reserved_generation or self._gen_config.max_new_tokens
        limit = self._model_info.context_length or getattr(
            settings, "MAX_CONTEXT_LENGTH", 32768
        )

        sys_tok = self.count_tokens(system) if system else 0
        q_tok = self.count_tokens(question) if question else 0
        hist_text = " ".join(m.get("content", "") for m in (history or []))
        hist_tok = self.count_tokens(hist_text) if hist_text else 0
        ctx_tok = self.count_tokens(context) if context else 0

        total_used = sys_tok + q_tok + hist_tok + ctx_tok + reserved
        available = limit - total_used

        return ContextBudget(
            context_length=limit,
            system_tokens=sys_tok,
            history_tokens=hist_tok,
            question_tokens=q_tok,
            context_tokens=ctx_tok,
            reserved_generation=reserved,
            available=available,
            over_budget=available < 0,
        )

    # ------------------------------------------------------------------
    # Prompt construction (backward-compatible public API)
    # ------------------------------------------------------------------

    def build_chat_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Format a chat prompt using the tokenizer's chat template.

        Args:
            user_message: The latest user turn.
            system_prompt: Optional system instruction override.
            history: Optional prior conversation turns.

        Returns:
            Fully-formatted prompt string ready for tokenisation.

        Raises:
            InvalidPromptError: If ``user_message`` fails validation.
        """
        self._validate_prompt(user_message)
        user_message = self._sanitize_prompt(user_message)

        sys_text: str = system_prompt or getattr(settings, "SYSTEM_PROMPT", "")
        messages: list[dict[str, str]] = []

        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        for turn in history or []:
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts: list[str] = []
        if sys_text:
            parts.append(sys_text)
        for turn in history or []:
            parts.append(f"{turn.get('role','').capitalize()}: {turn.get('content','')}")
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Tokenisation helpers (backward-compatible public API)
    # ------------------------------------------------------------------

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        return_tensors: str = "pt",
        use_cache: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Tokenise ``prompt`` (with optional LRU cache) and move tensors to device.

        Args:
            prompt: A formatted prompt string ready for the model.
            return_tensors: Framework identifier forwarded to the tokenizer;
                ``"pt"`` for PyTorch (the only supported value in this client).
            use_cache: If ``True``, check the LRU tokenisation cache before
                tokenising. Cache results are stored on CPU and moved to
                :attr:`_device` on retrieval, so the cached copy is
                device-agnostic.

        Returns:
            Dict of PyTorch tensors (``input_ids``, ``attention_mask``)
            on :attr:`_device`.

        Raises:
            RuntimeError: If the tokenizer has not been loaded.
            ContextLengthError: If the encoded prompt length meets or
                exceeds the model's maximum context window.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Ensure QwenClient initialised successfully."
            )

        if use_cache:
            cached = self._cache.get(prompt, self._device)
            if cached is not None:
                seq_len: int = cached["input_ids"].shape[-1]
                self._check_context_length(seq_len)
                return cached

        encoded = self._tokenizer(
            prompt,
            return_tensors=return_tensors,
            padding=False,
            truncation=False,
        )
        seq_len = encoded["input_ids"].shape[-1]
        self._check_context_length(seq_len)

        on_device = {k: v.to(self._device) for k, v in encoded.items()}

        if use_cache:
            self._cache.put(prompt, on_device)

        return on_device

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text``.

        Args:
            text: Any plain-text string.

        Returns:
            Token count as an integer.

        Raises:
            RuntimeError: If the tokenizer has not been loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Ensure QwenClient initialised successfully."
            )
        return len(self._tokenizer.encode(text))

    def _check_context_length(self, prompt_tokens: int) -> None:
        """Raise :class:`ContextLengthError` if the prompt fills the context window.

        Also logs a warning when the prompt consumes more than
        ``settings.CONTEXT_WARN_RATIO`` (default 85%) of the context window,
        giving callers an early signal to trim retrieved context before the
        hard limit is hit.

        Args:
            prompt_tokens: Number of tokens in the encoded prompt.

        Raises:
            ContextLengthError: If ``prompt_tokens`` meets or exceeds the
                model's maximum context window length.
        """
        limit = self._model_info.context_length
        if not limit:
            return  # Context length unknown; skip check.
        if prompt_tokens >= limit:
            raise ContextLengthError(
                f"Prompt length ({prompt_tokens:,} tokens) meets or exceeds the model's "
                f"context window ({limit:,} tokens). Reduce prompt or context size before "
                "retrying. Use context_budget_report() to diagnose token usage."
            )
        warn_ratio: float = getattr(settings, "CONTEXT_WARN_RATIO", 0.85)
        warn_threshold = int(limit * warn_ratio)
        if prompt_tokens > warn_threshold:
            logger.warning(
                "Prompt length %d tokens is >%.0f%% of the context window (%d tokens)."
                " Consider trimming retrieved context to leave headroom for generation.",
                prompt_tokens,
                warn_ratio * 100,
                limit,
            )

    # ------------------------------------------------------------------
    # Token budget utilities
    # ------------------------------------------------------------------

    def estimate_prompt_tokens(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> int:
        """Estimate tokens for a chat prompt without building the full string.

        Args:
            user_message: User turn text.
            system_prompt: Optional system text.
            history: Optional prior turns.

        Returns:
            Estimated token count.
        """
        prompt = self.build_chat_prompt(user_message, system_prompt, history)
        return self.count_tokens(prompt)

    def estimate_completion_tokens(self, text: str) -> int:
        """Estimate tokens for a completion text.

        Args:
            text: Completion / answer text.

        Returns:
            Estimated token count.
        """
        return self.count_tokens(text)

    def remaining_context(self, prompt_tokens: int) -> int:
        """Return tokens remaining after the prompt.

        Args:
            prompt_tokens: Tokens already consumed.

        Returns:
            Remaining token count.
        """
        return self.estimate_remaining_tokens(prompt_tokens)

    def token_budget(
        self,
        prompt_tokens: int,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, int]:
        """Return a token budget summary dict.

        Args:
            prompt_tokens: Tokens consumed by the prompt.
            max_new_tokens: Optional generation cap override.

        Returns:
            Dict with keys ``prompt``, ``max_new``, ``total``,
            ``context_window``, ``remaining``.
        """
        if self._gen_config is None:
            raise RuntimeError("GenerationConfig is not initialised.")
        cap = self.reserve_generation_tokens(prompt_tokens, max_new_tokens)
        limit = self._model_info.context_length
        return {
            "prompt": prompt_tokens,
            "max_new": cap,
            "total": prompt_tokens + cap,
            "context_window": limit,
            "remaining": max(0, limit - prompt_tokens - cap) if limit else 0,
        }

    # ------------------------------------------------------------------
    # Response decoding & cleanup (backward-compatible public API)
    # ------------------------------------------------------------------

    def _decode(self, output_ids: torch.Tensor, prompt_length: int) -> str:
        """Decode model output, stripping the echoed prompt tokens.

        Args:
            output_ids: Full output tensor (batch=1).
            prompt_length: Prompt tokens to skip.

        Returns:
            Decoded assistant text.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not loaded. Cannot decode model output."
            )
        new_ids = output_ids[0, prompt_length:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True)

    def cleanup_response(self, text: str) -> str:
        """Post-process decoded text: strip whitespace and role prefixes.

        Args:
            text: Raw decoded assistant text.

        Returns:
            Cleaned text string.
        """
        text = text.strip()
        for prefix in ("assistant\n", "assistant:", "Assistant:\n", "Assistant:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip()
                break
        return text

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        text: str,
        prompt_tokens: int,
        generated_tokens: int,
        latency: float,
        finish_reason: str,
        *,
        request_id: str = "",
        cache_hit: bool = False,
        tokenize_latency: float = 0.0,
        generate_latency: float = 0.0,
        decode_latency: float = 0.0,
        gpu_memory_before_mb: float = 0.0,
        retry_count: int = 0,
        validation_status: str = ValidationStatus.VALID,
        caller_metadata: Optional[dict[str, Any]] = None,
        queue_wait: float = 0.0,
        result_cache_hit: bool = False,
    ) -> GenerationResult:
        """Assemble a :class:`GenerationResult` from raw generation data.

        Args:
            text: Cleaned assistant response.
            prompt_tokens: Prompt token count.
            generated_tokens: New token count.
            latency: Total wall-clock time.
            finish_reason: Stop reason string.
            request_id: UUID for this request.
            cache_hit: Whether tokenisation came from cache.
            tokenize_latency: Seconds for tokenisation.
            generate_latency: Seconds inside ``model.generate()``.
            decode_latency: Seconds for decoding.
            gpu_memory_before_mb: GPU memory before generation (MiB).
            retry_count: Number of retries performed.
            validation_status: :class:`ValidationStatus` value.
            caller_metadata: Arbitrary caller dict.
            queue_wait: Seconds spent waiting for the generation lock.
            result_cache_hit: Whether the full result (not just
                tokenisation) was served from the result cache.

        Returns:
            Fully populated :class:`GenerationResult`.
        """
        tps = generated_tokens / latency if latency > 0 else 0.0
        gpu_now = self.gpu_memory_allocated()
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=prompt_tokens + generated_tokens,
            latency=latency,
            tokens_per_second=tps,
            finish_reason=finish_reason,
            model=self._model_info.model_name,
            device=str(self._device),
            gpu_memory_used_mb=gpu_now,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=request_id or str(uuid.uuid4()),
            cache_hit=cache_hit,
            tokenize_latency=tokenize_latency,
            generate_latency=generate_latency,
            decode_latency=decode_latency,
            gpu_memory_delta_mb=gpu_now - gpu_memory_before_mb,
            retry_count=retry_count,
            validation_status=validation_status,
            caller_metadata=caller_metadata or {},
            thread_id=threading.get_ident(),
            queue_wait=queue_wait,
            result_cache_hit=result_cache_hit,
        )

    def _record_request_history(
        self,
        *,
        request_id: str,
        prompt_hash: str,
        caller_metadata: dict[str, Any],
        queue_wait: float = 0.0,
        result: Optional[GenerationResult] = None,
        error: str = "",
    ) -> None:
        """Append a :class:`RequestRecord` to the bounded request-history
        ring buffer (no-op if disabled via ``settings.REQUEST_HISTORY_ENABLED``).

        Never stores the raw prompt text, only its hash, consistent with
        the rest of the client's "never log raw prompts" policy.

        Args:
            request_id: UUID for this request.
            prompt_hash: SHA-256 hex digest of the formatted prompt.
            caller_metadata: Arbitrary caller dict.
            queue_wait: Seconds spent waiting for the generation lock.
            result: The completed :class:`GenerationResult`, if successful.
            error: Exception message, if the request failed.
        """
        if not self._request_history_enabled:
            return
        record = RequestRecord(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_hash=prompt_hash,
            caller_metadata=caller_metadata or {},
            thread_id=threading.get_ident(),
            queue_wait=queue_wait,
            result=result,
            error=error,
        )
        with self._request_history_lock:
            self._request_history.append(record)

    def get_request_history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Return recent request records (most recent last).

        Args:
            limit: If provided, return at most this many of the most
                recent records.

        Returns:
            List of :class:`RequestRecord` dictionaries.
        """
        with self._request_history_lock:
            records = list(self._request_history)
        if limit is not None:
            records = records[-limit:]
        return [r.to_dict() for r in records]

    def get_request_by_id(self, request_id: str) -> Optional[dict[str, Any]]:
        """Look up a single request record by its ``request_id``.

        Args:
            request_id: The UUID to search for.

        Returns:
            The matching record as a dict, or ``None`` if not found (it may
            have aged out of the bounded history buffer).
        """
        with self._request_history_lock:
            for record in reversed(self._request_history):
                if record.request_id == request_id:
                    return record.to_dict()
        return None

    def _detect_finish_reason(
        self,
        output_ids: torch.Tensor,
        max_new_tokens: int,
        generated_tokens: int,
    ) -> str:
        """Detect why generation stopped.

        Args:
            output_ids: Full model output tensor.
            max_new_tokens: The generation cap that was in effect.
            generated_tokens: Actual new tokens produced.

        Returns:
            One of ``"length"``, ``"eos"``, ``"unknown"``.
        """
        if self._tokenizer is None:
            return FinishReason.UNKNOWN.value
        if generated_tokens >= max_new_tokens:
            return FinishReason.LENGTH.value
        last_token = int(output_ids[0, -1].item())
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None:
            if isinstance(eos_id, (list, tuple)):
                if last_token in eos_id:
                    return FinishReason.EOS.value
            elif last_token == eos_id:
                return FinishReason.EOS.value
        return FinishReason.UNKNOWN.value

    # ------------------------------------------------------------------
    # Core generation (internal — backward-compatible public API)
    # ------------------------------------------------------------------

    def _generate(
        self,
        inputs: dict[str, torch.Tensor],
        **gen_kwargs: Any,
    ) -> tuple[torch.Tensor, str, int]:
        """Low-level generation call with OOM recovery and retry.

        Args:
            inputs: Tokenised prompt tensors on the correct device.
            **gen_kwargs: Keyword arguments forwarded to ``model.generate``.

        Returns:
            3-tuple of ``(output_ids, finish_reason, generated_tokens)``.

        Raises:
            GPUOutOfMemoryError: If CUDA OOM persists after cache clear.
            GenerationError: For any other failure.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call reload_model() before generating.")
        if self._gen_config is None:
            raise RuntimeError("GenerationConfig is not initialised.")

        max_new_tokens: int = gen_kwargs.get(
            "max_new_tokens", self._gen_config.max_new_tokens
        )
        prompt_len: int = inputs["input_ids"].shape[-1]

        try:
            with torch.inference_mode():
                output_ids: torch.Tensor = self._model.generate(
                    **inputs, generation_config=self._gen_config, **gen_kwargs
                )
        except torch.cuda.OutOfMemoryError as exc:
            logger.error(
                "CUDA OOM during generation (prompt_len=%d, max_new=%d, "
                "gpu_allocated=%.1f MiB) – clearing cache and retrying …",
                prompt_len,
                max_new_tokens,
                self.gpu_memory_allocated(),
            )
            self._metrics.record_oom()
            self._telemetry.record_oom()
            self.clear_gpu_cache()
            try:
                with torch.inference_mode():
                    output_ids = self._model.generate(
                        **inputs, generation_config=self._gen_config, **gen_kwargs
                    )
            except torch.cuda.OutOfMemoryError as exc2:
                logger.error(
                    "CUDA OOM persists after cache clear (gpu_allocated=%.1f MiB). "
                    "Reduce max_new_tokens or prompt size.",
                    self.gpu_memory_allocated(),
                )
                raise GPUOutOfMemoryError(
                    f"GPU OOM persists after cache clear: {exc2}. "
                    "Reduce max_new_tokens or prompt length and retry."
                ) from exc2
        except Exception as exc:
            raise GenerationError(
                f"model.generate() failed unexpectedly: {exc}"
            ) from exc

        generated_tokens = output_ids.shape[-1] - prompt_len
        finish_reason = self._detect_finish_reason(
            output_ids, max_new_tokens, generated_tokens
        )
        return output_ids, finish_reason, generated_tokens

    # ------------------------------------------------------------------
    # Auto memory optimisation
    # ------------------------------------------------------------------

    def _maybe_auto_cleanup(self) -> None:
        """Trigger periodic GPU memory optimisation if configured.

        Runs every ``_auto_cleanup_interval`` requests.
        """
        if self._auto_cleanup_interval <= 0:
            return
        self._auto_cleanup_counter += 1
        if self._auto_cleanup_counter >= self._auto_cleanup_interval:
            self._auto_cleanup_counter = 0
            self.auto_cleanup()

    def auto_cleanup(self) -> None:
        """Run garbage collection and GPU cache / peak-stats reset.

        Called automatically every :attr:`_auto_cleanup_interval` requests
        when that setting is greater than zero. Safe to call manually at any
        time (e.g. between large document batches).
        """
        before = self.gpu_memory_allocated()
        gc.collect()
        self.clear_gpu_cache()
        self.reset_peak_memory()
        after = self.gpu_memory_allocated()
        logger.debug(
            "auto_cleanup: gc + GPU cache cleared | gpu_before=%.1f MiB -> after=%.1f MiB",
            before,
            after,
        )

    # ------------------------------------------------------------------
    # Public generation API (backward-compatible)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
        validate_output: bool = True,
        use_result_cache: Optional[bool] = None,
    ) -> GenerationResult:
        """Generate a response for ``prompt`` and return a :class:`GenerationResult`.

        Thread-safe: concurrent callers queue on :attr:`_generate_lock`. Each
        call creates a temporary :class:`~transformers.GenerationConfig` from
        the runtime overrides without permanently modifying the default config,
        so concurrent or sequential calls with different parameters do not
        interfere.

        Args:
            prompt: Raw user message or pre-formatted prompt string.
            system_prompt: Optional system instruction override. Defaults to
                ``settings.SYSTEM_PROMPT`` when omitted.
            history: Optional prior conversation turns as a list of
                ``{"role": str, "content": str}`` dicts.
            max_new_tokens: Override for maximum tokens to generate. Must be
                a positive integer if supplied.
            temperature: Override for sampling temperature in ``[0.0, 5.0]``.
            top_p: Override for nucleus-sampling probability mass in
                ``(0.0, 1.0]``.
            top_k: Override for top-k candidate count (``0`` disables).
            repetition_penalty: Override for the repetition penalty in
                ``(0.0, 10.0]``.
            do_sample: Override for stochastic sampling flag.
            metadata: Arbitrary caller metadata dict. Stored in the
                :class:`GenerationResult` and request history but never
                logged in plaintext.
            validate_output: If ``True`` (default), run
                :meth:`validate_response` on the decoded text and log a
                warning when validation fails. Does not raise on failure.
            use_result_cache: Explicitly enable (``True``) or disable
                (``False``) the full-result cache for this call. ``None``
                (default) defers to the client-wide
                :attr:`_result_cache_enabled` setting.

        Returns:
            A fully populated :class:`GenerationResult`.

        Raises:
            RuntimeError: If the client is not yet fully initialised.
            InvalidPromptError: If the prompt is empty or not a string.
            SafetyError: If the prompt violates a safety policy.
            ContextLengthError: If the encoded prompt meets or exceeds the
                model's context window.
            InvalidGenerationParameterError: If a sampling override is
                outside its valid range.
            GPUOutOfMemoryError: If a CUDA OOM persists after cache clear.
            GenerationError: On any other inference failure.
        """
        if not self._initialised:
            raise RuntimeError("QwenClient is not yet fully initialised.")

        request_id = str(uuid.uuid4())
        t_wall_start = time.perf_counter()

        # Validate sampling-parameter overrides up front (spec #14) so that
        # a bad value fails fast with a descriptive error instead of
        # surfacing as an opaque exception deep inside model.generate().
        self.validate_generation_parameters(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # Build prompt
        formatted = self.build_chat_prompt(prompt, system_prompt, history)
        prompt_hash = self.hash_prompt(formatted)
        logger.debug(
            "generate() | req=%s | prompt_hash=%s | meta=%s",
            request_id,
            prompt_hash,
            bool(metadata),
        )

        # Safety check
        self.validate_prompt_safety(formatted)

        # Assemble runtime config overrides
        gen_overrides: dict[str, Any] = {}
        if max_new_tokens is not None:
            gen_overrides["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_overrides["temperature"] = temperature
        if top_p is not None:
            gen_overrides["top_p"] = top_p
        if top_k is not None:
            gen_overrides["top_k"] = top_k
        if repetition_penalty is not None:
            gen_overrides["repetition_penalty"] = repetition_penalty
        if do_sample is not None:
            gen_overrides["do_sample"] = do_sample

        if self._gen_config is None:
            raise RuntimeError(
                "GenerationConfig is not initialised. "
                "Ensure QwenClient initialised successfully."
            )
        if gen_overrides:
            active_cfg = GenerationConfig(
                **{**self._gen_config.to_dict(), **gen_overrides}
            )
        else:
            active_cfg = self._gen_config

        # -- Result cache lookup (spec #4) --
        # A hit short-circuits tokenisation/generation entirely. Caching is
        # keyed on (prompt_hash, sampling-parameter signature) so a cached
        # answer is never returned for a request asking for different
        # sampling behaviour (e.g. a colder temperature) on the same prompt.
        cache_enabled = (
            use_result_cache if use_result_cache is not None
            else self._result_cache_enabled
        )
        gen_signature = self.hash_prompt(json.dumps(active_cfg.to_dict(), sort_keys=True, default=str))
        if cache_enabled:
            try:
                cached_entry = self._result_cache.get(prompt_hash)
            except Exception as exc:  # backend failure degrades to a miss
                logger.warning("Result cache get() failed (treated as miss): %s", exc)
                cached_entry = None
            if cached_entry is not None and cached_entry.gen_signature == gen_signature:
                self._fire(HookEvent.ON_CACHE_HIT, request_id=request_id, prompt_hash=prompt_hash)
                cached_result = cached_entry.result
                # Return a fresh copy stamped with this request's own id /
                # timestamp / queue_wait so callers can still distinguish
                # repeated cache hits in logs and tracing.
                hit_result = GenerationResult(
                    **{
                        **cached_result.to_dict(),
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "queue_wait": 0.0,
                        "result_cache_hit": True,
                        "thread_id": threading.get_ident(),
                    }
                )
                self._metrics.record_success(hit_result)
                self._telemetry.record(hit_result)
                self._advanced_metrics.request_finished(
                    latency=hit_result.latency, success=True, cache_hit=True
                )
                logger.info(
                    "Generation served from result cache | req=%s | hash=%s",
                    request_id,
                    prompt_hash,
                )
                return hit_result
            self._fire(HookEvent.ON_CACHE_MISS, request_id=request_id, prompt_hash=prompt_hash)

        # Queue wait measurement
        self._advanced_metrics.request_started()
        t_queue_start = time.perf_counter()
        with self._generate_lock:
            queue_wait = time.perf_counter() - t_queue_start
            try:
                # -- Tokenise (with cache) --
                self._fire(HookEvent.BEFORE_TOKENIZE, prompt_hash=prompt_hash)
                t_tok_start = time.perf_counter()
                cache_hit = self._cache.get(formatted, self._device) is not None
                inputs = self.tokenize_prompt(formatted, use_cache=True)
                tokenize_latency = time.perf_counter() - t_tok_start
                self._fire(HookEvent.AFTER_TOKENIZE, prompt_hash=prompt_hash)

                prompt_tokens: int = inputs["input_ids"].shape[-1]

                # Adjust max_new_tokens to fit context
                adj_max_new = self.reserve_generation_tokens(
                    prompt_tokens, active_cfg.max_new_tokens
                )
                if adj_max_new != active_cfg.max_new_tokens:
                    active_cfg = GenerationConfig(
                        **{**active_cfg.to_dict(), "max_new_tokens": adj_max_new}
                    )

                logger.debug(
                    "Generating | req=%s | prompt_tokens=%d | max_new=%d"
                    " | temp=%.2f | top_p=%.2f | gpu_alloc=%.1f MiB",
                    request_id,
                    prompt_tokens,
                    active_cfg.max_new_tokens,
                    active_cfg.temperature,
                    active_cfg.top_p,
                    self.gpu_memory_allocated(),
                )

                gpu_before = self.gpu_memory_allocated()
                with self._gen_config_lock:
                    original_cfg = self._gen_config
                    self._gen_config = active_cfg

                # -- Generation with retry --
                self._fire(HookEvent.BEFORE_GENERATION, request_id=request_id)
                t_gen_start = time.perf_counter()
                output_ids, finish_reason, generated_tokens = self._generate_with_retry(
                    inputs
                )
                generate_latency = time.perf_counter() - t_gen_start
                with self._gen_config_lock:
                    self._gen_config = original_cfg

                # -- Decode --
                self._fire(HookEvent.BEFORE_DECODE, request_id=request_id)
                t_dec_start = time.perf_counter()
                raw_text = self._decode(output_ids, prompt_tokens)
                cleaned = self.cleanup_response(raw_text)
                decode_latency = time.perf_counter() - t_dec_start
                self._fire(HookEvent.AFTER_DECODE, request_id=request_id)

                total_latency = time.perf_counter() - t_wall_start

                # -- Validate response --
                val_status = ValidationStatus.VALID
                if validate_output:
                    val = self.validate_response(cleaned)
                    val_status = val.status
                    if not val.valid:
                        logger.warning(
                            "Response validation failed | req=%s | status=%s | %s",
                            request_id,
                            val.status,
                            val.message,
                        )

                result = self._build_result(
                    cleaned,
                    prompt_tokens,
                    generated_tokens,
                    total_latency,
                    finish_reason,
                    request_id=request_id,
                    cache_hit=cache_hit,
                    tokenize_latency=tokenize_latency,
                    generate_latency=generate_latency,
                    decode_latency=decode_latency,
                    gpu_memory_before_mb=gpu_before,
                    retry_count=getattr(output_ids, "_retry_count", 0),
                    validation_status=val_status,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    result_cache_hit=False,
                )

                self._metrics.record_success(result)
                self._telemetry.record(result)
                self._advanced_metrics.request_finished(
                    latency=total_latency,
                    success=True,
                    queue_wait=queue_wait,
                    cache_hit=cache_hit,
                )
                self._fire(HookEvent.AFTER_GENERATION, result=result)
                self._maybe_auto_cleanup()

                # -- Result cache store (spec #4) --
                if cache_enabled:
                    try:
                        self._result_cache.put(
                            prompt_hash,
                            CachedGeneration(
                                prompt_hash=prompt_hash,
                                result=result,
                                created_at=time.time(),
                                gen_signature=gen_signature,
                            ),
                        )
                    except Exception as exc:  # never let caching break generation
                        logger.warning("Result cache put() failed (ignored): %s", exc)

                self._record_request_history(
                    request_id=request_id,
                    prompt_hash=prompt_hash,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    result=result,
                )

                logger.info(
                    "Generation complete | req=%s | prompt_tokens=%d"
                    " | generated_tokens=%d | latency=%.3fs | tps=%.1f"
                    " | finish=%s | tok_cache=%s",
                    request_id,
                    result.prompt_tokens,
                    result.generated_tokens,
                    result.latency,
                    result.tokens_per_second,
                    result.finish_reason,
                    result.cache_hit,
                )
                return result

            except (InvalidPromptError, ContextLengthError, SafetyError, InvalidGenerationParameterError) as exc:
                self._metrics.record_failure()
                self._telemetry.record_error()
                self._advanced_metrics.request_finished(
                    latency=time.perf_counter() - t_wall_start, success=False, queue_wait=queue_wait
                )
                self._record_request_history(
                    request_id=request_id,
                    prompt_hash=prompt_hash,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    error=str(exc),
                )
                self._fire(HookEvent.ON_ERROR, request_id=request_id)
                raise
            except GPUOutOfMemoryError as exc:
                self._metrics.record_failure()
                self._telemetry.record_error()
                self._telemetry.record_oom()
                self._advanced_metrics.request_finished(
                    latency=time.perf_counter() - t_wall_start,
                    success=False,
                    queue_wait=queue_wait,
                    oom=True,
                )
                self._record_request_history(
                    request_id=request_id,
                    prompt_hash=prompt_hash,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    error=str(exc),
                )
                self._fire(HookEvent.ON_ERROR, request_id=request_id)
                raise
            except GenerationError as exc:
                self._metrics.record_failure()
                self._telemetry.record_error()
                self._advanced_metrics.request_finished(
                    latency=time.perf_counter() - t_wall_start, success=False, queue_wait=queue_wait
                )
                self._record_request_history(
                    request_id=request_id,
                    prompt_hash=prompt_hash,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    error=str(exc),
                )
                self._fire(HookEvent.ON_ERROR, request_id=request_id)
                raise
            except Exception as exc:
                self._metrics.record_failure()
                self._telemetry.record_error()
                self._advanced_metrics.request_finished(
                    latency=time.perf_counter() - t_wall_start, success=False, queue_wait=queue_wait
                )
                self._record_request_history(
                    request_id=request_id,
                    prompt_hash=prompt_hash,
                    caller_metadata=metadata or {},
                    queue_wait=queue_wait,
                    error=str(exc),
                )
                self._fire(HookEvent.ON_ERROR, request_id=request_id)
                logger.exception("Unexpected error in generate() | req=%s: %s", request_id, exc)
                raise GenerationError(f"Unexpected generation failure: {exc}") from exc

    def _generate_with_retry(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, str, int]:
        """Invoke :meth:`_generate` with configurable retry logic.

        Args:
            inputs: Tokenised prompt tensors.

        Returns:
            3-tuple of ``(output_ids, finish_reason, generated_tokens)``.

        Raises:
            GPUOutOfMemoryError: On persistent OOM.
            GenerationError: On non-recoverable failure.
        """
        last_exc: Exception = GenerationError("Unknown error.")
        retry_count = 0
        for attempt in range(self._retry_manager.max_retries + 1):
            try:
                output_ids, finish_reason, generated_tokens = self._generate(inputs)
                # Attach retry count as attribute for upstream tracking
                output_ids._retry_count = retry_count  # type: ignore[attr-defined]
                return output_ids, finish_reason, generated_tokens
            except (GPUOutOfMemoryError, GenerationError) as exc:
                last_exc = exc
                if not self._retry_manager.is_recoverable(exc) or attempt == self._retry_manager.max_retries:
                    raise
                retry_count += 1
                # NOTE: _generate() already increments the OOM counter once,
                # at the point the actual torch.cuda.OutOfMemoryError is
                # caught, before re-raising as GPUOutOfMemoryError. This
                # branch must NOT call record_oom() again for that same
                # exception, and must NOT call it at all for a plain
                # GenerationError (a non-OOM, but still recoverable,
                # failure) — doing so previously inflated oom_count for
                # every recoverable retry regardless of cause.
                logger.warning(
                    "Recoverable error on attempt %d/%d – retrying in %.1fs: %s",
                    attempt + 1,
                    self._retry_manager.max_retries,
                    self._retry_manager.backoff_base * (2 ** attempt),
                    exc,
                )
                self.clear_gpu_cache()
                self._retry_manager.sleep(attempt)
        raise last_exc

    # ------------------------------------------------------------------
    # Streaming Generation
    # ------------------------------------------------------------------

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: float = 120.0,
        metadata: Optional[dict[str, Any]] = None,
        callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        request_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream the generation token by token.

        Yields decoded text chunks as they are produced.  Compatible with
        FastAPI ``StreamingResponse`` via ``iter()`` wrapping.

        Args:
            prompt: Raw user message.
            system_prompt: Optional system instruction override.
            history: Optional prior turns.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature override.
            top_p: Nucleus-sampling override.
            timeout: Maximum seconds to wait for the streamer.
            metadata: Arbitrary caller metadata.
            callback: Optional function invoked synchronously with each
                decoded chunk, in addition to it being yielded. Useful for
                side-channel logging/metrics without consuming the
                generator. Exceptions raised by ``callback`` are caught and
                logged, never propagated (a broken callback must not break
                streaming).
            cancel_event: Optional externally-owned :class:`threading.Event`.
                If not provided, an internal one is created and registered
                under ``request_id`` so :meth:`cancel_stream` can signal it.
                Setting the event (from any thread) stops generation before
                the next token is produced.
            request_id: Optional caller-supplied request id. If omitted, a
                new UUID is generated. Exposed so callers can call
                :meth:`cancel_stream` with a known id before the generator
                has yielded anything.

        Yields:
            Decoded text chunks (strings).

        Raises:
            InvalidPromptError: On invalid prompt.
            SafetyError: On safety violation.
            InvalidGenerationParameterError: If a sampling override is out
                of range.
            GenerationError: On generation failure.
        """
        if not self._initialised:
            raise RuntimeError("QwenClient is not yet fully initialised.")

        if self._model is None:
            raise RuntimeError("Model is not loaded. Call reload_model() before generating.")
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        if self._gen_config is None:
            raise RuntimeError("GenerationConfig is not initialised.")

        self.validate_generation_parameters(
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )

        req_id = request_id or str(uuid.uuid4())
        formatted = self.build_chat_prompt(prompt, system_prompt, history)
        self.validate_prompt_safety(formatted)

        logger.debug("stream_generate() | req=%s | hash=%s", req_id, self.hash_prompt(formatted))

        gen_overrides: dict[str, Any] = {}
        if max_new_tokens is not None:
            gen_overrides["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_overrides["temperature"] = temperature
        if top_p is not None:
            gen_overrides["top_p"] = top_p

        with self._gen_config_lock:
            base_cfg = self._gen_config
        if gen_overrides:
            active_cfg = GenerationConfig(**{**base_cfg.to_dict(), **gen_overrides})
        else:
            active_cfg = base_cfg

        inputs = self.tokenize_prompt(formatted, use_cache=False)
        prompt_tokens: int = inputs["input_ids"].shape[-1]
        adj_max = self.reserve_generation_tokens(prompt_tokens, active_cfg.max_new_tokens)
        if adj_max != active_cfg.max_new_tokens:
            active_cfg = GenerationConfig(**{**active_cfg.to_dict(), "max_new_tokens": adj_max})

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Cancellation: register the event so cancel_stream(req_id) can
        # reach this specific generation from another thread/request.
        event = cancel_event or threading.Event()
        with self._stream_cancel_lock:
            self._stream_cancel_events[req_id] = event
        stopping_criteria = StoppingCriteriaList([_CancellationStoppingCriteria(event)])

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "generation_config": active_cfg,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
        }

        exc_holder: list[Exception] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**gen_kwargs)
            except Exception as exc:
                exc_holder.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        self._metrics.record_streaming()
        try:
            for chunk in streamer:
                if exc_holder:
                    raise GenerationError(
                        f"Streaming generation failed: {exc_holder[0]}"
                    ) from exc_holder[0]
                if callback is not None:
                    try:
                        callback(chunk)
                    except Exception as cb_exc:
                        logger.warning(
                            "stream_generate() callback raised (ignored) | req=%s: %s",
                            req_id,
                            cb_exc,
                        )
                if event.is_set():
                    self._fire(HookEvent.ON_STREAM_CANCEL, request_id=req_id)
                    logger.info("stream_generate() cancelled | req=%s", req_id)
                    break
                self._fire(HookEvent.ON_STREAM_TOKEN, request_id=req_id, chunk=chunk)
                yield chunk

            t.join(timeout=timeout)
            if exc_holder and not event.is_set():
                raise GenerationError(
                    f"Streaming generation failed: {exc_holder[0]}"
                ) from exc_holder[0]
        finally:
            with self._stream_cancel_lock:
                self._stream_cancel_events.pop(req_id, None)

        logger.debug("stream_generate() complete | req=%s", req_id)

    # Alias matching the exact name requested by the production-hardening
    # spec (item 1: `generate_stream()`). `stream_generate` remains the
    # primary, original implementation; this is a zero-overhead passthrough
    # so both names work identically and neither call site needs to change.
    generate_stream = stream_generate

    def cancel_stream(self, request_id: str) -> bool:
        """Signal cancellation for an in-flight :meth:`stream_generate` call.

        Args:
            request_id: The ``request_id`` passed to (or returned by) the
                target :meth:`stream_generate` call.

        Returns:
            ``True`` if a matching in-flight stream was found and signalled,
            ``False`` if no stream with that id is currently active (it may
            have already finished, or the id may be unknown).
        """
        with self._stream_cancel_lock:
            event = self._stream_cancel_events.get(request_id)
        if event is None:
            return False
        event.set()
        logger.info("cancel_stream() | req=%s", request_id)
        return True

    def active_stream_ids(self) -> list[str]:
        """Return the request ids of all currently in-flight streams."""
        with self._stream_cancel_lock:
            return list(self._stream_cancel_events.keys())

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: float = 120.0,
    ) -> Generator[str, None, None]:
        """Streaming generation from a pre-built message list.

        Args:
            messages: List of ``{"role": …, "content": …}`` dicts.  The
                last message must have ``role=="user"``.
            system_prompt: Optional system instruction.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature override.
            top_p: Nucleus-sampling override.
            timeout: Maximum streamer timeout.

        Yields:
            Decoded text chunks.

        Raises:
            InvalidPromptError: If the last message is not a user turn.
        """
        if not messages or messages[-1].get("role") != "user":
            raise InvalidPromptError(
                "stream_chat() requires the last message to have role='user'."
            )
        user_message = messages[-1]["content"]
        history = messages[:-1]
        yield from self.stream_generate(
            user_message,
            system_prompt=system_prompt,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )

    def stream_from_context(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ) -> Generator[str, None, None]:
        """Streaming RAG generation from a retrieved context string.

        Args:
            question: User question.
            context: Retrieved document text.
            system_prompt: Optional system instruction.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            timeout: Maximum streamer timeout.

        Yields:
            Decoded text chunks.
        """
        combined = f"Context:\n{context}\n\nQuestion:\n{question}"
        yield from self.stream_generate(
            combined,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Batch Generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        validate_outputs: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[GenerationResult]:
        """Generate responses for multiple prompts sequentially.

        Batch semantics: each prompt is processed independently via
        :meth:`generate` so that GPU memory usage stays predictable and
        per-item error handling is straightforward.  A future upgrade can
        replace this with true parallel batching when the model supports it.

        Args:
            prompts: List of user messages.
            system_prompt: Shared system instruction.
            max_new_tokens: Shared generation cap.
            temperature: Shared temperature.
            top_p: Shared nucleus-sampling value.
            validate_outputs: Whether to validate each response.
            metadata: Shared caller metadata.

        Returns:
            List of :class:`GenerationResult` objects (one per prompt).
        """
        self._metrics.record_batch()
        results: list[GenerationResult] = []
        for p in prompts:
            result = self.generate(
                p,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                metadata=metadata,
                validate_output=validate_outputs,
            )
            results.append(result)
        logger.info("generate_batch() | count=%d", len(results))
        return results

    def stream_batch(
        self,
        prompts: list[str],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ) -> Generator[tuple[int, str], None, None]:
        """Stream responses for multiple prompts, yielding ``(index, chunk)`` tuples.

        Args:
            prompts: List of user messages.
            system_prompt: Shared system instruction.
            max_new_tokens: Shared generation cap.
            temperature: Shared temperature.
            timeout: Per-prompt streamer timeout.

        Yields:
            2-tuples of ``(prompt_index, text_chunk)``.
        """
        self._metrics.record_batch()
        for idx, p in enumerate(prompts):
            for chunk in self.stream_generate(
                p,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout=timeout,
            ):
                yield (idx, chunk)

    def batch_generate_from_context(
        self,
        questions: list[str],
        contexts: list[str],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[GenerationResult]:
        """RAG batch: pair each question with its retrieved context.

        Args:
            questions: User questions (one per item).
            contexts: Retrieved context strings (parallel with ``questions``).
            system_prompt: Shared system instruction.
            max_new_tokens: Shared generation cap.

        Returns:
            List of :class:`GenerationResult` objects.

        Raises:
            ValueError: If ``questions`` and ``contexts`` have different lengths.
        """
        if len(questions) != len(contexts):
            raise ValueError(
                f"questions ({len(questions)}) and contexts ({len(contexts)}) "
                "must have the same length."
            )
        combined = [
            f"Context:\n{ctx}\n\nQuestion:\n{q}"
            for q, ctx in zip(questions, contexts)
        ]
        return self.generate_batch(
            combined, system_prompt=system_prompt, max_new_tokens=max_new_tokens
        )

    # ------------------------------------------------------------------
    # Async APIs
    # ------------------------------------------------------------------

    async def async_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        validate_output: bool = True,
    ) -> GenerationResult:
        """Async wrapper around :meth:`generate`.

        Runs the synchronous generation in the default thread-pool executor
        so the event loop is never blocked. Compatible with FastAPI and
        other asyncio-based frameworks.

        Args:
            prompt: Raw user message.
            system_prompt: Optional system instruction override.
            history: Optional prior conversation turns.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature override.
            top_p: Nucleus-sampling probability mass override.
            metadata: Arbitrary caller metadata.
            validate_output: Whether to run :meth:`validate_response` on the
                generated text.

        Returns:
            :class:`GenerationResult`.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt,
                system_prompt=system_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                metadata=metadata,
                validate_output=validate_output,
            ),
        )

    async def async_generate_batch(
        self,
        prompts: list[str],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[GenerationResult]:
        """Async wrapper around :meth:`generate_batch`.

        Args:
            prompts: User messages.
            system_prompt: Shared system instruction.
            max_new_tokens: Shared generation cap.
            temperature: Shared temperature.
            metadata: Shared caller metadata.

        Returns:
            List of :class:`GenerationResult` objects.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_batch(
                prompts,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                metadata=metadata,
            ),
        )

    async def async_stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
        request_id: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> AsyncIterator[str]:
        """Async generator wrapping :meth:`stream_generate`.

        Suitable for use with ``async for chunk in client.async_stream_generate(…)``.

        Args:
            prompt: Raw user message.
            system_prompt: Optional system instruction.
            history: Optional prior turns.
            max_new_tokens: Max tokens override.
            temperature: Temperature override.
            timeout: Streamer timeout.
            request_id: Optional id, forwarded to :meth:`stream_generate` so
                :meth:`cancel_stream` can target this specific call.
            cancel_event: Optional pre-created cancellation event, forwarded
                to :meth:`stream_generate`.

        Yields:
            Decoded text chunks.
        """
        loop = asyncio.get_running_loop()
        gen = self.stream_generate(
            prompt,
            system_prompt=system_prompt,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout=timeout,
            request_id=request_id,
            cancel_event=cancel_event,
        )
        # Bridge sync generator to async generator via executor
        sentinel = object()
        q: asyncio.Queue[Any] = asyncio.Queue()

        def _producer() -> None:
            try:
                for chunk in gen:
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(sentinel), loop)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            item = await q.get()
            if item is sentinel:
                break
            yield item

    # Alias matching the exact name requested by the production-hardening
    # spec (item 3: `async_stream()`). Binding an async generator function
    # under a second name works exactly like binding any other function;
    # both names invoke the identical implementation.
    async_stream = async_stream_generate

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(
        self,
        *,
        prompt: str = "Hello.",
        max_new_tokens: int = 16,
    ) -> None:
        """Run a single warmup inference pass to initialise CUDA kernels.

        Args:
            prompt: Warmup prompt string.
            max_new_tokens: Token budget for the warmup pass.

        Raises:
            WarmupError: If the warmup inference fails.
        """
        logger.info("Running model warmup (prompt=%r, max_new_tokens=%d) …", prompt, max_new_tokens)
        try:
            formatted = self.build_chat_prompt(prompt)
            inputs = self.tokenize_prompt(formatted, use_cache=False)
            if self._model is None:
                raise RuntimeError("Model is not loaded.")
            if self._gen_config is None:
                raise RuntimeError("GenerationConfig is not initialised.")
            cfg = GenerationConfig(
                **{**self._gen_config.to_dict(), "max_new_tokens": max_new_tokens}
            )
            with torch.inference_mode():
                self._model.generate(**inputs, generation_config=cfg)
            self._warmup_complete = True
            logger.info("Warmup complete (max_new_tokens=%d).", max_new_tokens)
        except Exception as exc:
            raise WarmupError(f"Warmup failed: {exc}") from exc

    def warmup_batch(
        self,
        prompts: Optional[list[str]] = None,
        max_new_tokens: int = 16,
    ) -> None:
        """Run warmup for a list of prompts.

        Args:
            prompts: Optional list of warmup prompts.  Defaults to a small
                built-in set covering different lengths.
            max_new_tokens: Token budget per warmup pass.
        """
        default_prompts = ["Hi.", "Explain AI.", "List three facts about Python."]
        for p in prompts or default_prompts:
            self.warmup(prompt=p, max_new_tokens=max_new_tokens)

    def warmup_complete(self) -> bool:
        """Return ``True`` if at least one warmup pass has completed.

        Returns:
            Boolean warmup status.
        """
        return self._warmup_complete

    # ------------------------------------------------------------------
    # GPU Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def gpu_available() -> bool:
        """Return ``True`` if at least one CUDA device is available."""
        return torch.cuda.is_available()

    def gpu_memory_allocated(self) -> float:
        """Return currently allocated GPU memory in MiB."""
        if not self.gpu_available():
            return 0.0
        return torch.cuda.memory_allocated(self._device) / (1024 ** 2)

    def gpu_memory_reserved(self) -> float:
        """Return currently reserved (cached) GPU memory in MiB."""
        if not self.gpu_available():
            return 0.0
        return torch.cuda.memory_reserved(self._device) / (1024 ** 2)

    def gpu_peak_memory(self) -> float:
        """Return peak GPU memory allocation seen this process in MiB."""
        if not self.gpu_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)

    def gpu_summary(self) -> dict[str, Any]:
        """Return a structured snapshot of GPU memory usage.

        On CPU-only systems (``torch.cuda.is_available()`` returns ``False``),
        returns a minimal dict with ``available=False`` and all other fields
        absent, so callers can branch on that key without raising.

        Returns:
            Dict with keys:
              - ``available`` (bool): Whether CUDA is available.
              - ``device`` (str): Device string, e.g. ``"cuda:0"``.
              - ``name`` (str): GPU display name.
              - ``allocated_mb`` (float): Currently allocated VRAM (MiB).
              - ``reserved_mb`` (float): Memory reserved by the CUDA caching
                  allocator but not yet allocated to tensors (MiB).
              - ``peak_mb`` (float): Peak allocated VRAM since last reset (MiB).
              - ``total_mb`` (float): Total VRAM on the device (MiB).
              - ``free_mb`` (float): Estimated free VRAM (total − reserved, MiB).
        """
        if not self.gpu_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(self._device)
        total_mb = props.total_memory / (1024 ** 2)
        reserved_mb = self.gpu_memory_reserved()
        free_mb = total_mb - reserved_mb
        return {
            "available": True,
            "device": str(self._device),
            "name": torch.cuda.get_device_name(self._device),
            "allocated_mb": round(self.gpu_memory_allocated(), 2),
            "reserved_mb": round(reserved_mb, 2),
            "peak_mb": round(self.gpu_peak_memory(), 2),
            "total_mb": round(total_mb, 2),
            "free_mb": round(free_mb, 2),
        }

    def clear_gpu_cache(self) -> None:
        """Release unused cached GPU memory back to the CUDA allocator.

        Calls ``torch.cuda.empty_cache()``. This does not free memory still
        referenced by live tensors, but returns the cached allocator pages
        to the OS/driver so other processes can use them. A no-op on CPU.
        """
        if self.gpu_available():
            before = self.gpu_memory_reserved()
            torch.cuda.empty_cache()
            after = self.gpu_memory_reserved()
            logger.debug(
                "GPU cache cleared | reserved %.1f MiB -> %.1f MiB (freed %.1f MiB)",
                before, after, max(0.0, before - after),
            )

    def reset_peak_memory(self) -> None:
        """Reset the peak GPU memory allocation statistics counter.

        After calling this, :meth:`gpu_peak_memory` returns the peak since
        this reset rather than the lifetime peak. A no-op on CPU.
        """
        if self.gpu_available():
            torch.cuda.reset_peak_memory_stats(self._device)
            logger.debug("Peak GPU memory stats reset for device %s.", self._device)

    def optimize_memory(self) -> None:
        """Run a GPU memory optimisation pass.

        Clears the CUDA caching allocator and resets the peak-memory
        statistics counter. A no-op on CPU-only systems.
        """
        before = self.gpu_memory_allocated()
        self.clear_gpu_cache()
        self.reset_peak_memory()
        logger.info(
            "GPU memory optimised | allocated %.1f MiB (cache cleared).", before
        )

    # ------------------------------------------------------------------
    # Model Diagnostics
    # ------------------------------------------------------------------

    def model_summary(self) -> str:
        """Return a human-readable model architecture summary.

        Returns:
            ``repr`` of the loaded model, or empty string.
        """
        return repr(self._model) if self._model is not None else ""

    def parameter_count(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            trainable_only: If ``True``, count only trainable parameters.

        Returns:
            Integer parameter count.
        """
        if self._model is None:
            return 0
        params = (
            (p for p in self._model.parameters() if p.requires_grad)
            if trainable_only
            else self._model.parameters()
        )
        return sum(p.numel() for p in params)

    def dtype_summary(self) -> dict[str, int]:
        """Return a count of parameters per dtype.

        Returns:
            Dict mapping dtype name → parameter count.
        """
        if self._model is None:
            return {}
        counts: dict[str, int] = {}
        for p in self._model.parameters():
            key = str(p.dtype)
            counts[key] = counts.get(key, 0) + p.numel()
        return counts

    def device_summary(self) -> dict[str, Any]:
        """Return the parameter placement summary.

        Returns:
            Dict mapping device string → parameter count.
        """
        if self._model is None:
            return {}
        counts: dict[str, int] = {}
        for p in self._model.parameters():
            key = str(p.device)
            counts[key] = counts.get(key, 0) + p.numel()
        return counts

    # ------------------------------------------------------------------
    # Health Monitoring
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return a structured health-status dictionary.

        The original two-value ``status`` field (``"healthy"`` /
        ``"degraded"``) is preserved exactly as before for backward
        compatibility with any caller doing
        ``health_check()["status"] == "healthy"``. A new ``level`` field
        adds the four-tier classification from spec item 10
        (``healthy`` / ``warning`` / ``degraded`` / ``critical``), derived
        from GPU headroom, model/tokenizer presence, p95 response latency,
        and the recent generation-failure ratio. A new ``checks`` field
        breaks out the individual signal values that fed into ``level``.

        Returns:
            Dict with keys ``model_loaded``, ``tokenizer_loaded``,
            ``gpu_available``, ``gpu_memory_available_mb``,
            ``uptime_seconds``, ``warmup_complete``, ``status`` (legacy,
            two-tier), ``level`` (four-tier), and ``checks`` (breakdown).
        """
        gpu_info = self.gpu_summary()
        gpu_free_mb = gpu_info.get("free_mb", 0.0)
        min_free = getattr(settings, "MIN_GPU_FREE_MB", 512)
        # A second, lower threshold used to distinguish "warning" (low but
        # not yet dangerous) from "critical" (effectively out of memory).
        critical_free = getattr(settings, "CRITICAL_GPU_FREE_MB", min_free / 4)

        model_ok = self._model is not None
        tokenizer_ok = self._tokenizer is not None
        gpu_ok = not self.gpu_available() or gpu_free_mb >= min_free
        gpu_critical = self.gpu_available() and gpu_free_mb < critical_free

        # Legacy two-tier status, unchanged.
        status = "healthy" if (model_ok and tokenizer_ok and gpu_ok) else "degraded"

        adv = self._advanced_metrics.to_dict()
        p95_latency = adv["p95_latency_s"]
        latency_warn_s = getattr(settings, "HEALTH_LATENCY_WARN_SECONDS", 10.0)
        latency_critical_s = getattr(settings, "HEALTH_LATENCY_CRITICAL_SECONDS", 30.0)

        total_seen = self._metrics.successful_requests + self._metrics.failed_requests
        failure_ratio = (
            self._metrics.failed_requests / total_seen if total_seen > 0 else 0.0
        )
        failure_warn_ratio = getattr(settings, "HEALTH_FAILURE_WARN_RATIO", 0.05)
        failure_critical_ratio = getattr(settings, "HEALTH_FAILURE_CRITICAL_RATIO", 0.25)

        # Four-tier classification: model/tokenizer absence or a critically
        # starved GPU is always CRITICAL regardless of other signals; from
        # there, the worst of (GPU headroom, latency, failure ratio)
        # determines WARNING vs DEGRADED vs HEALTHY.
        if not model_ok or not tokenizer_ok or gpu_critical:
            level = HealthStatus.CRITICAL
        elif failure_ratio >= failure_critical_ratio or p95_latency >= latency_critical_s:
            level = HealthStatus.CRITICAL
        elif not gpu_ok or failure_ratio >= failure_warn_ratio or p95_latency >= latency_warn_s:
            level = HealthStatus.DEGRADED
        elif (
            gpu_free_mb < min_free * 1.5
            or p95_latency >= latency_warn_s * 0.5
            or failure_ratio > 0.0
        ):
            level = HealthStatus.WARNING
        else:
            level = HealthStatus.HEALTHY

        return {
            "model_loaded": model_ok,
            "tokenizer_loaded": tokenizer_ok,
            "gpu_available": self.gpu_available(),
            "gpu_memory_available_mb": gpu_free_mb,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "warmup_complete": self._warmup_complete,
            "status": status,
            "level": level.value,
            "checks": {
                "gpu_headroom_ok": gpu_ok,
                "gpu_critical": gpu_critical,
                "p95_latency_s": p95_latency,
                "p99_latency_s": adv["p99_latency_s"],
                "failure_ratio": round(failure_ratio, 4),
                "active_requests": adv["active_requests"],
            },
        }

    # ------------------------------------------------------------------
    # Result Cache Management (spec #4)
    # ------------------------------------------------------------------

    def set_result_cache_backend(self, backend: ResultCacheBackend) -> None:
        """Swap the result-cache storage backend at runtime.

        This is the seam for plugging in Redis (or any other store) later:
        implement :class:`ResultCacheBackend` against your store of choice
        and pass an instance here. No other method in this class needs to
        change — :meth:`generate` only ever calls ``get``/``put`` on
        whatever backend is currently installed.

        Args:
            backend: A :class:`ResultCacheBackend` implementation.

        Example::

            class RedisResultCache(ResultCacheBackend):
                def __init__(self, redis_client): ...
                def get(self, key): ...
                def put(self, key, entry): ...
                def delete(self, key): ...
                def clear(self): ...
                def stats(self): ...

            client.set_result_cache_backend(RedisResultCache(my_redis))
        """
        self._result_cache = backend
        logger.info("Result cache backend swapped to %s", type(backend).__name__)

    def enable_result_cache(self) -> None:
        """Enable the full-result cache for subsequent :meth:`generate` calls."""
        self._result_cache_enabled = True

    def disable_result_cache(self) -> None:
        """Disable the full-result cache for subsequent :meth:`generate` calls."""
        self._result_cache_enabled = False

    def clear_result_cache(self) -> None:
        """Evict all entries from the result cache."""
        self._result_cache.clear()
        logger.info("Result cache cleared.")

    def result_cache_stats(self) -> dict[str, Any]:
        """Return result-cache statistics from the active backend."""
        return self._result_cache.stats()

    # ------------------------------------------------------------------
    # Configuration Management
    # ------------------------------------------------------------------

    def export_generation_config(self) -> dict[str, Any]:
        """Return the active generation config as a plain dictionary."""
        with self._gen_config_lock:
            return self._gen_config.to_dict() if self._gen_config else {}

    def update_generation_config(self, **kwargs: Any) -> None:
        """Merge ``kwargs`` into the active :class:`GenerationConfig`.

        Thread-safe with respect to concurrent :meth:`generate` /
        :meth:`stream_generate` calls: acquires :attr:`_gen_config_lock`
        rather than mutating :attr:`_gen_config` unguarded, which previously
        could race with the temporary config swap performed inside
        :meth:`generate`'s critical section.

        Args:
            **kwargs: Any valid :class:`~transformers.GenerationConfig` kwargs.
        """
        with self._gen_config_lock:
            if self._gen_config is None:
                raise RuntimeError("GenerationConfig is not initialised.")
            current = self._gen_config.to_dict()
            current.update(kwargs)
            self._gen_config = GenerationConfig(**current)
        logger.info("GenerationConfig updated | changes=%s", kwargs)

    def reset_generation_config(self) -> None:
        """Restore the :class:`GenerationConfig` to its initial settings.

        Thread-safe: see :meth:`update_generation_config`.
        """
        with self._gen_config_lock:
            self._init_generation_config()
        logger.info("GenerationConfig reset to defaults.")

    def save_generation_config(self, path: str) -> None:
        """Persist the current :class:`GenerationConfig` to a JSON file.

        Args:
            path: Filesystem path for the output JSON file.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.export_generation_config(), fh, indent=2)
        logger.info("GenerationConfig saved to %s", path)

    def load_generation_config(self, path: str) -> None:
        """Load a :class:`GenerationConfig` from a JSON file.

        Thread-safe: see :meth:`update_generation_config`.

        Args:
            path: Filesystem path to a previously saved config JSON.
        """
        with open(path, encoding="utf-8") as fh:
            config_dict = json.load(fh)
        with self._gen_config_lock:
            self._gen_config = GenerationConfig(**config_dict)
        logger.info("GenerationConfig loaded from %s", path)

    # ------------------------------------------------------------------
    # Diagnostics export
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Return a comprehensive diagnostic snapshot of the client's current state.

        Suitable for a ``/health/diagnostics`` FastAPI endpoint or operator
        dashboards. All values are JSON-serialisable.

        Returns:
            Dict with the following top-level keys:

            - ``model``: :class:`ModelInfo` fields (name, device, dtype,
              param count, context length, …).
            - ``gpu``: GPU memory snapshot from :meth:`gpu_summary`.
            - ``metrics``: :class:`PerformanceMetrics` fields (request counts,
              latency averages, cache hit counts, OOM count, …).
            - ``health``: Structured health status from :meth:`health_check`.
            - ``generation_config``: Active :class:`~transformers.GenerationConfig`
              as a plain dict.
            - ``prompt_cache``: :class:`_PromptCache` statistics (size, hits,
              misses, hit rate).
            - ``telemetry``: :class:`Telemetry` counters.
            - ``advanced_metrics``: :class:`AdvancedMetrics` (percentile
              latencies, CPU/RAM usage, request rate, …).
            - ``result_cache``: Active :class:`ResultCacheBackend` statistics.
        """
        return {
            "model": self._model_info.to_dict(),
            "gpu": self.gpu_summary(),
            "metrics": self._metrics.to_dict(),
            "health": self.health_check(),
            "generation_config": self.export_generation_config(),
            "prompt_cache": self._cache.stats(),
            "telemetry": self._telemetry.to_dict(),
            "advanced_metrics": self._advanced_metrics.to_dict(),
            "result_cache": self._result_cache.stats(),
        }

    def export_diagnostics_json(self, path: str) -> None:
        """Save full diagnostics to ``path`` as a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.diagnostics(), fh, indent=2, default=str)
        logger.info("Diagnostics exported to %s", path)

    def export_metrics_json(self, path: str) -> None:
        """Save performance metrics to ``path`` as a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._metrics.to_dict(), fh, indent=2, default=str)
        logger.info("Metrics exported to %s", path)

    def export_metrics_csv(self, path: str) -> None:
        """Save performance metrics to ``path`` as a CSV file.

        Args:
            path: Output file path.
        """
        metrics = self._metrics.to_dict()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        logger.info("Metrics CSV exported to %s", path)

    def export_health_json(self, path: str) -> None:
        """Save the health-check snapshot to ``path`` as a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.health_check(), fh, indent=2, default=str)
        logger.info("Health exported to %s", path)

    # ------------------------------------------------------------------
    # Diagnostics Report (spec #9)
    # ------------------------------------------------------------------

    def full_report(self) -> dict[str, Any]:
        """Return the most complete diagnostic snapshot the client can
        produce: everything in :meth:`diagnostics`, plus model architecture
        text, parameter/dtype/device breakdowns, and warmup status.

        This is the recommended single entry point for an admin dashboard
        or a ``/diagnostics/full`` FastAPI route; :meth:`diagnostics`
        remains available unchanged for callers that only want the
        original, smaller snapshot.

        Returns:
            Dict with all keys from :meth:`diagnostics` plus
            ``model_summary``, ``parameter_count``, ``trainable_parameter_count``,
            ``dtype_summary``, ``device_summary``, ``warmup_complete``, and
            ``report_generated_at``.
        """
        base = self.diagnostics()
        return {
            **base,
            "model_summary": self.model_summary(),
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(trainable_only=True),
            "dtype_summary": self.dtype_summary(),
            "device_summary": self.device_summary(),
            "warmup_complete": self.warmup_complete(),
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _minimal_yaml_dump(data: Any, indent: int = 0) -> str:
        """Fallback YAML serialiser used only when ``pyyaml`` is not
        installed. Handles the JSON-like shapes (`dict`/`list`/scalar) that
        :meth:`full_report` and :meth:`diagnostics` produce; it is not a
        general-purpose YAML emitter.

        Args:
            data: Value to serialise (dict, list, or scalar).
            indent: Current indentation level (recursion only).

        Returns:
            A YAML-formatted string.
        """
        pad = "  " * indent
        lines: list[str] = []
        if isinstance(data, dict):
            if not data:
                return f"{pad}{{}}"
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    lines.append(f"{pad}{key}:")
                    lines.append(QwenClient._minimal_yaml_dump(value, indent + 1))
                else:
                    scalar = QwenClient._minimal_yaml_dump(value, 0)
                    lines.append(f"{pad}{key}: {scalar}")
            return "\n".join(lines)
        if isinstance(data, list):
            if not data:
                return f"{pad}[]"
            for item in data:
                if isinstance(item, (dict, list)) and item:
                    lines.append(f"{pad}-")
                    lines.append(QwenClient._minimal_yaml_dump(item, indent + 1))
                else:
                    lines.append(f"{pad}- {QwenClient._minimal_yaml_dump(item, 0)}")
            return "\n".join(lines)
        if data is None:
            return "null"
        if isinstance(data, bool):
            return "true" if data else "false"
        if isinstance(data, str):
            needs_quotes = any(c in data for c in (":", "#", "\n")) or data == ""
            return f'"{data}"' if needs_quotes else data
        return str(data)

    def export_yaml(self, path: str, *, full: bool = True) -> None:
        """Save a diagnostics report to ``path`` as a YAML file.

        Uses ``pyyaml`` if it is installed; otherwise falls back to a
        minimal built-in dumper sufficient for this report's shape (plain
        dicts/lists/scalars — no anchors, tags, or multi-line block
        scalars are needed here).

        Args:
            path: Output file path.
            full: If ``True`` (default), export :meth:`full_report`;
                if ``False``, export the smaller :meth:`diagnostics`.
        """
        report = self.full_report() if full else self.diagnostics()
        if _YAML_AVAILABLE:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    json.loads(json.dumps(report, default=str)),
                    fh,
                    sort_keys=False,
                    default_flow_style=False,
                )
        else:
            logger.warning(
                "pyyaml not installed; using minimal built-in YAML dumper for %s", path
            )
            serialisable = json.loads(json.dumps(report, default=str))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._minimal_yaml_dump(serialisable))
                fh.write("\n")
        logger.info("YAML report exported to %s", path)

    def export_markdown(self, path: str, *, full: bool = True) -> None:
        """Save a diagnostics report to ``path`` as a human-readable
        Markdown file (suitable for posting in a chat channel or a status
        page, unlike the raw JSON/YAML exports).

        Args:
            path: Output file path.
            full: If ``True`` (default), export :meth:`full_report`;
                if ``False``, export the smaller :meth:`diagnostics`.
        """
        report = self.full_report() if full else self.diagnostics()
        lines: list[str] = [
            f"# QwenClient Diagnostics Report",
            "",
            f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
            "",
        ]

        def _render_section(title: str, value: Any) -> None:
            lines.append(f"## {title}")
            lines.append("")
            if isinstance(value, dict):
                if value:
                    lines.append("| Key | Value |")
                    lines.append("|---|---|")
                    for k, v in value.items():
                        if isinstance(v, (dict, list)):
                            v_str = json.dumps(v, default=str)
                        else:
                            v_str = str(v)
                        lines.append(f"| {k} | {v_str} |")
                else:
                    lines.append("_(empty)_")
            else:
                lines.append(f"```\n{value}\n```")
            lines.append("")

        for section_title, section_value in report.items():
            _render_section(section_title.replace("_", " ").title(), section_value)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("Markdown report exported to %s", path)

    # ------------------------------------------------------------------
    # Memory Management
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Unload the model and tokenizer and release all GPU memory.

        After this call the client is non-functional until :meth:`reload_model`
        is invoked. The singleton instance is preserved so the same object
        reference remains valid.

        Thread-safety: in-flight :meth:`stream_generate` calls are cancelled
        first via their registered ``cancel_event``; then :attr:`_generate_lock`
        is acquired so that any currently running synchronous
        :meth:`generate` call finishes before the model is torn down.
        """
        logger.info(
            "Cleaning up QwenClient resources | gpu_allocated=%.1f MiB …",
            self.gpu_memory_allocated(),
        )

        # Cancel every active stream so background generation threads stop
        # promptly instead of continuing to call a model that is about to
        # become None.
        active_streams = self.active_stream_ids()
        if active_streams:
            logger.info(
                "Cancelling %d active stream(s) before cleanup: %s",
                len(active_streams),
                active_streams,
            )
        for req_id in active_streams:
            self.cancel_stream(req_id)

        # Wait for any in-flight generate() call to finish before tearing
        # down the model, rather than setting self._model = None while it
        # may still be inside model.generate().
        with self._generate_lock:
            self._model = None
            self._tokenizer = None
            self._initialised = False
            self._cache.clear()
            self.clear_gpu_cache()
            gc.collect()
            logger.debug(
                "Model and tokenizer unloaded | gpu_allocated=%.1f MiB after cleanup",
                self.gpu_memory_allocated(),
            )

        # Gracefully release the Redis connection pool if the result cache
        # backend is Redis-backed (it may not be, e.g. when the in-memory
        # backend is active or Redis was never reachable at start-up).
        if isinstance(self._result_cache, RedisResultCache):
            self._result_cache.close()

        logger.info("QwenClient cleanup complete.")

    def reload_model(self) -> None:
        """Reload the tokenizer and model without destroying the singleton.

        This is the recommended recovery path after a failed load or after
        :meth:`cleanup` has been called. Acquires :attr:`_reload_lock` to
        prevent concurrent reload attempts.

        Fires :class:`HookEvent.BEFORE_RELOAD` before cleanup and
        :class:`HookEvent.AFTER_RELOAD` after the client is fully ready.

        Raises:
            TokenizerLoadError: If the tokenizer cannot be reloaded.
            ModelLoadError: If the model cannot be reloaded.
            GPUOutOfMemoryError: If a CUDA OOM occurs during model loading.
        """
        with self._reload_lock:
            self._fire(HookEvent.BEFORE_RELOAD)
            logger.info("Reloading model | model=%s …", settings.MODEL_NAME)
            self.cleanup()
            self._device = self._resolve_device()
            self._dtype = self._resolve_dtype()
            self._load_tokenizer()
            self._load_model()
            self._init_generation_config()
            if getattr(settings, "AUTO_WARMUP", True):
                self.warmup()
            self._initialised = True
            self._fire(HookEvent.AFTER_RELOAD)
            logger.info(
                "Model reloaded successfully | model=%s | device=%s | dtype=%s",
                settings.MODEL_NAME,
                self._device,
                self._dtype,
            )

    def close(self) -> None:
        """Alias for :meth:`cleanup`; called by the context manager on exit."""
        self.cleanup()

    # ------------------------------------------------------------------
    # RAG Integration Helpers
    # ------------------------------------------------------------------

    def generate_from_context(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate an answer grounded in ``context``.

        Args:
            question: User question.
            context: Retrieved document / chunk text.
            system_prompt: Optional system instruction override.
            max_new_tokens: Generation cap override.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        combined = f"Context:\n{context}\n\nQuestion:\n{question}"
        return self.generate(
            combined,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def generate_from_documents(
        self,
        question: str,
        documents: list[str],
        *,
        separator: str = "\n\n---\n\n",
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate an answer from multiple retrieved documents.

        Args:
            question: User question.
            documents: List of document text strings.
            separator: String used to join multiple documents.
            system_prompt: Optional system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        combined_context = separator.join(
            f"[Doc {i + 1}]\n{doc}" for i, doc in enumerate(documents)
        )
        return self.generate_from_context(
            question,
            combined_context,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def generate_from_chunks(
        self,
        question: str,
        chunks: list[str],
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate an answer from a list of text chunks (e.g. from Qdrant).

        Args:
            question: User question.
            chunks: List of retrieved text chunks.
            system_prompt: Optional system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        return self.generate_from_documents(
            question,
            chunks,
            separator="\n\n",
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def generate_structured(
        self,
        prompt: str,
        output_schema: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate a structured response conforming to ``output_schema``.

        Prepends a schema description to the system prompt to guide the model.

        Args:
            prompt: User prompt.
            output_schema: Description of the expected output structure.
            system_prompt: Optional additional system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        schema_instruction = (
            f"You must respond ONLY with output that conforms exactly to "
            f"the following schema:\n{output_schema}"
        )
        combined_system = (
            f"{system_prompt}\n{schema_instruction}" if system_prompt else schema_instruction
        )
        return self.generate(
            prompt,
            system_prompt=combined_system,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate a JSON-formatted response.

        Args:
            prompt: User prompt.
            system_prompt: Optional additional system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult` whose ``text`` should be valid JSON.
        """
        json_system = "You must respond ONLY with valid, minified JSON and nothing else."
        combined_system = (
            f"{system_prompt}\n{json_system}" if system_prompt else json_system
        )
        return self.generate(
            prompt,
            system_prompt=combined_system,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            metadata=metadata,
        )

    def generate_markdown(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate a Markdown-formatted response.

        Args:
            prompt: User prompt.
            system_prompt: Optional additional system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        md_system = "Format your entire response using Markdown."
        combined_system = (
            f"{system_prompt}\n{md_system}" if system_prompt else md_system
        )
        return self.generate(
            prompt,
            system_prompt=combined_system,
            max_new_tokens=max_new_tokens,
            metadata=metadata,
        )

    def generate_agent_response(
        self,
        agent_id: str,
        task: str,
        context: str,
        *,
        agent_system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Generate a response in the context of a named agent.

        Args:
            agent_id: Identifier of the calling agent (e.g.
                ``"semantic_agent"``).
            task: The agent's task description / instruction.
            context: Retrieved or accumulated context for the task.
            agent_system_prompt: Optional agent-specific system instruction.
            max_new_tokens: Generation cap.
            metadata: Caller metadata.

        Returns:
            :class:`GenerationResult`.
        """
        agent_meta = dict(metadata or {})
        agent_meta["agent_id"] = agent_id
        combined_prompt = (
            f"[Agent: {agent_id}]\nTask: {task}\n\nContext:\n{context}"
        )
        return self.generate(
            combined_prompt,
            system_prompt=agent_system_prompt,
            max_new_tokens=max_new_tokens,
            metadata=agent_meta,
        )

    # ------------------------------------------------------------------
    # Benchmark Helpers
    # ------------------------------------------------------------------

    def benchmark_prompt(
        self,
        prompt: str,
        *,
        runs: int = 3,
        warmup: int = 1,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """Benchmark generation latency and throughput for a single prompt.

        Warmup runs are discarded before statistics are computed, which
        avoids one-time CUDA kernel compilation costs skewing the results.

        Args:
            prompt: Prompt string to benchmark.
            runs: Number of timed measurement runs. More runs reduce variance.
            warmup: Number of warmup runs whose results are discarded.
            **gen_kwargs: Generation keyword arguments forwarded to
                :meth:`generate` (e.g. ``max_new_tokens``, ``temperature``).

        Returns:
            Dict with keys ``prompt`` (truncated), ``runs``,
            ``mean_latency``, ``std_latency``, ``mean_tps``, ``std_tps``,
            ``mean_generate_latency``, and ``results`` (list of raw
            :class:`GenerationResult` dicts).
        """
        results: list[GenerationResult] = []
        for i in range(warmup + runs):
            result = self.generate(prompt, **gen_kwargs)
            if i >= warmup:
                results.append(result)

        latencies = [r.latency for r in results]
        tps_values = [r.tokens_per_second for r in results]
        gen_latencies = [r.generate_latency for r in results]
        return {
            "prompt": prompt[:80] + "…" if len(prompt) > 80 else prompt,
            "runs": runs,
            "mean_latency": round(statistics.mean(latencies), 4),
            "std_latency": round(
                statistics.stdev(latencies) if len(latencies) > 1 else 0.0, 4
            ),
            "mean_tps": round(statistics.mean(tps_values), 2),
            "std_tps": round(
                statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0, 2
            ),
            "mean_generate_latency": round(statistics.mean(gen_latencies), 4),
            "results": [r.to_dict() for r in results],
        }

    def benchmark_multiple(
        self,
        prompts: list[str],
        *,
        runs: int = 1,
        **gen_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Benchmark generation across multiple prompts.

        Args:
            prompts: List of prompt strings.
            runs: Number of runs per prompt.
            **gen_kwargs: Passed through to :meth:`benchmark_prompt`.

        Returns:
            List of benchmark result dicts.
        """
        return [self.benchmark_prompt(p, runs=runs, **gen_kwargs) for p in prompts]

    def benchmark(
        self,
        *,
        runs: int = 3,
        warmup: int = 1,
    ) -> dict[str, Any]:
        """Run the standard three-category benchmark suite.

        Tests short, medium, and long prompts to characterise latency and
        throughput across different input lengths. Each category runs
        ``warmup`` discarded passes then ``runs`` timed passes.

        Args:
            runs: Timed measurement runs per category. Must be >= 1.
            warmup: Warmup runs per category whose results are discarded.

        Returns:
            Dict keyed by ``"short"``, ``"medium"``, ``"long"``, each
            containing the output of :meth:`benchmark_prompt`.
        """
        test_prompts: dict[str, str] = {
            "short": "What is 2 + 2?",
            "medium": "Explain the difference between supervised and unsupervised learning.",
            "long": (
                "Provide a comprehensive overview of transformer architectures, "
                "covering attention mechanisms, positional encoding, encoder-decoder "
                "structures, and their applications in NLP and vision."
            ),
        }
        logger.info("Running benchmark | runs=%d warmup=%d", runs, warmup)
        return {
            cat: self.benchmark_prompt(p, runs=runs, warmup=warmup)
            for cat, p in test_prompts.items()
        }

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "QwenClient":
        """Support ``with QwenClient() as client:`` usage."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Release resources on context manager exit."""
        self.close()
        return False

    # ------------------------------------------------------------------
    # Magic Methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QwenClient("
            f"model={self._model_info.model_name!r}, "
            f"device={self._model_info.device!r}, "
            f"dtype={self._model_info.dtype!r}, "
            f"initialised={self._initialised}"
            f")"
        )

    def __str__(self) -> str:
        health = self.health_check()
        return (
            f"QwenClient | model={self._model_info.model_name} | "
            f"status={health['status']} | "
            f"uptime={health['uptime_seconds']}s | "
            f"{self._metrics.summary()}"
        )


# ===========================================================================
# Module-level convenience accessor
# ===========================================================================


def get_client() -> QwenClient:
    """Return the singleton :class:`QwenClient` instance.

    Suitable for dependency injection in FastAPI routes::

        from llm.qwen_client import get_client

        @app.get("/generate")
        async def generate_endpoint(prompt: str):
            client = get_client()
            result = await client.async_generate(prompt)
            return result.to_dict()

    Returns:
        The singleton :class:`QwenClient`.
    """
    return QwenClient()


# ===========================================================================
# Entrypoint – smoke test / diagnostics (no user prompts)
# ===========================================================================

if __name__ == "__main__":
    import pprint

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("QwenClient – enterprise self-test / diagnostics")
    logger.info("=" * 60)

    client = QwenClient()

    # Repr / str
    print("\n[repr]\n", repr(client))
    print("\n[str]\n", str(client))

    # Health
    print("\n[health_check]")
    pprint.pprint(client.health_check())

    # Model info (includes version report)
    print("\n[model_info]")
    info = client._model_info.to_dict()
    # Print only non-config fields to keep output readable
    pprint.pprint({k: v for k, v in info.items() if k != "hf_config"})

    # GPU summary
    print("\n[gpu_summary]")
    pprint.pprint(client.gpu_summary())

    # Context budget report
    print("\n[context_budget_report]")
    budget = client.context_budget_report(
        system="You are a helpful assistant.",
        question="What is RAG?",
        context="RAG stands for Retrieval-Augmented Generation.",
    )
    pprint.pprint(budget.to_dict())

    # Prompt cache stats
    print("\n[prompt_cache stats]")
    pprint.pprint(client._cache.stats())

    # Warmup status
    print(f"\n[warmup_complete] {client.warmup_complete()}")

    # Full diagnostics
    print("\n[diagnostics – top-level keys]")
    diag = client.diagnostics()
    print(list(diag.keys()))

    # Benchmark
    logger.info("Running benchmark …")
    bm_results = client.benchmark(runs=2, warmup=1)
    print("\n[benchmark summary]")
    for category, bm in bm_results.items():
        print(
            f"  {category}: mean_latency={bm['mean_latency']}s "
            f"mean_tps={bm['mean_tps']} tps"
        )

    # Metrics
    print("\n[metrics]")
    pprint.pprint(client._metrics.to_dict())

    # Telemetry
    print("\n[telemetry]")
    pprint.pprint(client._telemetry.to_dict())

    # Prompt cache after benchmark
    print("\n[prompt_cache after benchmark]")
    pprint.pprint(client._cache.stats())

    # Cleanup
    logger.info("Cleaning up …")
    client.cleanup()
    logger.info("Self-test complete.")
    sys.exit(0)