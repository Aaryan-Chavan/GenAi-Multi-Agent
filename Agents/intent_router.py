"""
intent_router.py
============================================================

Universal, Fully Configuration-Driven Intent Routing Engine
============================================================

Every routing decision in this module — intent/label names, matcher
patterns, routing rules, retriever names, routing paths, fallback
label, schema boosts, plugin pipeline order, and cache behaviour — is
driven entirely by external configuration (YAML/JSON). The router
source code contains NO domain vocabulary, NO fixed intent names, NO
hardcoded retriever names (sql/vector/etc.), and NO hardcoded routing
paths. It works identically whether your configuration describes an
ecommerce catalogue, a hospital's clinical records, a legal case
database, a manufacturing IoT pipeline, or any future workflow you
haven't invented yet.

Design principles
------------------
* Configuration is mandatory input, not an optional nicety. If none is
  supplied, the router either (a) auto-generates a minimal, empty,
  always-fallback configuration (default), or (b) raises a descriptive
  ``ConfigValidationError`` if ``require_config=True`` is set — your
  choice, both are real code paths, neither is a domain-specific
  default dressed up as "generic".
* A **matcher registry** (:data:`MATCHER_REGISTRY`) supplies pluggable
  scoring strategies (regex, keyword, phrase, and any custom type you
  register) that intents reference by name in configuration.
* A **plugin registry** (:data:`PLUGIN_REGISTRY`) supplies pluggable
  routing stages. The pipeline order, per-stage confidence thresholds,
  and enable/disable state are all configuration-driven.
* Routing output is a generic ``retrievers: Dict[str, bool]`` mapping
  plus an open ``extra_flags: Dict[str, Any]`` — no assumption that
  "sql" and "vector" are the only retrievers, or that any particular
  flag name has special meaning to the router itself.
* Schema and dataset metadata are free-form; schema boosts iterate
  whatever attributes configuration references, never a fixed list.
* Thread-safe, cache-accelerated, hot-reloadable, and warmup-capable,
  exactly as before — none of the original operational guarantees are
  lost in generalising the routing logic itself.

Output fields (IntentResult)
-----------------------------
    label          : str              — resolved routing label (config-defined; may be
                                         any string your configuration declares — sales_analysis,
                                         medical_summary, fraud_detection, intent_47, etc.)
    retrievers     : Dict[str, bool]  — which configured retrievers this query should use
                                         (config-defined names; sql/vector/graph/elastic/api/... —
                                         anything you declare)
    routing_path   : str              — routing path label, taken verbatim from configuration
    confidence     : float            — [0, 1]
    method         : str              — identifier of whichever plugin/cache stage produced this
                                         result (config-defined plugin names, or "cache")
    latency_ms     : float            — wall-clock time for this call
    extra_flags    : Dict[str, Any]   — arbitrary additional flags declared per-label in configuration
    plugin_scores  : Dict[str, Any]   — diagnostic: raw score/verdict per plugin that ran

Legacy compatibility properties (``query_type``, ``needs_sql``,
``needs_vector``) are provided as computed properties over the new
generic fields for callers migrating from the previous version; they
are NOT dataclass fields and never appear in ``to_dict()`` output,
since a truly generic router cannot assume "sql"/"vector" exist.
============================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "IntentRouter",
    "IntentResult",
    "SchemaContext",
    "DatasetMetadata",
    "RoutingPlugin",
    "RuleEnginePlugin",
    "ClassifierPlugin",
    "Matcher",
    "RegexMatcher",
    "KeywordMatcher",
    "PhraseMatcher",
    "RouterConfig",
    "ConfigValidationError",
    "MATCHER_REGISTRY",
    "PLUGIN_REGISTRY",
    "register_matcher",
    "register_plugin",
    "get_router",
    "route",
    "set_log_level",
]

# ============================================================
# LOGGING
# ============================================================

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


def set_log_level(level: Union[int, str]) -> None:
    """Configure this module's log verbosity at runtime.

    Args:
        level: A ``logging`` level constant (e.g. ``logging.DEBUG``) or
            its string name (e.g. ``"DEBUG"``).
    """
    LOGGER.setLevel(level)


# ============================================================
# OPTIONAL SOFT DEPENDENCIES
# ============================================================

try:
    import yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False

try:
    import numpy as np  # type: ignore[import]  # noqa: F401
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import]
    from sklearn.pipeline import Pipeline  # type: ignore[import]

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================
# EXCEPTIONS
# ============================================================


class ConfigValidationError(ValueError):
    """Raised when a router configuration file/dict fails validation, or
    when configuration is required but absent.

    The message always names the offending section/key so operators can
    fix the configuration without reading source code.
    """


# ============================================================
# PUBLIC DATA CLASSES
# ============================================================


@dataclass
class IntentResult:
    """Routing diagnostics produced by :meth:`IntentRouter.route`.

    Fully generic: ``retrievers`` and ``extra_flags`` carry whatever
    keys your configuration defines. Nothing here assumes any specific
    retriever, routing path, or label vocabulary.
    """

    label: str = ""
    retrievers: Dict[str, bool] = field(default_factory=dict)
    routing_path: str = ""
    confidence: float = 0.0
    method: str = ""
    latency_ms: float = 0.0
    extra_flags: Dict[str, Any] = field(default_factory=dict)
    plugin_scores: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)

    # ---- Legacy compatibility (computed, not dataclass fields) ----

    @property
    def query_type(self) -> str:
        """Deprecated alias for :attr:`label`, kept for callers migrating
        from the previous ``query_type``-based API."""
        return self.label

    @property
    def needs_sql(self) -> bool:
        """Deprecated convenience accessor. Equivalent to
        ``retrievers.get("sql", False)``; only meaningful if your
        configuration declares a retriever literally named ``"sql"``."""
        return bool(self.retrievers.get("sql", False))

    @property
    def needs_vector(self) -> bool:
        """Deprecated convenience accessor. Equivalent to
        ``retrievers.get("vector", False)``; only meaningful if your
        configuration declares a retriever literally named ``"vector"``."""
        return bool(self.retrievers.get("vector", False))


@dataclass
class SchemaContext:
    """Optional dataset schema metadata used to bias routing decisions.

    Nothing is required, and no attribute name has special meaning to
    the router itself — :attr:`RuleEnginePlugin` iterates
    ``settings.schema_boosts`` from configuration and looks up whatever
    attribute names appear there against :meth:`to_dict`. The named
    fields below exist purely as convenient, commonly-useful shortcuts;
    :attr:`extra` accepts any attribute name at all (e.g.
    ``image_columns``, ``ontology_nodes``, ``relationship_graph``),
    with no code change required to support a new one.

    Attributes:
        tables: Known table/collection names.
        columns: All known column/field names across the dataset.
        column_types: Mapping of column name to a type label.
        numeric_columns: Columns known to hold numeric data.
        text_columns: Columns known to hold free text.
        time_columns: Columns known to hold temporal data.
        primary_keys: Known primary key column names.
        foreign_keys: Known foreign key column names.
        relationships: Arbitrary relationship descriptors.
        extra: Arbitrary additional schema attributes not covered by
            the named fields above (e.g. ``{"image_columns": [...]}``).
    """

    tables: Sequence[str] = field(default_factory=tuple)
    columns: Sequence[str] = field(default_factory=tuple)
    column_types: Dict[str, str] = field(default_factory=dict)
    numeric_columns: Sequence[str] = field(default_factory=tuple)
    text_columns: Sequence[str] = field(default_factory=tuple)
    time_columns: Sequence[str] = field(default_factory=tuple)
    primary_keys: Sequence[str] = field(default_factory=tuple)
    foreign_keys: Sequence[str] = field(default_factory=tuple)
    relationships: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a merged, JSON-serialisable dictionary of every
        attribute (named fields plus :attr:`extra`), used by
        schema-boost lookups. ``extra`` keys take precedence on
        collision so callers can override a named field if needed."""
        base = {
            "tables": self.tables,
            "columns": self.columns,
            "column_types": self.column_types,
            "numeric_columns": self.numeric_columns,
            "text_columns": self.text_columns,
            "time_columns": self.time_columns,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
            "relationships": self.relationships,
        }
        base.update(self.extra)
        return base


@dataclass
class DatasetMetadata:
    """Optional dataset/domain metadata used to adapt routing.

    Attributes:
        domain: Free-text domain label, purely informational.
        dataset_type: Free-text dataset shape label, purely informational.
        available_retrievers: Names of retrieval backends actually wired
            up for this deployment (e.g. ``{"sql", "vector", "graph"}``
            — any names matching your configuration's retriever keys).
            When supplied, the router will disable any ``retrievers``
            flag in the result that isn't in this set.
        custom: Arbitrary additional metadata for custom plugins.
    """

    domain: str = ""
    dataset_type: str = ""
    available_retrievers: Sequence[str] = field(default_factory=tuple)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


@dataclass
class _RoutingContext:
    """Internal bundle passed to plugins during a single route() call."""

    normalized_text: str
    raw_text: str
    schema: Optional[SchemaContext]
    dataset_metadata: Optional[DatasetMetadata]
    config: "RouterConfig"


@dataclass
class _PluginVerdict:
    """A single plugin's routing opinion."""

    label: str
    confidence: float
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# MATCHER REGISTRY (requirement 9, 18)
# ============================================================


class Matcher(ABC):
    """Abstract base class for a single scoring strategy used inside an
    intent's matcher list. Register new matcher types via
    :func:`register_matcher`; unregistered matcher types referenced in
    configuration are skipped with a warning, never a crash.
    """

    @abstractmethod
    def score(self, normalized_text: str) -> float:
        """Return a non-negative score for how strongly ``normalized_text``
        matches this matcher's criteria. A score of ``0.0`` means no match.

        Args:
            normalized_text: The already-normalized query text.

        Returns:
            A non-negative float score contribution.
        """


class RegexMatcher(Matcher):
    """Matches via one or more compiled regular expressions, each with
    its own weight. Weights of all matching patterns are summed."""

    def __init__(self, patterns: List[Tuple[str, float]]) -> None:
        """Args:
            patterns: List of ``[regex_string, weight]`` pairs.

        Raises:
            ConfigValidationError: If any pattern fails to compile.
        """
        self._compiled: List[Tuple[re.Pattern, float]] = []
        for entry in patterns:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], (int, float))
            ):
                raise ConfigValidationError(
                    f"RegexMatcher received an invalid pattern entry {entry!r}; "
                    "expected [regex_string, numeric_weight]."
                )
            pattern_str, weight = entry
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
            except re.error as exc:
                raise ConfigValidationError(
                    f"RegexMatcher has an invalid regex {pattern_str!r}: {exc}"
                ) from exc
            self._compiled.append((compiled, float(weight)))

    def score(self, normalized_text: str) -> float:
        total = 0.0
        for pattern, weight in self._compiled:
            if pattern.search(normalized_text):
                total += weight
        return total


class KeywordMatcher(Matcher):
    """Matches via exact whole-word membership (fast, no regex overhead)."""

    def __init__(self, keywords: List[Tuple[str, float]]) -> None:
        """Args:
            keywords: List of ``[keyword, weight]`` pairs. Matching is
                case-insensitive and requires the keyword to appear as
                a standalone word in the normalized text.
        """
        self._items: List[Tuple[str, float]] = []
        for entry in keywords:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], (int, float))
            ):
                raise ConfigValidationError(
                    f"KeywordMatcher received an invalid entry {entry!r}; "
                    "expected [keyword_string, numeric_weight]."
                )
            self._items.append((entry[0].lower(), float(entry[1])))

    def score(self, normalized_text: str) -> float:
        words = set(normalized_text.split())
        total = 0.0
        for keyword, weight in self._items:
            if keyword in words:
                total += weight
        return total


class PhraseMatcher(Matcher):
    """Matches via substring containment of multi-word phrases."""

    def __init__(self, phrases: List[Tuple[str, float]]) -> None:
        """Args:
            phrases: List of ``[phrase, weight]`` pairs. Matching is
                case-insensitive substring containment against the
                normalized text.
        """
        self._items: List[Tuple[str, float]] = []
        for entry in phrases:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], (int, float))
            ):
                raise ConfigValidationError(
                    f"PhraseMatcher received an invalid entry {entry!r}; "
                    "expected [phrase_string, numeric_weight]."
                )
            self._items.append((entry[0].lower(), float(entry[1])))

    def score(self, normalized_text: str) -> float:
        total = 0.0
        for phrase, weight in self._items:
            if phrase in normalized_text:
                total += weight
        return total


def _build_regex_matcher(cfg: Dict[str, Any]) -> Matcher:
    return RegexMatcher(cfg.get("patterns", []))


def _build_keyword_matcher(cfg: Dict[str, Any]) -> Matcher:
    return KeywordMatcher(cfg.get("keywords", []))


def _build_phrase_matcher(cfg: Dict[str, Any]) -> Matcher:
    return PhraseMatcher(cfg.get("phrases", []))


MATCHER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Matcher]] = {}


def register_matcher(type_name: str, factory: Callable[[Dict[str, Any]], Matcher]) -> None:
    """Register a new matcher type usable in configuration.

    Args:
        type_name: The string identifier used as ``matchers[].type`` in
            configuration (e.g. ``"embedding"``, ``"ontology"``, ``"graph"``).
        factory: A callable taking the matcher's raw config dict (the
            full ``{"type": ..., ...}`` mapping from configuration) and
            returning a :class:`Matcher` instance.

    Example:
        >>> class MyLLMMatcher(Matcher):
        ...     def __init__(self, prompt: str): self._prompt = prompt
        ...     def score(self, normalized_text: str) -> float: ...
        >>> register_matcher("llm", lambda cfg: MyLLMMatcher(cfg["prompt"]))
    """
    MATCHER_REGISTRY[type_name] = factory


register_matcher("regex", _build_regex_matcher)
register_matcher("keyword", _build_keyword_matcher)
register_matcher("phrase", _build_phrase_matcher)


# ============================================================
# TEXT NORMALIZER
# ============================================================


class TextNormalizer:
    """Configurable, Unicode-aware text normalizer.

    Handles accent stripping, case folding (multi-language safe via
    ``str.casefold()``), punctuation normalization, and whitespace
    collapsing — each independently toggleable via configuration.
    """

    def __init__(self, options: Dict[str, Any]) -> None:
        self._strip_accents: bool = bool(options.get("strip_accents", True))
        self._casefold: bool = bool(options.get("casefold", True))
        self._remove_punctuation: bool = bool(options.get("remove_punctuation", True))
        self._collapse_whitespace: bool = bool(options.get("collapse_whitespace", True))
        self._punct_re = re.compile(r"[^\w\s]", re.UNICODE)
        self._ws_re = re.compile(r"\s+", re.UNICODE)

    def normalize(self, text: str) -> str:
        """Normalize ``text`` according to the configured options.

        Args:
            text: Raw input text, any language/script.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        result = text

        if self._strip_accents:
            result = unicodedata.normalize("NFKD", result)
            result = "".join(ch for ch in result if not unicodedata.combining(ch))

        result = result.casefold() if self._casefold else result.lower()

        if self._remove_punctuation:
            result = self._punct_re.sub(" ", result)

        if self._collapse_whitespace:
            result = self._ws_re.sub(" ", result).strip()

        return result


# ============================================================
# CONTEXT-AWARE LRU CACHE (requirement 12)
# ============================================================


class _LRUCache:
    """Thread-safe O(1) LRU cache keyed by a composite hash of the
    normalized query PLUS routing context (config version, schema, and
    dataset metadata), so a configuration reload or a per-call schema
    change can never return a stale cached result."""

    __slots__ = ("_cap", "_cache", "_lock")

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._cache: Dict[str, IntentResult] = {}
        self._lock = threading.Lock()

    @staticmethod
    def build_key(
        normalized_text: str,
        config_version: str,
        schema: Optional[SchemaContext],
        dataset_metadata: Optional[DatasetMetadata],
    ) -> str:
        """Build a composite cache key covering every input that can
        affect a routing decision.

        Args:
            normalized_text: The normalized query text.
            config_version: The active :attr:`RouterConfig.config_version`.
            schema: The effective :class:`SchemaContext` for this call, if any.
            dataset_metadata: The effective :class:`DatasetMetadata` for
                this call, if any.

        Returns:
            A 32-character MD5 hex digest uniquely identifying this
            combination of inputs.
        """
        schema_repr = json.dumps(schema.to_dict(), sort_keys=True, default=str) if schema else ""
        metadata_repr = (
            json.dumps(dataset_metadata.to_dict(), sort_keys=True, default=str)
            if dataset_metadata
            else ""
        )
        composite = f"{normalized_text}|{config_version}|{schema_repr}|{metadata_repr}"
        return hashlib.md5(composite.encode("utf-8", errors="ignore")).hexdigest()

    def get(self, key: str) -> Optional[IntentResult]:
        """Retrieve a cached result by composite key, promoting it to
        most-recently-used on hit."""
        with self._lock:
            if key in self._cache:
                self._cache[key] = self._cache.pop(key)
                return self._cache[key]
        return None

    def put(self, key: str, result: IntentResult) -> None:
        """Store a result under a composite key, evicting the
        least-recently-used entry if the cache is full."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._cache.clear()


# ============================================================
# CONFIGURATION MODEL
# ============================================================


@dataclass
class RouterConfig:
    """Fully parsed, validated, and compiled router configuration.

    Produced by :meth:`RouterConfig.load` or :meth:`RouterConfig.from_dict`.
    Never constructed directly by application code.

    Nothing here assumes any particular label name, retriever name, or
    routing path string. ``settings.fallback_label`` names whichever
    label configuration designates as the universal fallback, and every
    validation error message references that configured name rather
    than any hardcoded string.
    """

    intents: Dict[str, List[Matcher]]
    routing: Dict[str, Dict[str, Any]]
    settings: Dict[str, Any]
    training_examples: Dict[str, List[str]]
    pipeline: List[Dict[str, Any]]
    config_version: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RouterConfig":
        """Validate and compile a raw configuration dictionary.

        Args:
            raw: Parsed YAML/JSON configuration content.

        Returns:
            A fully compiled :class:`RouterConfig`.

        Raises:
            ConfigValidationError: If the configuration is structurally
                invalid, references undefined labels, contains invalid
                matcher definitions, or has out-of-range settings.
        """
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Top-level configuration must be a mapping, got {type(raw).__name__}."
            )

        settings = dict(raw.get("settings") or {})
        settings.setdefault("dominant_threshold", 1.2)
        settings.setdefault("gap_threshold", 0.4)
        settings.setdefault("low_confidence_threshold", 0.35)
        settings.setdefault("min_score", 0.5)
        settings.setdefault("cache_size", 4096)
        settings.setdefault("fallback_label", settings.get("fallback_intent", "generic"))
        settings.setdefault("unavailable_routing_path", "none")
        settings.setdefault("warmup_queries", [])
        settings.setdefault("demo_queries", [])
        settings.setdefault(
            "normalizer",
            {
                "strip_accents": True,
                "casefold": True,
                "remove_punctuation": True,
                "collapse_whitespace": True,
            },
        )
        settings.setdefault("schema_boosts", {})
        settings.setdefault("pipeline", [])

        fallback_label = str(settings["fallback_label"])

        for key in ("dominant_threshold", "gap_threshold", "low_confidence_threshold"):
            value = settings.get(key)
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigValidationError(
                    f"settings.{key} must be a non-negative number, got {value!r}."
                )

        cache_size = settings.get("cache_size")
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ConfigValidationError(
                f"settings.cache_size must be a positive integer, got {cache_size!r}."
            )

        raw_intents = raw.get("intents") or {}
        if not isinstance(raw_intents, dict):
            raise ConfigValidationError(
                f"'intents' must be a mapping, got {type(raw_intents).__name__}."
            )

        compiled_intents: Dict[str, List[Matcher]] = {}
        for label, intent_def in raw_intents.items():
            if not isinstance(intent_def, dict):
                raise ConfigValidationError(
                    f"intents.{label} must be a mapping, got {type(intent_def).__name__}."
                )

            matcher_defs = intent_def.get("matchers")
            if matcher_defs is None and "patterns" in intent_def:
                # Backward-compatible shim: a bare "patterns" list is
                # treated as a single implicit regex matcher.
                matcher_defs = [{"type": "regex", "patterns": intent_def["patterns"]}]

            if not isinstance(matcher_defs, list) or not matcher_defs:
                raise ConfigValidationError(
                    f"intents.{label} must define a non-empty 'matchers' list "
                    "(each entry a mapping with a 'type' key), or a legacy "
                    "'patterns' list for backward compatibility."
                )

            compiled_matchers: List[Matcher] = []
            for matcher_def in matcher_defs:
                if not isinstance(matcher_def, dict) or "type" not in matcher_def:
                    raise ConfigValidationError(
                        f"intents.{label} has an invalid matcher entry "
                        f"{matcher_def!r}; expected a mapping with a 'type' key."
                    )
                matcher_type = matcher_def["type"]
                factory = MATCHER_REGISTRY.get(matcher_type)
                if factory is None:
                    LOGGER.warning(
                        "intents.%s references unknown matcher type '%s'; "
                        "ignoring it (no matcher registered for this type — "
                        "call register_matcher() to add support).",
                        label,
                        matcher_type,
                    )
                    continue
                compiled_matchers.append(factory(matcher_def))

            if not compiled_matchers:
                raise ConfigValidationError(
                    f"intents.{label} has no usable matchers after filtering "
                    "unknown/unregistered matcher types."
                )

            compiled_intents[label] = compiled_matchers

        raw_routing = raw.get("routing") or {}
        if not isinstance(raw_routing, dict):
            raise ConfigValidationError(
                f"'routing' must be a mapping, got {type(raw_routing).__name__}."
            )

        if fallback_label not in raw_routing:
            raise ConfigValidationError(
                f"routing must define a '{fallback_label}' entry, matching "
                f"settings.fallback_label='{fallback_label}'. This is the "
                "label used whenever no matcher/plugin produces a confident "
                "result."
            )

        normalized_routing: Dict[str, Dict[str, Any]] = {}
        for label, route_def in raw_routing.items():
            if not isinstance(route_def, dict):
                raise ConfigValidationError(
                    f"routing.{label} must be a mapping, got {type(route_def).__name__}."
                )
            retrievers = route_def.get("retrievers", {})
            if not isinstance(retrievers, dict):
                raise ConfigValidationError(
                    f"routing.{label}.retrievers must be a mapping of "
                    f"retriever_name -> bool, got {type(retrievers).__name__}."
                )
            for retriever_name, flag in retrievers.items():
                if not isinstance(flag, bool):
                    raise ConfigValidationError(
                        f"routing.{label}.retrievers.{retriever_name} must be "
                        f"a boolean, got {flag!r}."
                    )
            extra_flags = route_def.get("extra_flags", {})
            if not isinstance(extra_flags, dict):
                raise ConfigValidationError(
                    f"routing.{label}.extra_flags must be a mapping, got "
                    f"{type(extra_flags).__name__}."
                )
            normalized_routing[label] = {
                "retrievers": {k: bool(v) for k, v in retrievers.items()},
                "routing_path": str(route_def.get("routing_path", label)),
                "extra_flags": dict(extra_flags),
            }

        for label in compiled_intents:
            if label not in normalized_routing:
                raise ConfigValidationError(
                    f"intents.{label} has no matching entry under 'routing'. "
                    "Every declared intent/label must have a routing rule."
                )

        raw_training = raw.get("training_examples") or {}
        if not isinstance(raw_training, dict):
            raise ConfigValidationError(
                "training_examples must be a mapping of label -> list[str]."
            )
        training_examples: Dict[str, List[str]] = {}
        for label, examples in raw_training.items():
            if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
                raise ConfigValidationError(
                    f"training_examples.{label} must be a list of strings."
                )
            if examples:
                training_examples[label] = examples

        raw_pipeline = settings.get("pipeline") or []
        if not isinstance(raw_pipeline, list):
            raise ConfigValidationError(
                f"settings.pipeline must be a list, got {type(raw_pipeline).__name__}."
            )
        pipeline: List[Dict[str, Any]] = []
        for stage_idx, stage in enumerate(raw_pipeline):
            if not isinstance(stage, dict) or "plugin" not in stage:
                raise ConfigValidationError(
                    f"settings.pipeline[{stage_idx}] must be a mapping with "
                    f"a 'plugin' key, got {stage!r}."
                )
            pipeline.append(dict(stage))

        if not pipeline:
            # Structural default (registry stage names, not domain content):
            # rule engine first, generic ML classifier as fallback. Still
            # fully overridable via settings.pipeline in configuration.
            pipeline = [
                {"plugin": "rule_engine", "enabled": True},
                {"plugin": "classifier", "enabled": True},
            ]

        for warmup_query in settings.get("warmup_queries", []):
            if not isinstance(warmup_query, str):
                raise ConfigValidationError(
                    f"settings.warmup_queries entries must be strings, got {warmup_query!r}."
                )
        for demo_query in settings.get("demo_queries", []):
            if not isinstance(demo_query, str):
                raise ConfigValidationError(
                    f"settings.demo_queries entries must be strings, got {demo_query!r}."
                )

        config_version = hashlib.md5(
            json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

        LOGGER.info(
            "RouterConfig compiled | version=%s | intents=%d | routing_entries=%d | "
            "pipeline_stages=%d | training_examples_labels=%d | fallback_label=%s",
            config_version,
            len(compiled_intents),
            len(normalized_routing),
            len(pipeline),
            len(training_examples),
            fallback_label,
        )

        return cls(
            intents=compiled_intents,
            routing=normalized_routing,
            settings=settings,
            training_examples=training_examples,
            pipeline=pipeline,
            config_version=config_version,
        )

    @classmethod
    def minimal(cls) -> "RouterConfig":
        """Build a minimal, domain-empty configuration: zero declared
        intents, a single fallback label ("generic") with no retrievers
        enabled. Used automatically when no configuration is supplied
        and ``require_config=False`` (the default) — see
        :meth:`RouterConfig.load`.

        This is intentionally NOT a "default configuration" in the old
        sense: it contains no matcher patterns, no domain vocabulary,
        and every query routed against it will simply resolve to the
        fallback label with zero confidence, until real configuration
        is supplied.

        Returns:
            A valid, minimal :class:`RouterConfig`.
        """
        return cls.from_dict(
            {
                "settings": {"fallback_label": "generic"},
                "intents": {},
                "routing": {"generic": {"retrievers": {}, "routing_path": "none"}},
                "training_examples": {},
            }
        )

    @classmethod
    def load(
        cls,
        path: Optional[Union[str, Path]],
        require_config: bool = False,
    ) -> "RouterConfig":
        """Load configuration from a YAML or JSON file.

        Args:
            path: Filesystem path to a ``.yaml``/``.yml``/``.json``
                configuration file, or ``None``.
            require_config: If ``True`` and ``path`` is ``None`` or does
                not exist, raise :class:`ConfigValidationError` instead
                of falling back to a minimal auto-generated configuration
                (requirement 1, Option B). Default ``False`` uses Option
                A: an empty, always-fallback configuration.

        Returns:
            A fully compiled :class:`RouterConfig`.

        Raises:
            ConfigValidationError: If the file exists but fails
                validation, uses a YAML extension while ``pyyaml`` is
                not installed, or if ``require_config=True`` and no
                usable configuration was found.
        """
        if path is None:
            if require_config:
                raise ConfigValidationError(
                    "No configuration was supplied (config_path/config are "
                    "both None) and require_config=True. Provide a YAML/JSON "
                    "configuration file, a config dict, or set "
                    "require_config=False to use a minimal auto-generated "
                    "fallback-only configuration instead."
                )
            LOGGER.info(
                "No config_path supplied; using a minimal, domain-empty "
                "auto-generated configuration (fallback-only routing)."
            )
            return cls.minimal()

        file_path = Path(path)
        if not file_path.exists():
            if require_config:
                raise ConfigValidationError(
                    f"Configuration file {file_path} does not exist and "
                    "require_config=True. Provide a valid configuration file "
                    "or set require_config=False."
                )
            LOGGER.warning(
                "Config file %s not found; using a minimal, domain-empty "
                "auto-generated configuration (fallback-only routing).",
                file_path,
            )
            return cls.minimal()

        suffix = file_path.suffix.lower()
        text = file_path.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise ConfigValidationError(
                    f"Config file {file_path} is YAML but the 'pyyaml' "
                    "package is not installed. Install it with: pip install pyyaml"
                )
            raw = yaml.safe_load(text)
        elif suffix == ".json":
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigValidationError(
                    f"Config file {file_path} is not valid JSON: {exc}"
                ) from exc
        else:
            raise ConfigValidationError(
                f"Unsupported config file extension '{suffix}' for {file_path}. "
                "Use .yaml, .yml, or .json."
            )

        LOGGER.info("Loaded router configuration from %s", file_path)
        return cls.from_dict(raw)


# ============================================================
# PLUGIN ARCHITECTURE (requirements 10, 11, 17)
# ============================================================


class RoutingPlugin(ABC):
    """Abstract base class for a routing pipeline stage.

    Implement this to add custom rule engines, LLM-based routers,
    zero-shot classifiers, embedding classifiers, or knowledge-graph
    routers to the pipeline. Register your subclass with
    :func:`register_plugin` so configuration can reference it by name
    without any code change to :class:`IntentRouter`.
    """

    #: Overridable default identifier for this plugin, surfaced in
    #: IntentResult.method. Configuration may override this per-stage
    #: via the pipeline entry's "name" key.
    default_name: str = "plugin"

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name or self.default_name

    @property
    def name(self) -> str:
        """Identifier used in :attr:`IntentResult.method`. Configurable
        per pipeline stage via ``{"plugin": ..., "name": "custom_name"}``."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @abstractmethod
    def evaluate(self, context: _RoutingContext) -> Optional[_PluginVerdict]:
        """Evaluate the query and optionally return a routing verdict.

        Args:
            context: The current routing context (normalized text, raw
                text, optional schema/metadata, and the active config).

        Returns:
            A :class:`_PluginVerdict`, or ``None`` if this plugin has no
            opinion (e.g. insufficient training data, feature unavailable).
        """


class RuleEnginePlugin(RoutingPlugin):
    """Matcher-driven rule engine, entirely configuration-controlled.

    For each label declared in ``config.intents``, sums the score of
    every configured :class:`Matcher` against the normalized query text,
    then optionally applies schema boosts (``settings.schema_boosts``)
    using whatever schema attribute names configuration references —
    no fixed attribute list.

    The winning label is resolved purely by relative score margins
    (``dominant_threshold``/``gap_threshold`` from settings); ties or
    close contests resolve to ``settings.fallback_label`` rather than
    any hardcoded label name.
    """

    default_name = "rule_engine"

    def evaluate(self, context: _RoutingContext) -> Optional[_PluginVerdict]:
        config = context.config
        if not config.intents:
            # No intents declared at all (e.g. minimal auto-generated
            # config) — this plugin has nothing to contribute.
            return None

        scores: Dict[str, float] = {label: 0.0 for label in config.intents}
        LOGGER.debug("RuleEngine initial scores: %s", scores)

        for label, matchers in config.intents.items():
            for matcher in matchers:
                scores[label] += matcher.score(context.normalized_text)

        LOGGER.debug("RuleEngine post-matcher scores: %s", scores)


        if context.schema is not None:
            schema_boosts: Dict[str, Dict[str, float]] = config.settings.get(
                "schema_boosts", {}
            )
            schema_dict = context.schema.to_dict()
            for label, boosts in schema_boosts.items():
                if label not in scores:
                    continue
                for attr_name, boost_weight in boosts.items():
                    if schema_dict.get(attr_name):
                        scores[label] += float(boost_weight)

        fallback_label = str(config.settings["fallback_label"])
        total = sum(scores.values())
        if total == 0.0:
            return _PluginVerdict(
                label=fallback_label, confidence=0.0, method=self.name,
                diagnostics={"scores": scores},
            )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # NEW SAFE CONFIDENCE (prevents inflated confidence on noise)
        confidence = top_score / (top_score + second_score + 1e-9)

        dominant_threshold = config.settings["dominant_threshold"]
        gap_threshold = config.settings["gap_threshold"]

        # HARD REJECTION ZONE (IMPORTANT FIX)
        min_score = config.settings.get("min_score", 0.5)

        if top_score < min_score:
            return _PluginVerdict(
                label=fallback_label,
                confidence=0.0,
                method=self.name,
                diagnostics={"reason": "below_min_score", "scores": scores},
            )
        if top_score >= dominant_threshold and (top_score - second_score) >= gap_threshold:
            final_label, final_conf = top_label, min(confidence, 0.98)
        elif second_score >= 0.6 * dominant_threshold and (top_score - second_score) < gap_threshold:
            final_label, final_conf = fallback_label, confidence * 0.85
        else:
            final_label, final_conf = top_label, min(confidence, 0.90)
        # SAFETY CAP: prevent fallback from looking "confident"
        if final_label == fallback_label:
            final_conf = min(final_conf, 0.35)

        return _PluginVerdict(
            label=final_label,
            confidence=final_conf,
            method=self.name,
            diagnostics={"scores": scores},
        )


class _ClassifierBackend(ABC):
    """Internal interface shared by the embedding and TF-IDF backends."""

    @abstractmethod
    def is_trained(self) -> bool:
        ...

    @abstractmethod
    def train(self, examples: Dict[str, List[str]]) -> None:
        ...

    @abstractmethod
    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        ...


class _EmbeddingClassifierBackend(_ClassifierBackend):
    """SentenceTransformers embeddings + LogisticRegression classifier."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._encoder: Optional[Any] = None
        self._clf: Optional[Any] = None
        self._lock = threading.Lock()

    def is_trained(self) -> bool:
        return self._clf is not None

    def _ensure_encoder(self) -> None:
        if self._encoder is None:
            LOGGER.info("Loading SentenceTransformer model '%s' …", self._model_name)
            self._encoder = SentenceTransformer(self._model_name)

    def train(self, examples: Dict[str, List[str]]) -> None:
        if not examples:
            return
        with self._lock:
            self._ensure_encoder()
            texts: List[str] = []
            labels: List[str] = []
            for label, samples in examples.items():
                texts.extend(samples)
                labels.extend([label] * len(samples))
            if len(set(labels)) < 2:
                LOGGER.warning(
                    "Embedding classifier needs training_examples for at "
                    "least 2 distinct labels; skipping training."
                )
                return
            embeddings = self._encoder.encode(texts, show_progress_bar=False)
            clf = LogisticRegression(max_iter=500)
            clf.fit(embeddings, labels)
            self._clf = clf
            LOGGER.info(
                "Embedding classifier trained | labels=%d | examples=%d",
                len(set(labels)), len(texts),
            )

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        if self._clf is None or self._encoder is None:
            return None
        embedding = self._encoder.encode([text], show_progress_bar=False)
        proba = self._clf.predict_proba(embedding)[0]
        classes = self._clf.classes_
        best_idx = int(proba.argmax())
        return str(classes[best_idx]), float(proba[best_idx])


class _TfidfClassifierBackend(_ClassifierBackend):
    """TF-IDF (char n-grams) + LogisticRegression classifier."""

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._lock = threading.Lock()

    def is_trained(self) -> bool:
        return self._pipeline is not None

    def train(self, examples: Dict[str, List[str]]) -> None:
        if not examples:
            return
        with self._lock:
            texts: List[str] = []
            labels: List[str] = []
            for label, samples in examples.items():
                texts.extend(samples)
                labels.extend([label] * len(samples))
            if len(set(labels)) < 2:
                LOGGER.warning(
                    "TF-IDF classifier needs training_examples for at "
                    "least 2 distinct labels; skipping training."
                )
                return
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 4),
                    max_features=2000, sublinear_tf=True,
                )),
                ("clf", LogisticRegression(C=4.0, max_iter=300, solver="lbfgs")),
            ])
            pipeline.fit(texts, labels)
            self._pipeline = pipeline
            LOGGER.info(
                "TF-IDF classifier trained | labels=%d | examples=%d",
                len(set(labels)), len(texts),
            )

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        if self._pipeline is None:
            return None
        proba = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        best_idx = int(proba.argmax())
        return str(classes[best_idx]), float(proba[best_idx])


class ClassifierPlugin(RoutingPlugin):
    """Generic ML fallback plugin.

    Predicts an arbitrary routing **label** — not necessarily an
    "intent" in any narrow sense; whatever keys appear in
    ``training_examples`` are valid outputs, be they intents, workflow
    names, retrieval strategies, or anything else your configuration
    defines.

    Automatically selects SentenceTransformers + LogisticRegression when
    available, otherwise falls back to TF-IDF + LogisticRegression, and
    becomes a silent no-op (always returns ``None``) if neither
    ``scikit-learn`` is installed nor training examples are supplied.
    """

    default_name = "classifier"

    def __init__(
        self,
        training_examples: Dict[str, List[str]],
        prefer_embeddings: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._backend: Optional[_ClassifierBackend] = None
        self._training_examples = dict(training_examples)

        if not training_examples:
            LOGGER.info(
                "ClassifierPlugin: no training_examples supplied in "
                "configuration; ML fallback stage will be a no-op."
            )
            return

        if not _SKLEARN_AVAILABLE:
            LOGGER.warning(
                "ClassifierPlugin: scikit-learn not installed; ML fallback "
                "stage will be a no-op. Install with: pip install scikit-learn"
            )
            return

        if prefer_embeddings and _SENTENCE_TRANSFORMERS_AVAILABLE:
            LOGGER.info("ClassifierPlugin: using SentenceTransformer embeddings backend.")
            self._backend = _EmbeddingClassifierBackend()
        else:
            if prefer_embeddings and not _SENTENCE_TRANSFORMERS_AVAILABLE:
                LOGGER.info(
                    "ClassifierPlugin: sentence-transformers not installed; "
                    "falling back to TF-IDF + LogisticRegression backend."
                )
            self._backend = _TfidfClassifierBackend()

    def ensure_trained(self) -> None:
        """Train the backend (idempotent) if it has training data."""
        if self._backend is not None and not self._backend.is_trained():
            self._backend.train(self._training_examples)

    def retrain(self, additional_examples: Dict[str, List[str]]) -> None:
        """Retrain the backend with additional labelled examples.

        Args:
            additional_examples: Mapping of label -> list of example
                query strings to merge with the existing training set.
        """
        if self._backend is None:
            LOGGER.warning(
                "ClassifierPlugin.retrain() called but no backend is active "
                "(scikit-learn missing or no initial training data)."
            )
            return
        for label, samples in additional_examples.items():
            self._training_examples.setdefault(label, [])
            self._training_examples[label].extend(samples)
        self._backend.train(self._training_examples)

    def evaluate(self, context: _RoutingContext) -> Optional[_PluginVerdict]:
        if self._backend is None:
            return None
        self.ensure_trained()
        if not self._backend.is_trained():
            return None
        prediction = self._backend.predict(context.normalized_text)
        if prediction is None:
            return None
        label, confidence = prediction

        # normalize classifier confidence to avoid overconfidence spikes
        confidence = min(max(confidence, 0.0), 0.95)

        return _PluginVerdict(
            label=label,
            confidence=confidence,
            method=self.name
        )


PLUGIN_REGISTRY: Dict[str, Callable[..., RoutingPlugin]] = {
    "rule_engine": lambda **kwargs: RuleEnginePlugin(**kwargs),
}


def register_plugin(name: str, factory: Callable[..., RoutingPlugin]) -> None:
    """Register a new routing plugin type usable in ``settings.pipeline``.

    Args:
        name: The string identifier used as ``pipeline[].plugin`` in
            configuration (e.g. ``"llm_router"``, ``"graph_router"``).
        factory: A callable accepting arbitrary keyword arguments (taken
            from the pipeline stage's ``params`` mapping) and returning
            a :class:`RoutingPlugin` instance.

    Example:
        >>> class MyGraphPlugin(RoutingPlugin):
        ...     default_name = "graph_router"
        ...     def evaluate(self, context): ...
        >>> register_plugin("graph_router", lambda **kw: MyGraphPlugin(**kw))
        # config: settings.pipeline: [{"plugin": "graph_router", "params": {...}}]
    """
    PLUGIN_REGISTRY[name] = factory


# ============================================================
# CORE ROUTER
# ============================================================


@dataclass
class _PipelineStage:
    """A compiled, ready-to-execute pipeline stage."""

    plugin: RoutingPlugin
    stop_on_confidence: float


class IntentRouter:
    """Universal, fully configuration-driven routing engine.

    Routing pipeline order, per-stage confidence thresholds, and
    enable/disable state all come from ``settings.pipeline`` in
    configuration (or the structural two-stage default of
    rule_engine → classifier if configuration omits it — a registry
    reference, not domain content). Additional plugin instances can
    still be injected directly via the ``plugins`` constructor
    argument, appended after the configured pipeline.

    Backward compatible with the original constructor signature
    (``cache_size``, ``classifier_confidence_threshold``, ``use_cache``);
    all new capabilities are additive, optional keyword arguments.
    """

    def __init__(
        self,
        cache_size: int = 4096,
        classifier_confidence_threshold: float = 0.35,
        use_cache: bool = True,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        require_config: bool = False,
        schema: Optional[SchemaContext] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        plugins: Optional[List[RoutingPlugin]] = None,
        log_level: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize the router.

        Args:
            cache_size: Legacy alias for ``settings.cache_size``; used
                only when the caller passes a non-default value.
            classifier_confidence_threshold: Legacy alias for
                ``settings.low_confidence_threshold``.
            use_cache: Whether to enable the LRU result cache.
            config_path: Path to a YAML/JSON configuration file.
            config: A pre-parsed configuration dictionary, taking
                precedence over ``config_path`` if both are supplied.
            require_config: If ``True``, raise :class:`ConfigValidationError`
                when neither ``config`` nor a valid ``config_path`` is
                available, instead of falling back to a minimal
                auto-generated fallback-only configuration.
            schema: Default :class:`SchemaContext` applied to every
                :meth:`route` call unless overridden per-call.
            dataset_metadata: Default :class:`DatasetMetadata` applied
                to every :meth:`route` call unless overridden per-call.
            plugins: Additional :class:`RoutingPlugin` instances appended
                after the configured pipeline stages.
            log_level: Optional log level override for this module.

        Raises:
            ConfigValidationError: If configuration is invalid, or
                ``require_config=True`` and none was found.
        """
        if log_level is not None:
            set_log_level(log_level)

        if config is not None:
            self._config = RouterConfig.from_dict(config)
        else:
            self._config = RouterConfig.load(config_path, require_config=require_config)

        if cache_size != 4096:
            self._config.settings["cache_size"] = cache_size
        if classifier_confidence_threshold != 0.35:
            self._config.settings["low_confidence_threshold"] = classifier_confidence_threshold

        self._normalizer = TextNormalizer(self._config.settings["normalizer"])
        self._cache = _LRUCache(self._config.settings["cache_size"]) if use_cache else None
        self._default_schema = schema
        self._default_dataset_metadata = dataset_metadata
        self._extra_plugins = list(plugins) if plugins else []

        self._stages: List[_PipelineStage] = self._build_pipeline(self._config)

        self._reload_lock = threading.Lock()

        LOGGER.info(
            "IntentRouter initialised | config_version=%s | labels=%d | "
            "pipeline=%s | cache=%s",
            self._config.config_version,
            len(self._config.intents),
            [stage.plugin.name for stage in self._stages],
            use_cache,
        )

    # --------------------------------------------------------
    # Pipeline construction
    # --------------------------------------------------------

    def _build_pipeline(self, config: RouterConfig) -> List[_PipelineStage]:
        """Instantiate the routing pipeline from ``config.pipeline`` plus
        any directly-injected plugin instances.

        Args:
            config: The active :class:`RouterConfig`.

        Returns:
            An ordered list of compiled :class:`_PipelineStage` objects.
        """
        stages: List[_PipelineStage] = []
        default_threshold = config.settings["low_confidence_threshold"]

        for stage_def in config.pipeline:
            if not stage_def.get("enabled", True):
                continue

            plugin_name = stage_def["plugin"]
            params = dict(stage_def.get("params", {}))

            if plugin_name == "classifier":
                plugin: RoutingPlugin = ClassifierPlugin(
                    training_examples=config.training_examples,
                    prefer_embeddings=params.get("prefer_embeddings", True),
                )
            else:
                factory = PLUGIN_REGISTRY.get(plugin_name)
                if factory is None:
                    LOGGER.warning(
                        "settings.pipeline references unknown plugin '%s'; "
                        "skipping this stage (call register_plugin() to add "
                        "support).",
                        plugin_name,
                    )
                    continue
                plugin = factory(**params)

            if "name" in stage_def:
                plugin.name = str(stage_def["name"])

            stop_on_confidence = float(stage_def.get("stop_on_confidence", default_threshold))
            stages.append(_PipelineStage(plugin=plugin, stop_on_confidence=stop_on_confidence))

        for extra_plugin in self._extra_plugins:
            stages.append(
                _PipelineStage(plugin=extra_plugin, stop_on_confidence=default_threshold)
            )

        return stages

    # --------------------------------------------------------
    # Configuration management
    # --------------------------------------------------------

    def reload_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        require_config: bool = False,
    ) -> None:
        """Hot-reload configuration without restarting the process.

        Validates the new configuration before swapping it in; if
        validation fails, the router keeps its previous, working
        configuration. The result cache is always invalidated on a
        successful reload, since ``config_version`` changes and would
        naturally miss on every cached key anyway — clearing it
        proactively also reclaims memory immediately.

        Args:
            config_path: Path to a new YAML/JSON configuration file.
            config: A pre-parsed configuration dictionary, taking
                precedence over ``config_path`` if both are supplied.
            require_config: See :meth:`__init__`.

        Raises:
            ConfigValidationError: If the new configuration is invalid.
        """
        new_config = (
            RouterConfig.from_dict(config)
            if config is not None
            else RouterConfig.load(config_path, require_config=require_config)
        )
        with self._reload_lock:
            self._config = new_config
            self._normalizer = TextNormalizer(self._config.settings["normalizer"])
            self._stages = self._build_pipeline(self._config)
            if self._cache:
                self._cache.clear()
        LOGGER.info(
            "IntentRouter configuration reloaded | new_version=%s",
            self._config.config_version,
        )

    # --------------------------------------------------------
    # Core routing
    # --------------------------------------------------------

    def route(
        self,
        query: str,
        schema: Optional[SchemaContext] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> IntentResult:
        """Route a raw user query to the appropriate retrieval path.

        Args:
            query: Raw user query text, any language/script.
            schema: Per-call :class:`SchemaContext` override. If omitted,
                the router's default schema (set at construction) is used.
            dataset_metadata: Per-call :class:`DatasetMetadata` override.
                If omitted, the router's default metadata is used.

        Returns:
            A populated :class:`IntentResult`.
        """
        t0 = time.perf_counter()
        fallback_label = str(self._config.settings["fallback_label"])
        fallback_routing = self._config.routing.get(
            fallback_label, {"retrievers": {}, "routing_path": "none", "extra_flags": {}}
        )

        if not query or not query.strip():
            return self._finalize(
                IntentResult(
                    label=fallback_label,
                    retrievers=dict(fallback_routing["retrievers"]),
                    routing_path=fallback_routing["routing_path"],
                    method="empty_query",
                ),
                t0,
            )

        effective_schema = schema if schema is not None else self._default_schema
        effective_metadata = (
            dataset_metadata if dataset_metadata is not None else self._default_dataset_metadata
        )

        normalized = self._normalizer.normalize(query)

        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = _LRUCache.build_key(
                normalized, self._config.config_version, effective_schema, effective_metadata
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                result = IntentResult(
                    **{k: v for k, v in asdict(cached).items() if k != "latency_ms"}
                )
                result.method = "cache"
                return self._finalize(result, t0)

        context = _RoutingContext(
            normalized_text=normalized,
            raw_text=query,
            schema=effective_schema,
            dataset_metadata=effective_metadata,
            config=self._config,
        )

        best_verdict: Optional[_PluginVerdict] = None
        plugin_scores: Dict[str, Any] = {}

        for stage in self._stages:
            try:
                verdict = stage.plugin.evaluate(context)
            except Exception as exc:  # noqa: BLE001 - plugin failures must not break routing
                LOGGER.warning(
                    "Routing plugin '%s' raised an exception (skipped): %s",
                    stage.plugin.name, exc,
                )
                continue

            if verdict is None:
                continue

            plugin_scores[verdict.method] = verdict.diagnostics or {
                "label": verdict.label, "confidence": verdict.confidence
            }

            if best_verdict is None or verdict.confidence > best_verdict.confidence:
                best_verdict = verdict

            if best_verdict.confidence >= stage.stop_on_confidence:
                break

        if best_verdict is None:
            best_verdict = _PluginVerdict(label=fallback_label, confidence=0.0, method="fallback")

        routing_entry = self._config.routing.get(best_verdict.label, fallback_routing)
        if best_verdict.label not in self._config.routing:
            LOGGER.warning(
                "Plugin '%s' returned unknown label '%s'; falling back to "
                "'%s' routing.",
                best_verdict.method, best_verdict.label, fallback_label,
            )

        result = IntentResult(
            label=best_verdict.label,
            retrievers=dict(routing_entry["retrievers"]),
            routing_path=routing_entry["routing_path"],
            confidence=round(best_verdict.confidence, 4),
            method=best_verdict.method,
            extra_flags=dict(routing_entry.get("extra_flags", {})),
            plugin_scores=plugin_scores,
        )

        self._apply_retriever_availability(result, effective_metadata)

        if self._cache is not None and cache_key is not None:
            self._cache.put(cache_key, result)

        return self._finalize(result, t0)

    def _apply_retriever_availability(
        self, result: IntentResult, metadata: Optional[DatasetMetadata]
    ) -> None:
        """Downgrade routing flags to only what's actually available.

        Iterates dynamically over every key currently present in
        ``result.retrievers`` — no fixed retriever-name list — and
        disables any retriever not present in
        ``metadata.available_retrievers``.

        Args:
            result: The :class:`IntentResult` being finalised, mutated
                in place.
            metadata: The effective :class:`DatasetMetadata` for this
                call, if any.
        """
        if metadata is None or not metadata.available_retrievers:
            return

        available = set(metadata.available_retrievers)
        downgraded = False

        for retriever_name in list(result.retrievers.keys()):
            if result.retrievers[retriever_name] and retriever_name not in available:
                result.retrievers[retriever_name] = False
                downgraded = True

        if downgraded:
            result.extra_flags["retriever_downgraded"] = True
            if not any(result.retrievers.values()):
                unavailable_path = str(
                    self._config.settings.get("unavailable_routing_path", "none")
                )
                result.routing_path = unavailable_path
                LOGGER.warning(
                    "No configured retriever is available for this query's "
                    "label '%s'; routing_path set to '%s'.",
                    result.label, unavailable_path,
                )

    def route_batch(
        self,
        queries: List[str],
        schema: Optional[SchemaContext] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> List[IntentResult]:
        """Route multiple queries. Each call is independent and safe to
        parallelize externally.

        Args:
            queries: List of raw query strings.
            schema: Optional shared schema override for all queries.
            dataset_metadata: Optional shared metadata override for all
                queries.

        Returns:
            List of :class:`IntentResult`, one per input query.
        """
        return [self.route(q, schema=schema, dataset_metadata=dataset_metadata) for q in queries]

    # --------------------------------------------------------
    # Classifier management
    # --------------------------------------------------------

    def retrain_classifier(self, labelled_samples: List[Tuple[str, str]]) -> None:
        """Hot-swap the ML fallback classifier(s) with additional
        labelled data. Applies to every :class:`ClassifierPlugin`
        instance currently in the pipeline.

        Args:
            labelled_samples: List of ``(query_text, label)`` pairs.
        """
        grouped: Dict[str, List[str]] = {}
        for text, label in labelled_samples:
            grouped.setdefault(label, []).append(text)

        retrained_any = False
        for stage in self._stages:
            if isinstance(stage.plugin, ClassifierPlugin):
                stage.plugin.retrain(grouped)
                retrained_any = True

        if not retrained_any:
            LOGGER.warning(
                "retrain_classifier() called but no ClassifierPlugin stage "
                "is active in the current pipeline."
            )

        if self._cache:
            self._cache.clear()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def warmup(self) -> None:
        """Pre-train every classifier-capable plugin and, if
        ``settings.warmup_queries`` is non-empty, route each one to
        pre-compile/pre-cache the fast path. Safe to call multiple times.

        Unlike the previous version, this never issues a hardcoded
        warmup query — if no warmup queries are configured, only plugin
        training occurs.
        """
        for stage in self._stages:
            if hasattr(stage.plugin, "ensure_trained"):
                stage.plugin.ensure_trained()  # type: ignore[attr-defined]

        warmup_queries: List[str] = self._config.settings.get("warmup_queries", [])
        for warmup_query in warmup_queries:
            self.route(warmup_query)

        if self._cache:
            self._cache.clear()

    def clear_cache(self) -> None:
        """Remove all cached routing results."""
        if self._cache:
            self._cache.clear()

    @staticmethod
    def _finalize(result: IntentResult, t0: float) -> IntentResult:
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 4)
        return result


# ============================================================
# MODULE-LEVEL SINGLETON
# ============================================================

_default_router: Optional[IntentRouter] = None
_singleton_lock = threading.Lock()


def get_router(**kwargs: Any) -> IntentRouter:
    """Return the module-level singleton :class:`IntentRouter` (lazy init).

    Args:
        **kwargs: Forwarded to :class:`IntentRouter` on first construction
            only; ignored on subsequent calls once the singleton exists.

    Returns:
        The shared :class:`IntentRouter` instance.
    """
    global _default_router
    if _default_router is None:
        with _singleton_lock:
            if _default_router is None:
                _default_router = IntentRouter(**kwargs)
                _default_router.warmup()
    return _default_router


def route(query: str, **kwargs: Any) -> IntentResult:
    """One-line convenience wrapper: ``from intent_router import route``.

    Args:
        query: Raw user query text.
        **kwargs: Forwarded to :meth:`IntentRouter.route` (e.g. ``schema``,
            ``dataset_metadata``).

    Returns:
        A populated :class:`IntentResult`.
    """
    return get_router().route(query, **kwargs)


# ============================================================
# SELF-TEST / DEMO
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="intent_router.py self-test / demo")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML/JSON router configuration file. If omitted, "
             "a minimal fallback-only configuration is used and the demo "
             "will note that no meaningful routing can occur without "
             "real configuration.",
    )
    args = parser.parse_args()

    router = IntentRouter(config_path=args.config)
    router.warmup()

    demo_queries: List[str] = list(router._config.settings.get("demo_queries", []))
    if not demo_queries:
        # Fall back to using training_examples as demo material, if any.
        for examples in router._config.training_examples.values():
            demo_queries.extend(examples)

    if not demo_queries:
        print(
            "No settings.demo_queries or training_examples found in the "
            "active configuration — skipping the query-routing demo.\n"
            "Supply a configuration file with 'settings.demo_queries' "
            "(a list of example query strings) to see routing output here."
        )
    else:
        print(
            f"\n{'Query':<55} {'Label':<20} {'Path':<15} {'Retrievers':<30} "
            f"{'Conf':^6} {'Method':<14} {'ms':>8}"
        )
        print("-" * 155)

        for q in demo_queries:
            r = router.route(q)
            display = (q[:52] + "...") if len(q) > 55 else q
            active_retrievers = ",".join(k for k, v in r.retrievers.items() if v) or "-"
            print(
                f"{display:<55} {r.label:<20} {r.routing_path:<15} "
                f"{active_retrievers:<30} {r.confidence:^6.2f} {r.method:<14} "
                f"{r.latency_ms:>8.4f}"
            )

        print("\n── Cache Demo ──")
        q = demo_queries[0]
        r1 = router.route(q)
        r2 = router.route(q)
        print(f"First call  : {r1.latency_ms:.4f} ms  method={r1.method}")
        print(f"Second call : {r2.latency_ms:.4f} ms  method={r2.method}")

        print("\n── JSON Output ──")
        print(json.dumps(router.route(demo_queries[0]).to_dict(), indent=2))

    print(f"\nActive config_version: {router._config.config_version}")
    print(f"Fallback label: {router._config.settings['fallback_label']}")
    print(f"Pipeline stages: {[s.plugin.name for s in router._stages]}")