#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_generator.py

Automated benchmark question generator for Hybrid RAG pipeline.
Produces structured, semantic, and hybrid questions with ground truth
from ANY tabular dataset, without hardcoding column names or file names.

Key dynamic-dataset features added:
  - Auto-discovers structured/unstructured CSVs in data_dir instead of
    requiring files named exactly "structured.csv"/"unstructured.csv".
  - Per-dataset canonical mapping resolution: looks for a mapping file
    next to the dataset (data_dir/canonical_mapping.json) before falling
    back to a global config/canonical_mapping.json or built-in defaults.
  - Column classification falls back to DuckDB-inferred dtypes when a
    column can't be matched to a canonical synonym, so numeric/text/
    categorical/timestamp columns are still usable even on unfamiliar
    datasets.
  - Token-overlap based fuzzy matching instead of naive substring
    containment (avoids false positives like "id" matching "paid").
  - Diagnostics: after schema inspection, logs which canonical roles
    (rating, category, price, text, etc.) were and were not resolved,
    so failures on a new dataset are visible instead of silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

# -----------------------------------------------------------------------------
# Project configuration (with fallback)
# -----------------------------------------------------------------------------
try:
    from config import settings  # type: ignore
    from config.settings import DATA_DIR, EVAL_DIR, LOGS_DIR  # type: ignore
except ImportError:
    # Fallback defaults
    DATA_DIR = Path("F:/PROJECT/Data/Processed")
    EVAL_DIR = Path("evaluation")
    LOGS_DIR = Path("logs")
    settings = None  # type: ignore

try:
    import duckdb
except ImportError:
    raise ImportError("duckdb is required. Install with: pip install duckdb")

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "benchmark_generator.log"

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
class QuestionType(Enum):
    STRUCTURED = "structured"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RetrievalType(Enum):
    SQL = "sql"
    VECTOR = "vector"
    HYBRID = "hybrid"


class Intent(Enum):
    AGGREGATION = "aggregation"
    FILTER = "filter"
    COMPARE = "compare"
    RANKING = "ranking"
    DISTRIBUTION = "distribution"
    TREND = "trend"
    OPINION = "opinion"
    ASPECT = "aspect"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    KEYWORD = "keyword"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """Configuration parameters for benchmark generation."""

    structured_count: int = 100
    semantic_count: int = 100
    hybrid_count: int = 100
    random_seed: int = 42
    sample_size: int = 10000  # number of rows to sample for question generation
    output_path: Path = EVAL_DIR / "benchmark_questions.json"
    canonical_mapping_path: Optional[Path] = None
    data_dir: Path = DATA_DIR
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.canonical_mapping_path is None:
            # Try dataset-local mapping first, then a global default.
            local_mapping = Path(self.data_dir) / "canonical_mapping.json"
            global_mapping = Path("config/canonical_mapping.json")
            if local_mapping.exists():
                self.canonical_mapping_path = local_mapping
            elif global_mapping.exists():
                self.canonical_mapping_path = global_mapping
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkRecord:
    """Single benchmark question with all metadata and ground truth."""

    id: str
    question: str
    query_type: QuestionType
    difficulty: Difficulty
    intent: Intent
    retrieval_type: RetrievalType
    expected_answer: Union[str, float, int, List[str], Dict[str, Any]]
    expected_keywords: List[str] = field(default_factory=list)
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "question": self.question,
            "query_type": self.query_type.value,
            "difficulty": self.difficulty.value,
            "intent": self.intent.value,
            "retrieval_type": self.retrieval_type.value,
            "expected_answer": self.expected_answer,
            "expected_keywords": self.expected_keywords,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


@dataclass
class DatasetProfile:
    """Schema and statistics of the dataset."""

    structured_table: str
    unstructured_table: Optional[str]
    row_count: int
    columns: Dict[str, str]  # canonical_name -> original_name
    primary_key: Optional[str] = None
    foreign_key: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    timestamp_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)  # table -> join_key

    def get_column(self, canonical: str) -> Optional[str]:
        """Get original column name for a canonical name."""
        return self.columns.get(canonical)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def generate_id(question: str, seed: int = 42) -> str:
    """Generate a deterministic unique ID for a question."""
    content = f"{question}_{seed}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def normalize_text(text: str) -> str:
    """Basic text normalization for keyword extraction."""
    if not isinstance(text, str):
        return ""
    # Lowercase, remove punctuation, split
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text


def tokenize_column_name(name: str) -> Set[str]:
    """Split a column name into lowercase tokens on _, -, space, and camelCase."""
    # Insert boundary before capital letters that follow lowercase (camelCase)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    spaced = re.sub(r"[\s\-]+", "_", spaced)
    tokens = {t.lower() for t in spaced.split("_") if t}
    return tokens


# A small set of stopwords for keyword extraction
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "without", "via", "during",
    "in", "on", "to", "from", "into", "through", "although", "whereas",
    "etc", "e.g", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "am",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "shall", "can", "etc", "vs", "vs.", "e.g", "i.e",
}

# Simple sentiment lexicon (positive/negative)
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic", "wonderful",
    "love", "like", "perfect", "best", "nice", "beautiful", "happy", "satisfied",
    "pleased", "superb", "outstanding", "terrific", "fabulous", "marvelous",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worse", "worst",
    "disappointing", "disappointed", "hate", "dislike", "unhappy", "unsatisfied",
    "inferior", "mediocre", "subpar", "frustrating", "annoying",
}


def extract_keywords(texts: Sequence[str], top_n: int = 5) -> List[str]:
    """Extract top keywords from a list of texts using simple frequency."""
    word_counts = Counter()
    for text in texts:
        normalized = normalize_text(text)
        words = normalized.split()
        for w in words:
            if len(w) > 2 and w not in STOPWORDS:
                word_counts[w] += 1
    # Get top N
    return [word for word, _ in word_counts.most_common(top_n)]


def get_sentiment(texts: Sequence[str]) -> float:
    """Compute average sentiment score based on positive/negative word counts."""
    scores = []
    for text in texts:
        normalized = normalize_text(text)
        words = normalized.split()
        pos = sum(1 for w in words if w in POSITIVE_WORDS)
        neg = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            scores.append(0.0)
        else:
            scores.append((pos - neg) / total)
    return sum(scores) / len(scores) if scores else 0.0


def get_sentiment_label(score: float) -> str:
    if score > 0.2:
        return "positive"
    elif score < -0.2:
        return "negative"
    else:
        return "neutral"


# DuckDB dtype -> broad classification bucket, used as a fallback when a
# column can't be matched to a canonical synonym. This is what lets the
# generator work reasonably well on datasets it has never seen a schema for.
_NUMERIC_DTYPE_PREFIXES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                            "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
                            "FLOAT", "DOUBLE", "DECIMAL", "REAL", "NUMERIC")
_TIMESTAMP_DTYPE_PREFIXES = ("DATE", "TIME", "TIMESTAMP", "INTERVAL")
_BOOL_DTYPE_PREFIXES = ("BOOLEAN",)
_TEXT_DTYPE_PREFIXES = ("VARCHAR", "CHAR", "TEXT", "STRING", "BLOB")


def classify_by_dtype(dtype: str) -> str:
    """Map a DuckDB column dtype string to one of:
    'numeric', 'timestamp', 'categorical', 'text', 'unknown'.
    """
    dtype_up = (dtype or "").upper()
    if any(dtype_up.startswith(p) for p in _NUMERIC_DTYPE_PREFIXES):
        return "numeric"
    if any(dtype_up.startswith(p) for p in _TIMESTAMP_DTYPE_PREFIXES):
        return "timestamp"
    if any(dtype_up.startswith(p) for p in _BOOL_DTYPE_PREFIXES):
        return "categorical"
    if any(dtype_up.startswith(p) for p in _TEXT_DTYPE_PREFIXES):
        return "text"
    return "unknown"


# -----------------------------------------------------------------------------
# Schema Inspector
# -----------------------------------------------------------------------------
class SchemaInspector:
    """
    Reads CSVs, infers schema, builds DatasetProfile.
    Uses canonical mapping to map columns to standardized names, with a
    dtype-based fallback so datasets whose columns don't match any known
    synonym are still classified and usable.
    """

    # Default synonym mapping (if no external file provided).
    # This is intentionally broad-ish, but any dataset with different
    # vocabulary will still work via the dtype fallback in `inspect()`.
    DEFAULT_CANONICAL_MAP = {
        "id": ["id", "row_id", "record_id"],
        "primary_key": ["pk", "primary_key"],
        "foreign_key": ["fk", "foreign_key"],
        "review_id": ["review_id", "reviewid", "review id"],
        "product_id": ["product_id", "productid", "product id", "asin", "parent_asin"],
        "parent_asin": ["parent_asin"],
        "asin": ["asin", "amazon_asin"],
        "brand": ["brand", "manufacturer", "company"],
        "manufacturer": ["manufacturer", "brand"],
        "title": ["title", "product_title", "name"],
        "price": ["price", "selling_price", "list_price"],
        "rating": ["rating", "star_rating", "stars", "score"],
        "avg_rating": ["avg_rating", "average_rating"],
        "review_text": ["review_text", "reviewtext", "review", "text", "content", "body", "comment"],
        "summary": ["summary", "review_summary", "headline"],
        "timestamp": ["timestamp", "date", "review_date", "time", "created_at"],
        "verified_purchase": ["verified_purchase", "verified", "is_verified"],
        "helpfulness": ["helpfulness", "helpful", "helpful_count"],
        "category": ["category", "categories", "product_category", "genre", "type"],
        "aspect": ["aspect", "aspects", "feature"],
        "sentiment": ["sentiment", "sentiment_score"],
        "topic": ["topic", "topics", "theme"],
        "emotion": ["emotion", "emotions"],
        "chunks": ["chunks", "text_chunks"],
        "embeddings": ["embeddings", "vector"],
    }

    def __init__(
        self,
        structured_path: Path,
        unstructured_path: Optional[Path] = None,
        canonical_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        self.structured_path = structured_path
        self.unstructured_path = unstructured_path
        self.canonical_mapping = canonical_mapping or self.DEFAULT_CANONICAL_MAP
        # Invert mapping for fast lookup: synonym -> canonical, plus
        # pre-tokenized synonyms for fuzzy token-overlap matching.
        self._synonym_to_canonical: Dict[str, str] = {}
        self._synonym_tokens: List[Tuple[Set[str], str]] = []
        for canon, syns in self.canonical_mapping.items():
            for syn in syns:
                syn_lower = syn.lower()
                self._synonym_to_canonical[syn_lower] = canon
                self._synonym_tokens.append((tokenize_column_name(syn), canon))

    def inspect(self) -> DatasetProfile:
        """Run schema inspection and return a DatasetProfile."""
        logger.info("Inspecting dataset schema...")
        if not self.structured_path.exists():
            raise FileNotFoundError(f"Structured CSV not found: {self.structured_path}")

        con = duckdb.connect(":memory:")

        structured_table = "structured"
        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {structured_table} AS "
                f"SELECT * FROM read_csv_auto('{self.structured_path}')"
            )
        except Exception as e:
            con.close()
            raise RuntimeError(f"Failed to read structured CSV: {e}") from e

        struct_cols = con.execute(f"SELECT * FROM {structured_table} LIMIT 0").df().columns.tolist()
        if not struct_cols:
            con.close()
            raise ValueError("Structured CSV has no columns or is empty.")

        struct_types = con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name='{structured_table}'"
        ).fetchall()
        col_types = {row[0]: row[1] for row in struct_types}

        canon_map_struct = self._map_columns(struct_cols)

        unstructured_table = None
        canon_map_unstruct: Dict[str, str] = {}
        unstruct_col_types: Dict[str, str] = {}
        if self.unstructured_path and self.unstructured_path.exists():
            unstructured_table = "unstructured"
            try:
                con.execute(
                    f"CREATE OR REPLACE TABLE {unstructured_table} AS "
                    f"SELECT * FROM read_csv_auto('{self.unstructured_path}')"
                )
                unstruct_cols = con.execute(f"SELECT * FROM {unstructured_table} LIMIT 0").df().columns.tolist()
                if unstruct_cols:
                    canon_map_unstruct = self._map_columns(unstruct_cols)
                    unstruct_types = con.execute(
                        f"SELECT column_name, data_type FROM information_schema.columns "
                        f"WHERE table_name='{unstructured_table}'"
                    ).fetchall()
                    unstruct_col_types = {row[0]: row[1] for row in unstruct_types}
            except Exception as e:
                logger.warning(f"Failed to read unstructured CSV: {e}. Proceeding with structured only.")
                unstructured_table = None

        combined_canon = {**canon_map_unstruct, **canon_map_struct}
        combined_types = {**unstruct_col_types, **col_types}

        # Primary key
        pk_canon = None
        for canon, orig in canon_map_struct.items():
            if canon == "primary_key":
                pk_canon = orig
                break
        if pk_canon is None:
            for canon, orig in canon_map_struct.items():
                if canon in ("id", "review_id", "product_id"):
                    pk_canon = orig
                    break

        # Foreign key
        fk_canon = None
        if unstructured_table:
            for canon, orig in canon_map_unstruct.items():
                if canon in ("product_id", "asin", "foreign_key"):
                    fk_canon = orig
                    break

        relationships = {}
        if unstructured_table and pk_canon and fk_canon:
            relationships[structured_table] = pk_canon
            relationships[unstructured_table] = fk_canon
        elif unstructured_table and "review_id" in canon_map_struct and "review_id" in canon_map_unstruct:
            relationships[structured_table] = canon_map_struct["review_id"]
            relationships[unstructured_table] = canon_map_unstruct["review_id"]

        # -------------------------------------------------------------------
        # Classify columns.
        #
        # `combined_canon` is keyed by CANONICAL name and valued by the REAL
        # (original) CSV column name, e.g. {"review_text": "reviewText"}.
        # Classification tests the *canonical* name and stores the
        # *original* column name -- this is required, otherwise downstream
        # SQL uses canonical labels that don't exist in the actual table
        # (Binder errors like: Referenced column "review_text" not found).
        #
        # For columns that did NOT match any known synonym (i.e. their
        # canonical name == their original name, meaning `_map_columns`
        # fell through to `mapping[col] = col`), we fall back to
        # classifying by DuckDB-inferred dtype so unfamiliar datasets still
        # get usable numeric/text/categorical/timestamp columns instead of
        # being silently dropped from every template.
        # -------------------------------------------------------------------
        text_cols: List[str] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        timestamp_cols: List[str] = []
        id_cols: List[str] = []

        known_text = {"review_text", "summary", "title", "category", "aspect", "topic", "emotion", "chunks"}
        known_numeric = {"price", "rating", "avg_rating", "helpfulness"}
        known_categorical = {"brand", "manufacturer", "verified_purchase"}
        known_timestamp = {"timestamp"}
        known_id = {"id", "review_id", "product_id", "asin", "parent_asin", "primary_key", "foreign_key"}

        for canonical_name, original_col in combined_canon.items():
            was_unmatched = canonical_name == original_col
            dtype = combined_types.get(original_col, "")
            dtype_bucket = classify_by_dtype(dtype)

            if canonical_name in known_text:
                text_cols.append(original_col)
            elif canonical_name in known_numeric:
                numeric_cols.append(original_col)
            elif canonical_name in known_categorical:
                categorical_cols.append(original_col)
            elif canonical_name in known_timestamp:
                timestamp_cols.append(original_col)
            elif canonical_name in known_id:
                id_cols.append(original_col)
            elif was_unmatched:
                # Unknown column: classify purely by dtype so it still
                # participates in question generation where relevant.
                if dtype_bucket == "numeric":
                    numeric_cols.append(original_col)
                elif dtype_bucket == "timestamp":
                    timestamp_cols.append(original_col)
                elif dtype_bucket == "text":
                    # Long free text vs short categorical text is hard to
                    # know from dtype alone; use average length as a cheap
                    # heuristic when available.
                    try:
                        avg_len = con.execute(
                            f"SELECT AVG(LENGTH({original_col})) FROM {structured_table} "
                            f"WHERE {original_col} IS NOT NULL"
                        ).fetchone()[0]
                    except Exception:
                        avg_len = None
                    if avg_len is not None and avg_len > 40:
                        text_cols.append(original_col)
                    else:
                        categorical_cols.append(original_col)
                # dtype_bucket == "unknown" -> skip, nothing useful to infer

        row_count = con.execute(f"SELECT COUNT(*) FROM {structured_table}").fetchone()[0]

        profile = DatasetProfile(
            structured_table=structured_table,
            unstructured_table=unstructured_table,
            row_count=row_count,
            columns=combined_canon,
            primary_key=pk_canon,
            foreign_key=fk_canon,
            text_columns=text_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            timestamp_columns=timestamp_cols,
            id_columns=id_cols,
            relationships=relationships,
        )

        con.close()
        logger.info(
            f"Schema inspection complete. Found {len(profile.columns)} columns, {profile.row_count} rows."
        )
        self._log_role_diagnostics(profile)
        return profile

    def _log_role_diagnostics(self, profile: DatasetProfile) -> None:
        """Log which important semantic roles were / weren't resolved, so a
        new dataset's gaps are visible immediately instead of surfacing as
        silent empty output later."""
        roles = {
            "rating/avg_rating (needed for most structured & hybrid Qs)": (
                profile.get_column("rating") or profile.get_column("avg_rating")
            ),
            "title": profile.get_column("title"),
            "price": profile.get_column("price"),
            "category": profile.get_column("category") or (profile.categorical_columns[:1] or [None])[0],
            "brand/manufacturer": profile.get_column("brand") or profile.get_column("manufacturer"),
            "timestamp": profile.get_column("timestamp"),
            "free text (reviews/summary)": (
                profile.get_column("review_text")
                or profile.get_column("summary")
                or (profile.text_columns[:1] or [None])[0]
            ),
        }
        resolved = {k: v for k, v in roles.items() if v}
        missing = [k for k, v in roles.items() if not v]
        logger.info(f"Resolved semantic roles: {resolved}")
        if missing:
            logger.warning(
                "Unresolved semantic roles (related question templates will be skipped): "
                + ", ".join(missing)
            )

    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map original column names to canonical names.

        Matching strategy, in order:
          1. Exact synonym match (case-insensitive).
          2. Token-overlap fuzzy match: tokenize both the column name and
             each synonym (splitting on _, -, space, camelCase) and match
             if they share a token. This avoids false positives like the
             old substring check (`"id" in "paid"`) while still catching
             things like "productTitle" -> "title" via shared tokens, or
             "review_body" -> "review_text" via shared "review" token
             combined with an exact-ish body/text overlap when possible.
          3. Unmatched: keep original name as its own canonical key, to be
             classified later via dtype inference.
        """
        mapping: Dict[str, str] = {}
        for col in columns:
            col_lower = col.lower().strip()

            if col_lower in self._synonym_to_canonical:
                mapping[self._synonym_to_canonical[col_lower]] = col
                continue

            col_tokens = tokenize_column_name(col)
            best_canon = None
            best_overlap = 0
            for syn_tokens, canon in self._synonym_tokens:
                if not syn_tokens or not col_tokens:
                    continue
                overlap = len(col_tokens & syn_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_canon = canon

            if best_canon is not None and best_overlap > 0:
                mapping[best_canon] = col
            else:
                mapping[col] = col  # unknown; classified later via dtype

        return mapping


# -----------------------------------------------------------------------------
# Question Template Repository
# -----------------------------------------------------------------------------
class QuestionTemplateRepository:
    """
    Stores and provides templates for generating questions.
    Each template is a callable that returns a list of BenchmarkRecord.
    """

    def __init__(self):
        self.structured_templates: List[Any] = []
        self.semantic_templates: List[Any] = []
        self.hybrid_templates: List[Any] = []

    def register_structured(self, func: Any) -> None:
        self.structured_templates.append(func)

    def register_semantic(self, func: Any) -> None:
        self.semantic_templates.append(func)

    def register_hybrid(self, func: Any) -> None:
        self.hybrid_templates.append(func)

    def get_structured(self) -> List[Any]:
        return self.structured_templates

    def get_semantic(self) -> List[Any]:
        return self.semantic_templates

    def get_hybrid(self) -> List[Any]:
        return self.hybrid_templates


# -----------------------------------------------------------------------------
# Structured Question Generator
# -----------------------------------------------------------------------------
class StructuredQuestionGenerator:
    """Generates structured (SQL-like) questions with computed answers."""

    def __init__(
        self,
        profile: DatasetProfile,
        con: duckdb.DuckDBPyConnection,
        sample_size: int,
        rng: random.Random,
    ):
        self.profile = profile
        self.con = con
        self.sample_size = sample_size
        self.rng = rng
        self.table = profile.structured_table

    def generate(self, count: int) -> List[BenchmarkRecord]:
        """Generate `count` structured questions."""
        records = []
        templates = [
            self._avg_rating_by_category,
            self._max_price_product,
            self._min_rating_review,
            self._count_by_brand,
            self._top_rated_products,
            self._bottom_rated_products,
            self._filter_by_price,
            self._group_by_rating,
            self._compare_brands,
            self._distribution_of_ratings,
            self._trend_over_time,
        ]
        templates = self.rng.sample(templates, len(templates))
        attempts = 0
        max_attempts = 20
        while len(records) < count and attempts < max_attempts:
            for template in templates:
                if len(records) >= count:
                    break
                try:
                    rec = template()
                    if rec and rec.question:
                        records.append(rec)
                except Exception as e:
                    logger.warning(f"Template {template.__name__} failed: {e}")
            attempts += 1
        logger.info(f"Generated {len(records)} structured questions (target {count})")
        return records

    def _compute_answer(self, sql: str) -> Any:
        try:
            result = self.con.execute(sql).fetchall()
            if not result:
                return None
            if len(result) == 1:
                return result[0][0]
            return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return None

    def _get_category_column(self) -> Optional[str]:
        cat_col = self.profile.get_column("category")
        if not cat_col and self.profile.categorical_columns:
            cat_col = self.profile.categorical_columns[0]
        return cat_col

    def _get_rating_column(self) -> Optional[str]:
        col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not col and self.profile.numeric_columns:
            col = self.profile.numeric_columns[0]
        return col

    def _get_price_column(self) -> Optional[str]:
        return self.profile.get_column("price")

    def _get_brand_column(self) -> Optional[str]:
        col = self.profile.get_column("brand") or self.profile.get_column("manufacturer")
        if not col and self.profile.categorical_columns:
            col = self.profile.categorical_columns[0]
        return col

    def _get_timestamp_column(self) -> Optional[str]:
        col = self.profile.get_column("timestamp")
        if not col and self.profile.timestamp_columns:
            col = self.profile.timestamp_columns[0]
        return col

    def _get_title_column(self) -> Optional[str]:
        col = self.profile.get_column("title")
        if not col and self.profile.id_columns:
            col = self.profile.id_columns[0]
        return col

    def _avg_rating_by_category(self) -> Optional[BenchmarkRecord]:
        cat_col = self._get_category_column()
        if not cat_col:
            return None
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT {cat_col}, AVG({rating_col}) as avg_rating FROM {self.table} GROUP BY {cat_col} ORDER BY avg_rating DESC LIMIT 1"
        answer = self._compute_answer(sql)
        if answer is None:
            return None
        cat_sql = f"SELECT {cat_col} FROM {self.table} GROUP BY {cat_col} ORDER BY AVG({rating_col}) DESC LIMIT 1"
        cat = self._compute_answer(cat_sql)
        if cat is None:
            return None
        question = f"What is the average rating of products in the {cat} category?"
        expected_answer = round(answer, 2)
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.EASY,
            intent=Intent.AGGREGATION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=[cat, "average", "rating"],
            ground_truth={"sql": sql, "category": cat, "average_rating": expected_answer},
        )

    def _max_price_product(self) -> Optional[BenchmarkRecord]:
        price_col = self._get_price_column()
        if not price_col:
            return None
        title_col = self._get_title_column()
        sql = f"SELECT {price_col} FROM {self.table} ORDER BY {price_col} DESC LIMIT 1"
        answer = self._compute_answer(sql)
        if answer is None:
            return None
        title_sql = f"SELECT {title_col} FROM {self.table} ORDER BY {price_col} DESC LIMIT 1" if title_col else None
        title = self._compute_answer(title_sql) if title_sql else "product"
        question = "What is the maximum price among all products?"
        expected_answer = round(answer, 2)
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.EASY,
            intent=Intent.AGGREGATION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["max", "price", "product"],
            ground_truth={"sql": sql, "max_price": expected_answer, "product": title},
        )

    def _min_rating_review(self) -> Optional[BenchmarkRecord]:
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT MIN({rating_col}) FROM {self.table}"
        answer = self._compute_answer(sql)
        if answer is None:
            return None
        question = "What is the minimum rating given to any product?"
        expected_answer = round(answer, 2)
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.EASY,
            intent=Intent.AGGREGATION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["min", "rating"],
            ground_truth={"sql": sql, "min_rating": expected_answer},
        )

    def _count_by_brand(self) -> Optional[BenchmarkRecord]:
        brand_col = self._get_brand_column()
        if not brand_col:
            return None
        sql = f"SELECT {brand_col}, COUNT(*) as cnt FROM {self.table} GROUP BY {brand_col} ORDER BY cnt DESC LIMIT 1"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        brand, count = result[0]
        question = f"How many products are from the brand {brand}?"
        expected_answer = count
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.EASY,
            intent=Intent.AGGREGATION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=[brand, "count", "products"],
            ground_truth={"sql": sql, "brand": brand, "count": count},
        )

    def _top_rated_products(self) -> Optional[BenchmarkRecord]:
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        title_col = self._get_title_column()
        if not title_col:
            return None
        sql = f"SELECT {title_col}, {rating_col} FROM {self.table} ORDER BY {rating_col} DESC LIMIT 5"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        products = [row[0] for row in result]
        question = "What are the top 5 highest-rated products?"
        expected_answer = products
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.RANKING,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["top", "highest", "rated", "products"],
            ground_truth={"sql": sql, "top_products": products},
        )

    def _bottom_rated_products(self) -> Optional[BenchmarkRecord]:
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        title_col = self._get_title_column()
        if not title_col:
            return None
        sql = f"SELECT {title_col}, {rating_col} FROM {self.table} ORDER BY {rating_col} ASC LIMIT 5"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        products = [row[0] for row in result]
        question = "What are the 5 lowest-rated products?"
        expected_answer = products
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.RANKING,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["lowest", "rated", "products"],
            ground_truth={"sql": sql, "bottom_products": products},
        )

    def _filter_by_price(self) -> Optional[BenchmarkRecord]:
        price_col = self._get_price_column()
        if not price_col:
            return None
        avg_sql = f"SELECT AVG({price_col}) FROM {self.table}"
        avg_price = self._compute_answer(avg_sql)
        if avg_price is None:
            return None
        threshold = avg_price * 1.5
        sql = f"SELECT COUNT(*) FROM {self.table} WHERE {price_col} > {threshold}"
        count = self._compute_answer(sql)
        if count is None:
            return None
        question = f"How many products have a price greater than {threshold:.2f}?"
        expected_answer = count
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.FILTER,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["price", "greater", "count"],
            ground_truth={"sql": sql, "threshold": threshold, "count": count},
        )

    def _group_by_rating(self) -> Optional[BenchmarkRecord]:
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT FLOOR({rating_col}) as rating_bin, COUNT(*) as cnt FROM {self.table} GROUP BY rating_bin ORDER BY rating_bin"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        distribution = {row[0]: row[1] for row in result}
        rating_val = self.rng.choice(list(distribution.keys()))
        count = distribution[rating_val]
        question = f"How many products have a rating of {rating_val} stars?"
        expected_answer = count
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.DISTRIBUTION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["rating", "stars", "count"],
            ground_truth={"sql": sql, "rating_bin": rating_val, "count": count},
        )

    def _compare_brands(self) -> Optional[BenchmarkRecord]:
        brand_col = self._get_brand_column()
        if not brand_col:
            return None
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT {brand_col}, AVG({rating_col}) as avg_rating FROM {self.table} GROUP BY {brand_col} ORDER BY avg_rating DESC LIMIT 2"
        result = self.con.execute(sql).fetchall()
        if len(result) < 2:
            return None
        brand1, avg1 = result[0]
        brand2, avg2 = result[1]
        question = f"Which brand has a higher average rating, {brand1} or {brand2}?"
        expected_answer = brand1 if avg1 > avg2 else brand2
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.HARD,
            intent=Intent.COMPARE,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["brand", "average", "rating", "compare"],
            ground_truth={"sql": sql, "brand1": brand1, "avg1": avg1, "brand2": brand2, "avg2": avg2},
        )

    def _distribution_of_ratings(self) -> Optional[BenchmarkRecord]:
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT {rating_col}, COUNT(*) as cnt FROM {self.table} GROUP BY {rating_col} ORDER BY {rating_col}"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        dist = {row[0]: row[1] for row in result}
        question = "How are ratings distributed across all products?"
        expected_answer = dist
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.HARD,
            intent=Intent.DISTRIBUTION,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["distribution", "ratings"],
            ground_truth={"sql": sql, "distribution": dist},
        )

    def _trend_over_time(self) -> Optional[BenchmarkRecord]:
        ts_col = self._get_timestamp_column()
        if not ts_col:
            return None
        rating_col = self._get_rating_column()
        if not rating_col:
            return None
        sql = f"SELECT DATE_TRUNC('month', {ts_col}) as month, AVG({rating_col}) as avg_rating FROM {self.table} GROUP BY month ORDER BY month"
        try:
            result = self.con.execute(sql).fetchall()
        except Exception:
            sql = f"SELECT STRFTIME({ts_col}, '%Y-%m') as month, AVG({rating_col}) as avg_rating FROM {self.table} GROUP BY month ORDER BY month"
            result = self.con.execute(sql).fetchall()
        if not result:
            return None
        month_data = {row[0]: row[1] for row in result}
        month = self.rng.choice(list(month_data.keys()))
        avg = month_data[month]
        question = f"What was the average rating in {month}?"
        expected_answer = round(avg, 2)
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.STRUCTURED,
            difficulty=Difficulty.HARD,
            intent=Intent.TREND,
            retrieval_type=RetrievalType.SQL,
            expected_answer=expected_answer,
            expected_keywords=["trend", "average", "rating", "month"],
            ground_truth={"sql": sql, "month": month, "average_rating": expected_answer},
        )


# -----------------------------------------------------------------------------
# Semantic Question Generator
# -----------------------------------------------------------------------------
class SemanticQuestionGenerator:
    """Generates semantic (text-based) questions with expected keywords/answers."""

    def __init__(
        self,
        profile: DatasetProfile,
        con: duckdb.DuckDBPyConnection,
        sample_size: int,
        rng: random.Random,
    ):
        self.profile = profile
        self.con = con
        self.sample_size = sample_size
        self.rng = rng
        self.table = profile.structured_table
        self.text_col = self._pick_text_column()

    def _pick_text_column(self) -> Optional[str]:
        col = self.profile.get_column("review_text") or self.profile.get_column("summary")
        if col:
            return col
        if self.profile.text_columns:
            return self.profile.text_columns[0]
        return None

    def generate(self, count: int) -> List[BenchmarkRecord]:
        if self.text_col is None:
            logger.warning("No text column found for semantic questions.")
            return []
        records = []
        templates = [
            self._opinion_summary,
            self._pros_cons,
            self._aspect_sentiment,
            self._topic_keywords,
            self._sentiment_distribution,
            self._emotion_analysis,
            self._comparison_opinion,
        ]
        templates = self.rng.sample(templates, len(templates))
        attempts = 0
        max_attempts = 20
        while len(records) < count and attempts < max_attempts:
            for template in templates:
                if len(records) >= count:
                    break
                try:
                    rec = template()
                    if rec and rec.question:
                        records.append(rec)
                except Exception as e:
                    logger.warning(f"Template {template.__name__} failed: {e}")
            attempts += 1
        logger.info(f"Generated {len(records)} semantic questions (target {count})")
        return records

    def _get_sample_texts(self, n: int = 50) -> List[str]:
        sql = f"SELECT {self.text_col} FROM {self.table} WHERE {self.text_col} IS NOT NULL AND LENGTH({self.text_col}) > 10 LIMIT {n}"
        result = self.con.execute(sql).fetchall()
        return [row[0] for row in result]

    def _opinion_summary(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(20)
        if not texts:
            return None
        keywords = extract_keywords(texts, top_n=5)
        sentiment_score = get_sentiment(texts)
        sentiment_label = get_sentiment_label(sentiment_score)
        question = "What is the overall sentiment of the reviews?"
        expected_answer = sentiment_label
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.EASY,
            intent=Intent.SENTIMENT,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=keywords + [sentiment_label],
            ground_truth={"sentiment_score": sentiment_score, "sentiment_label": sentiment_label},
        )

    def _pros_cons(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(30)
        if not texts:
            return None
        word_counts = Counter()
        for text in texts:
            normalized = normalize_text(text)
            words = normalized.split()
            for w in words:
                if len(w) > 2 and w not in STOPWORDS:
                    word_counts[w] += 1
        pos_words = [w for w, _ in word_counts.most_common(10) if w in POSITIVE_WORDS]
        neg_words = [w for w, _ in word_counts.most_common(10) if w in NEGATIVE_WORDS]
        if not pos_words and not neg_words:
            return None
        question = "What are the most common positive and negative aspects mentioned in reviews?"
        expected_answer = {"positive": pos_words, "negative": neg_words}
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.ASPECT,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=pos_words + neg_words,
            ground_truth={"positive_keywords": pos_words, "negative_keywords": neg_words},
        )

    def _aspect_sentiment(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(40)
        if not texts:
            return None
        word_counts = Counter()
        for text in texts:
            normalized = normalize_text(text)
            words = normalized.split()
            for w in words:
                if len(w) > 3 and w not in STOPWORDS:
                    word_counts[w] += 1
        common_words = [w for w, _ in word_counts.most_common(10)]
        if not common_words:
            return None
        aspect = self.rng.choice(common_words)
        sql = f"SELECT {self.text_col} FROM {self.table} WHERE LOWER({self.text_col}) LIKE '%{aspect}%' LIMIT 20"
        aspect_texts = [row[0] for row in self.con.execute(sql).fetchall()]
        if not aspect_texts:
            return None
        sentiment_score = get_sentiment(aspect_texts)
        sentiment_label = get_sentiment_label(sentiment_score)
        question = f"What is the general sentiment regarding '{aspect}' in the reviews?"
        expected_answer = sentiment_label
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.ASPECT,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=[aspect, sentiment_label],
            ground_truth={"aspect": aspect, "sentiment_score": sentiment_score, "sentiment_label": sentiment_label},
        )

    def _topic_keywords(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(50)
        if not texts:
            return None
        keywords = extract_keywords(texts, top_n=10)
        question = "What are the main topics discussed in the reviews?"
        expected_answer = keywords
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.KEYWORD,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=keywords,
            ground_truth={"topics": keywords},
        )

    def _sentiment_distribution(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(50)
        if not texts:
            return None
        scores = []
        for text in texts:
            score = get_sentiment([text])
            scores.append(score)
        pos = sum(1 for s in scores if s > 0.2)
        neg = sum(1 for s in scores if s < -0.2)
        neu = len(scores) - pos - neg
        distribution = {"positive": pos, "neutral": neu, "negative": neg}
        question = "What is the distribution of sentiment in the reviews?"
        expected_answer = distribution
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.HARD,
            intent=Intent.SENTIMENT,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=["sentiment", "distribution"],
            ground_truth={"distribution": distribution},
        )

    def _emotion_analysis(self) -> Optional[BenchmarkRecord]:
        texts = self._get_sample_texts(30)
        if not texts:
            return None
        sentiment_score = get_sentiment(texts)
        if sentiment_score > 0.5:
            emotion = "joy"
        elif sentiment_score > 0.1:
            emotion = "satisfaction"
        elif sentiment_score > -0.1:
            emotion = "neutral"
        elif sentiment_score > -0.5:
            emotion = "disappointment"
        else:
            emotion = "anger"
        question = "What is the predominant emotion expressed in the reviews?"
        expected_answer = emotion
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.HARD,
            intent=Intent.SENTIMENT,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=[emotion, "emotion"],
            ground_truth={"sentiment_score": sentiment_score, "emotion": emotion},
        )

    def _comparison_opinion(self) -> Optional[BenchmarkRecord]:
        brand_col = self.profile.get_column("brand")
        if not brand_col:
            return None
        sql = f"SELECT {brand_col}, COUNT(*) as cnt FROM {self.table} GROUP BY {brand_col} HAVING cnt >= 5 ORDER BY cnt DESC LIMIT 5"
        brands = [row[0] for row in self.con.execute(sql).fetchall()]
        if len(brands) < 2:
            return None
        brand1, brand2 = self.rng.sample(brands, 2)
        sql1 = f"SELECT {self.text_col} FROM {self.table} WHERE {brand_col} = '{brand1}' LIMIT 20"
        texts1 = [row[0] for row in self.con.execute(sql1).fetchall()]
        sql2 = f"SELECT {self.text_col} FROM {self.table} WHERE {brand_col} = '{brand2}' LIMIT 20"
        texts2 = [row[0] for row in self.con.execute(sql2).fetchall()]
        if not texts1 or not texts2:
            return None
        sent1 = get_sentiment(texts1)
        sent2 = get_sentiment(texts2)
        question = f"Compare the user opinions of {brand1} vs {brand2}. Which brand do users prefer?"
        expected_answer = brand1 if sent1 > sent2 else brand2
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.SEMANTIC,
            difficulty=Difficulty.HARD,
            intent=Intent.COMPARE,
            retrieval_type=RetrievalType.VECTOR,
            expected_answer=expected_answer,
            expected_keywords=[brand1, brand2, "opinion", "compare"],
            ground_truth={
                "brand1": brand1,
                "sentiment1": sent1,
                "brand2": brand2,
                "sentiment2": sent2,
                "winner": expected_answer,
            },
        )


# -----------------------------------------------------------------------------
# Hybrid Question Generator
# -----------------------------------------------------------------------------
class HybridQuestionGenerator:
    """Generates hybrid questions that require both structured and semantic retrieval."""

    def __init__(
        self,
        profile: DatasetProfile,
        con: duckdb.DuckDBPyConnection,
        sample_size: int,
        rng: random.Random,
    ):
        self.profile = profile
        self.con = con
        self.sample_size = sample_size
        self.rng = rng
        self.table = profile.structured_table
        self.text_col = self._pick_text_column()

    def _pick_text_column(self) -> Optional[str]:
        col = self.profile.get_column("review_text") or self.profile.get_column("summary")
        if col:
            return col
        if self.profile.text_columns:
            return self.profile.text_columns[0]
        return None

    def generate(self, count: int) -> List[BenchmarkRecord]:
        if self.text_col is None:
            logger.warning("No text column found for hybrid questions.")
            return []
        records = []
        templates = [
            self._highest_rated_with_reasons,
            self._brand_comparison_evidence,
            self._aspect_with_rating,
            self._price_sentiment_relationship,
            self._category_review_summary,
            self._top_products_with_opinions,
        ]
        templates = self.rng.sample(templates, len(templates))
        attempts = 0
        max_attempts = 20
        while len(records) < count and attempts < max_attempts:
            for template in templates:
                if len(records) >= count:
                    break
                try:
                    rec = template()
                    if rec and rec.question:
                        records.append(rec)
                except Exception as e:
                    logger.warning(f"Template {template.__name__} failed: {e}")
            attempts += 1
        logger.info(f"Generated {len(records)} hybrid questions (target {count})")
        return records

    def _highest_rated_with_reasons(self) -> Optional[BenchmarkRecord]:
        rating_col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not rating_col:
            return None
        title_col = self.profile.get_column("title")
        if not title_col:
            return None
        sql = f"SELECT {title_col}, {rating_col} FROM {self.table} ORDER BY {rating_col} DESC LIMIT 1"
        result = self.con.execute(sql).fetchall()
        if not result:
            return None
        product, rating = result[0]
        if self.profile.unstructured_table and self.profile.relationships:
            join_key = self.profile.relationships.get(self.profile.structured_table)
            if join_key:
                sql_text = f"""
                    SELECT {self.text_col} FROM {self.profile.unstructured_table}
                    WHERE {join_key} IN (SELECT {join_key} FROM {self.table} WHERE {title_col} = '{product}')
                    LIMIT 10
                """
            else:
                sql_text = f"SELECT {self.text_col} FROM {self.table} WHERE {title_col} = '{product}' LIMIT 10"
        else:
            sql_text = f"SELECT {self.text_col} FROM {self.table} WHERE {title_col} = '{product}' LIMIT 10"
        texts = [row[0] for row in self.con.execute(sql_text).fetchall()]
        if not texts:
            return None
        keywords = extract_keywords(texts, top_n=5)
        question = f"Product '{product}' has a high rating of {rating}. What are the main reasons mentioned in reviews?"
        expected_answer = keywords
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.REASONING,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=keywords,
            ground_truth={"product": product, "rating": rating, "keywords": keywords},
        )

    def _brand_comparison_evidence(self) -> Optional[BenchmarkRecord]:
        brand_col = self.profile.get_column("brand")
        if not brand_col:
            return None
        rating_col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not rating_col:
            return None
        sql = f"SELECT {brand_col}, COUNT(*) as cnt FROM {self.table} GROUP BY {brand_col} ORDER BY cnt DESC LIMIT 2"
        brands = [row[0] for row in self.con.execute(sql).fetchall()]
        if len(brands) < 2:
            return None
        b1, b2 = brands
        avg1 = self._compute_avg_rating(brand_col, b1, rating_col)
        avg2 = self._compute_avg_rating(brand_col, b2, rating_col)
        sql_text1 = f"SELECT {self.text_col} FROM {self.table} WHERE {brand_col} = '{b1}' LIMIT 20"
        texts1 = [row[0] for row in self.con.execute(sql_text1).fetchall()]
        sql_text2 = f"SELECT {self.text_col} FROM {self.table} WHERE {brand_col} = '{b2}' LIMIT 20"
        texts2 = [row[0] for row in self.con.execute(sql_text2).fetchall()]
        if not texts1 or not texts2:
            return None
        sent1 = get_sentiment(texts1)
        sent2 = get_sentiment(texts2)
        question = f"Compare {b1} and {b2} in terms of user satisfaction. Which brand has more positive reviews?"
        expected_answer = b1 if sent1 > sent2 else b2
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.HARD,
            intent=Intent.COMPARE,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=[b1, b2, "satisfaction", "positive"],
            ground_truth={
                "brand1": b1,
                "avg_rating1": avg1,
                "sentiment1": sent1,
                "brand2": b2,
                "avg_rating2": avg2,
                "sentiment2": sent2,
                "winner": expected_answer,
            },
        )

    def _aspect_with_rating(self) -> Optional[BenchmarkRecord]:
        rating_col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not rating_col:
            return None
        title_col = self.profile.get_column("title")
        if not title_col:
            return None
        sql = f"SELECT {self.text_col} FROM {self.table} WHERE {self.text_col} IS NOT NULL LIMIT 100"
        texts = [row[0] for row in self.con.execute(sql).fetchall()]
        if not texts:
            return None
        word_counts = Counter()
        for text in texts:
            normalized = normalize_text(text)
            words = normalized.split()
            for w in words:
                if len(w) > 3 and w not in STOPWORDS:
                    word_counts[w] += 1
        common = [w for w, _ in word_counts.most_common(10)]
        if not common:
            return None
        aspect = self.rng.choice(common)
        sql_prods = f"SELECT {title_col}, {rating_col} FROM {self.table} WHERE LOWER({self.text_col}) LIKE '%{aspect}%' ORDER BY {rating_col} DESC LIMIT 1"
        result = self.con.execute(sql_prods).fetchall()
        if not result:
            return None
        product, rating = result[0]
        question = f"Which product is the highest-rated that mentions '{aspect}' in its reviews?"
        expected_answer = product
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.HARD,
            intent=Intent.FILTER,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=[aspect, "highest", "rated"],
            ground_truth={"aspect": aspect, "product": product, "rating": rating},
        )

    def _price_sentiment_relationship(self) -> Optional[BenchmarkRecord]:
        price_col = self.profile.get_column("price")
        if not price_col:
            return None
        rating_col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not rating_col:
            return None
        sql = f"SELECT CORR({price_col}, {rating_col}) FROM {self.table}"
        corr = self.con.execute(sql).fetchone()[0]
        if corr is None:
            corr = 0.0
        question = "Is there a correlation between price and user satisfaction (rating)?"
        expected_answer = "positive" if corr > 0.1 else "negative" if corr < -0.1 else "neutral"
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.HARD,
            intent=Intent.REASONING,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=["price", "satisfaction", "correlation"],
            ground_truth={"correlation": corr, "interpretation": expected_answer},
        )

    def _category_review_summary(self) -> Optional[BenchmarkRecord]:
        cat_col = self.profile.get_column("category")
        if not cat_col:
            return None
        sql = f"SELECT {cat_col}, COUNT(*) as cnt FROM {self.table} GROUP BY {cat_col} HAVING cnt >= 10 ORDER BY cnt DESC LIMIT 5"
        categories = [row[0] for row in self.con.execute(sql).fetchall()]
        if not categories:
            return None
        category = self.rng.choice(categories)
        sql_text = f"SELECT {self.text_col} FROM {self.table} WHERE {cat_col} = '{category}' LIMIT 30"
        texts = [row[0] for row in self.con.execute(sql_text).fetchall()]
        if not texts:
            return None
        keywords = extract_keywords(texts, top_n=5)
        question = f"What are the most common keywords in reviews for the category '{category}'?"
        expected_answer = keywords
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.KEYWORD,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=keywords,
            ground_truth={"category": category, "keywords": keywords},
        )

    def _top_products_with_opinions(self) -> Optional[BenchmarkRecord]:
        rating_col = self.profile.get_column("rating") or self.profile.get_column("avg_rating")
        if not rating_col:
            return None
        title_col = self.profile.get_column("title")
        if not title_col:
            return None
        sql = f"SELECT {title_col}, {rating_col} FROM {self.table} ORDER BY {rating_col} DESC LIMIT 5"
        products = [(row[0], row[1]) for row in self.con.execute(sql).fetchall()]
        if not products:
            return None
        top_product, top_rating = products[0]
        sql_text = f"SELECT {self.text_col} FROM {self.table} WHERE {title_col} = '{top_product}' LIMIT 10"
        texts = [row[0] for row in self.con.execute(sql_text).fetchall()]
        if not texts:
            return None
        keywords = extract_keywords(texts, top_n=5)
        question = f"Product '{top_product}' is the highest-rated. What do users like about it?"
        expected_answer = keywords
        return BenchmarkRecord(
            id=generate_id(question),
            question=question,
            query_type=QuestionType.HYBRID,
            difficulty=Difficulty.MEDIUM,
            intent=Intent.OPINION,
            retrieval_type=RetrievalType.HYBRID,
            expected_answer=expected_answer,
            expected_keywords=keywords,
            ground_truth={"product": top_product, "rating": top_rating, "keywords": keywords},
        )

    def _compute_avg_rating(self, brand_col: str, brand: str, rating_col: str) -> float:
        sql = f"SELECT AVG({rating_col}) FROM {self.table} WHERE {brand_col} = '{brand}'"
        result = self.con.execute(sql).fetchone()
        return result[0] if result and result[0] is not None else 0.0


# -----------------------------------------------------------------------------
# Dataset file discovery (dynamic, replaces hardcoded structured.csv / unstructured.csv)
# -----------------------------------------------------------------------------
# Filename hints used to *prefer* a candidate for a given role when several
# CSVs are present. These are hints, not requirements -- if nothing matches,
# we fall back to "first CSV" / "largest remaining CSV" heuristics below.
_STRUCTURED_NAME_HINTS = ("structured", "products", "product", "metadata", "items", "catalog", "main")
_UNSTRUCTURED_NAME_HINTS = ("unstructured", "reviews", "review", "comments", "text", "feedback")


def discover_dataset_files(data_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Find a structured (tabular/attributes) CSV and an optional
    unstructured (free-text) CSV inside `data_dir`, without assuming fixed
    file names.

    Resolution order:
      1. Exact legacy names "structured.csv" / "unstructured.csv" if present
         (keeps old behavior working unchanged).
      2. Filename-hint matching against `_STRUCTURED_NAME_HINTS` /
         `_UNSTRUCTURED_NAME_HINTS`.
      3. If only one CSV exists, treat it as the structured table (many
         datasets are a single flat file with both attributes and text).
      4. If multiple CSVs exist and hints don't disambiguate, pick the
         largest file as structured (attribute tables are usually the
         "main" table) and, if any other CSV contains a long-text-like
         column, use the largest of the rest as unstructured.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # 1. Legacy exact names
    legacy_structured = data_dir / "structured.csv"
    legacy_unstructured = data_dir / "unstructured.csv"
    if legacy_structured.exists():
        return legacy_structured, (legacy_unstructured if legacy_unstructured.exists() else None)

    if len(csv_files) == 1:
        return csv_files[0], None

    def hint_score(path: Path, hints: Sequence[str]) -> int:
        name = path.stem.lower()
        return sum(1 for h in hints if h in name)

    structured_candidates = sorted(
        csv_files, key=lambda p: hint_score(p, _STRUCTURED_NAME_HINTS), reverse=True
    )
    unstructured_candidates = sorted(
        csv_files, key=lambda p: hint_score(p, _UNSTRUCTURED_NAME_HINTS), reverse=True
    )

    structured_path = None
    if hint_score(structured_candidates[0], _STRUCTURED_NAME_HINTS) > 0:
        structured_path = structured_candidates[0]

    unstructured_path = None
    if hint_score(unstructured_candidates[0], _UNSTRUCTURED_NAME_HINTS) > 0:
        cand = unstructured_candidates[0]
        if cand != structured_path:
            unstructured_path = cand

    if structured_path is None:
        # Fall back to largest file as structured.
        remaining = [p for p in csv_files if p != unstructured_path]
        structured_path = max(remaining, key=lambda p: p.stat().st_size)

    if unstructured_path is None:
        remaining = [p for p in csv_files if p != structured_path]
        if remaining:
            unstructured_path = max(remaining, key=lambda p: p.stat().st_size)

    return structured_path, unstructured_path


# -----------------------------------------------------------------------------
# Benchmark Generator (Main orchestrator)
# -----------------------------------------------------------------------------
class BenchmarkGenerator:
    """Main orchestrator for generating benchmark questions."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.profile: Optional[DatasetProfile] = None
        self.con: Optional[duckdb.DuckDBPyConnection] = None
        self.rng = random.Random(config.random_seed)
        self.records: List[BenchmarkRecord] = []

    def run(self) -> None:
        """Execute the benchmark generation pipeline."""
        start_time = time.perf_counter()
        logger.info("Starting benchmark generation.")
        logger.info(f"Configuration: {self.config}")

        # 1. Load canonical mapping (dataset-local, then global, then default)
        mapping = self._load_canonical_mapping()

        # 2. Discover dataset files dynamically (no fixed filenames required)
        structured_path, unstructured_path = discover_dataset_files(self.config.data_dir)
        logger.info(f"Discovered structured file: {structured_path}")
        logger.info(f"Discovered unstructured file: {unstructured_path}")

        inspector = SchemaInspector(
            structured_path=structured_path,
            unstructured_path=unstructured_path,
            canonical_mapping=mapping,
        )
        self.profile = inspector.inspect()
        if self.profile.row_count == 0:
            raise ValueError("Dataset is empty. Cannot generate benchmark.")

        # 3. Connect to DuckDB
        self.con = duckdb.connect(":memory:")
        self.con.execute(
            f"CREATE OR REPLACE TABLE {self.profile.structured_table} AS "
            f"SELECT * FROM read_csv_auto('{structured_path}')"
        )
        if self.profile.unstructured_table and unstructured_path and unstructured_path.exists():
            self.con.execute(
                f"CREATE OR REPLACE TABLE {self.profile.unstructured_table} AS "
                f"SELECT * FROM read_csv_auto('{unstructured_path}')"
            )

        # 4. Generate questions
        self._generate_structured()
        self._generate_semantic()
        self._generate_hybrid()

        # 5. Deduplicate
        self._deduplicate()

        # 6. Validate
        self._validate_records()

        # 7. Shuffle
        self.rng.shuffle(self.records)

        # 8. Export
        self._export()

        # 9. Statistics
        self._print_statistics(time.perf_counter() - start_time)

        # 10. Cleanup
        if self.con:
            self.con.close()

    def _load_canonical_mapping(self) -> Optional[Dict[str, List[str]]]:
        """Load canonical mapping, preferring a dataset-local file so each
        dataset can ship its own column-synonym config without clobbering
        others sharing the same codebase."""
        if self.config.canonical_mapping_path and self.config.canonical_mapping_path.exists():
            try:
                with open(self.config.canonical_mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded canonical mapping from {self.config.canonical_mapping_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load canonical mapping: {e}")

        local_path = Path(self.config.data_dir) / "canonical_mapping.json"
        if local_path.exists():
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded canonical mapping from {local_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load dataset-local mapping: {e}")

        default_path = Path("config/canonical_mapping.json")
        if default_path.exists():
            try:
                with open(default_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded canonical mapping from {default_path}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load default mapping: {e}")

        logger.info("Using built-in default canonical mapping (with dtype-based fallback for unmatched columns).")
        return None

    def _generate_structured(self) -> None:
        logger.info("Generating structured questions...")
        if self.config.structured_count <= 0:
            logger.info("Structured question count is 0; skipping.")
            return
        generator = StructuredQuestionGenerator(
            profile=self.profile,
            con=self.con,
            sample_size=self.config.sample_size,
            rng=self.rng,
        )
        records = generator.generate(self.config.structured_count)
        self.records.extend(records)
        logger.info(f"Generated {len(records)} structured questions.")

    def _generate_semantic(self) -> None:
        logger.info("Generating semantic questions...")
        if self.config.semantic_count <= 0:
            logger.info("Semantic question count is 0; skipping.")
            return
        generator = SemanticQuestionGenerator(
            profile=self.profile,
            con=self.con,
            sample_size=self.config.sample_size,
            rng=self.rng,
        )
        records = generator.generate(self.config.semantic_count)
        self.records.extend(records)
        logger.info(f"Generated {len(records)} semantic questions.")

    def _generate_hybrid(self) -> None:
        logger.info("Generating hybrid questions...")
        if self.config.hybrid_count <= 0:
            logger.info("Hybrid question count is 0; skipping.")
            return
        generator = HybridQuestionGenerator(
            profile=self.profile,
            con=self.con,
            sample_size=self.config.sample_size,
            rng=self.rng,
        )
        records = generator.generate(self.config.hybrid_count)
        self.records.extend(records)
        logger.info(f"Generated {len(records)} hybrid questions.")

    def _deduplicate(self) -> None:
        """Remove duplicate questions (based on question text)."""
        seen = set()
        unique = []
        for rec in self.records:
            key = rec.question.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        removed = len(self.records) - len(unique)
        if removed:
            logger.info(f"Removed {removed} duplicate questions.")
        self.records = unique

    def _validate_records(self) -> None:
        """Validate each record has required fields and non-empty."""
        valid = []
        for rec in self.records:
            if (
                rec.question
                and rec.expected_answer is not None
                and rec.query_type
                and rec.difficulty
                and rec.intent
            ):
                valid.append(rec)
            else:
                logger.warning(f"Invalid record: {rec}")
        invalid = len(self.records) - len(valid)
        if invalid:
            logger.warning(f"Removed {invalid} invalid records.")
        self.records = valid

    def _export(self) -> None:
        """Write benchmark records to JSON file."""
        output_path = self.config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [rec.to_dict() for rec in self.records]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(data)} questions to {output_path}")

    def _print_statistics(self, elapsed: float) -> None:
        """Print generation statistics."""
        total = len(self.records)
        if total == 0:
            logger.warning("No records generated.")
            return
        type_counts = Counter(rec.query_type.value for rec in self.records)
        diff_counts = Counter(rec.difficulty.value for rec in self.records)
        intent_counts = Counter(rec.intent.value for rec in self.records)

        logger.info("=" * 50)
        logger.info("Benchmark Generation Statistics")
        logger.info(f"Total questions: {total}")
        logger.info(f"Generation time: {elapsed:.2f}s")
        logger.info("Question type distribution:")
        for qtype, cnt in type_counts.items():
            logger.info(f"  {qtype}: {cnt}")
        logger.info("Difficulty distribution:")
        for diff, cnt in diff_counts.items():
            logger.info(f"  {diff}: {cnt}")
        logger.info("Intent distribution:")
        for intent, cnt in intent_counts.most_common():
            logger.info(f"  {intent}: {cnt}")
        logger.info("=" * 50)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark questions for RAG pipeline (works on any tabular dataset).")
    parser.add_argument("--structured", type=int, default=100, help="Number of structured questions")
    parser.add_argument("--semantic", type=int, default=100, help="Number of semantic questions")
    parser.add_argument("--hybrid", type=int, default=100, help="Number of hybrid questions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "benchmark_questions.json", help="Output JSON file path")
    parser.add_argument("--sample-size", type=int, default=10000, help="Sample size for question generation")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing the dataset's CSV file(s)")
    parser.add_argument("--canonical-mapping", type=Path, default=None, help="Explicit path to a canonical_mapping.json for this dataset")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = BenchmarkConfig(
        structured_count=args.structured,
        semantic_count=args.semantic,
        hybrid_count=args.hybrid,
        random_seed=args.seed,
        sample_size=args.sample_size,
        output_path=args.output,
        canonical_mapping_path=args.canonical_mapping,
        data_dir=args.data_dir,
        verbose=args.verbose,
    )

    generator = BenchmarkGenerator(config)
    try:
        generator.run()
    except Exception as e:
        logger.error(f"Benchmark generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
