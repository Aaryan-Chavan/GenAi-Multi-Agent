# storage/duckdb_loader.py
"""
Production-grade, schema-aware, dataset-agnostic DuckDB ingestion engine.

This module replaces the "load everything as-is" pattern with a generic
profiling -> type-inference -> validated-cast -> load pipeline that works
for *any* structured dataset (reviews, orders, healthcare records, IoT
telemetry, financial ledgers, ...), without any dataset-specific code.

High level design
------------------
1. ColumnProfiler        - pure-pandas heuristics that look at a column's
                            dtype + a sample of its values and propose the
                            best-fitting DuckDBType (never touches the DB).
2. CastExpressionBuilder  - turns a proposed DuckDBType into a concrete,
                            safe SQL expression (TRY_CAST / STRPTIME /
                            REGEXP_REPLACE / CASE ... END) that DuckDB can
                            execute directly against the raw, zero-copy
                            registered DataFrame.
3. SchemaInferenceEngine  - orchestrates profiling for every column, then
                            *validates* each candidate cast against DuckDB
                            itself (cheap, sampled, single combined query)
                            and downgrades to VARCHAR whenever a cast would
                            silently destroy data. This is what makes the
                            engine safe on real, messy data.
4. DuckDBLoader           - the public API. Registers the DataFrame as a
                            zero-copy view, asks the engine for a schema,
                            and materializes the table with one
                            `CREATE TABLE ... AS SELECT <casts> FROM view`
                            statement (or chunked INSERTs for very large
                            frames). No per-cell Python conversion is ever
                            performed - all casting happens inside DuckDB's
                            vectorized engine, which is both faster and
                            more memory-efficient than pandas-side
                            conversion. CSV files are profiled from a
                            DuckDB-side sample and loaded end-to-end via
                            SQL, so even multi-hundred-million-row files
                            never need to be fully materialized as a
                            pandas DataFrame.

Nothing in this file references a concrete dataset, column name, or
business domain. Every decision is derived from the data itself.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid as uuid_module
from dataclasses import dataclass, field
from datetime import datetime as _datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuration is soft-imported: an existing project's Config.settings
# module is used when available (kept for backward compatibility with
# callers that already rely on it), but the module never hard-fails at
# import time just because that project layout isn't present - safe,
# overridable-via-environment-variable defaults are used instead.
# ---------------------------------------------------------------------
try:
    from Config.settings import (
        DUCKDB_FILE,
        STRUCTURED_TABLE,
        METADATA_TABLE,
        INTELLIGENCE_TABLE,
    )
except Exception:  # pragma: no cover - fallback path for standalone use
    DUCKDB_FILE = os.environ.get("DUCKDB_FILE", "warehouse.duckdb")
    STRUCTURED_TABLE = os.environ.get("DUCKDB_STRUCTURED_TABLE", "structured")
    METADATA_TABLE = os.environ.get("DUCKDB_METADATA_TABLE", "metadata")
    INTELLIGENCE_TABLE = os.environ.get("DUCKDB_INTELLIGENCE_TABLE", "intelligence")


# =====================================================================
# 1. Types & constants
# =====================================================================

class DuckDBType(str, Enum):
    """Every DuckDB type this engine is able to infer and cast into."""

    TINYINT = "TINYINT"
    SMALLINT = "SMALLINT"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    HUGEINT = "HUGEINT"
    DOUBLE = "DOUBLE"
    FLOAT = "FLOAT"
    DECIMAL = "DECIMAL"          # parametrized: precision / scale
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMPTZ = "TIMESTAMP WITH TIME ZONE"
    INTERVAL = "INTERVAL"
    UUID = "UUID"
    JSON = "JSON"
    VARCHAR = "VARCHAR"
    BLOB = "BLOB"


# Boolean token sets (case-insensitive, trimmed). Only ever applied to
# object/string columns - numeric dtypes are never silently reinterpreted
# as booleans, since e.g. an integer column that happens to contain only
# {0, 1} in a sample is far more likely to be a genuine count/flag than a
# string boolean.
_BOOL_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "on", "enabled", "active"}
_BOOL_FALSE_TOKENS = {"false", "f", "no", "n", "0", "off", "disabled", "inactive"}
_BOOL_ALL_TOKENS = _BOOL_TRUE_TOKENS | _BOOL_FALSE_TOKENS

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

_CURRENCY_CHARS = "$€£¥₹"
_NUMERIC_CLEAN_RE = re.compile(r"[,%\s" + re.escape(_CURRENCY_CHARS) + r"]")
_SCI_NOTATION_RE = re.compile(r"[+-]?\d+(\.\d+)?[eE][+-]?\d+")
_LEADING_ZERO_RE = re.compile(r"0\d")

# HUGEINT is DuckDB's largest native integer type; anything outside this
# range (extremely rare in real data) is not silently truncated - it is
# routed to DOUBLE (if it round-trips acceptably) or left as VARCHAR.
_HUGEINT_MIN = -170141183460469231731687303715884105728
_HUGEINT_MAX = 170141183460469231731687303715884105727

# (python strptime format, is_date_only) - ordered by specificity so the
# first confident match wins. Extend freely; nothing else needs to change.
_DATETIME_FORMATS: list[tuple[str, bool]] = [
    ("%Y-%m-%d", True),
    ("%Y/%m/%d", True),
    ("%m/%d/%Y", True),
    ("%d/%m/%Y", True),
    ("%d-%m-%Y", True),
    ("%m-%d-%Y", True),
    ("%Y-%m-%dT%H:%M:%S", False),
    ("%Y-%m-%dT%H:%M:%S.%f", False),
    ("%Y-%m-%d %H:%M:%S", False),
    ("%Y-%m-%d %H:%M:%S.%f", False),
    ("%m/%d/%Y %H:%M:%S", False),
    ("%d/%m/%Y %H:%M:%S", False),
    ("%Y%m%d", True),
    ("%d %b %Y", True),
    ("%B %d, %Y", True),
    ("%a, %d %b %Y %H:%M:%S", False),
]

_SAMPLE_SIZE_DEFAULT = 50_000
_MISMATCH_TOLERANCE = 0.02  # 2% of non-null values may fail to cast
_LEADING_ZERO_TOLERANCE = 0.02  # >2% leading-zero numeric strings -> keep as text


# =====================================================================
# 2. Identifier sanitation
# =====================================================================

class IdentifierSanitizer:
    """
    Makes arbitrary column/table names safe to use as DuckDB identifiers.

    DuckDB accepts virtually any string as a double-quoted identifier, so
    the only real hazards are: empty/duplicate names, embedded double
    quotes, and non-string (e.g. integer / tuple) column labels coming
    from unusual DataFrames. Reserved keywords are handled for free
    because every identifier this module emits is always double-quoted.
    """

    @staticmethod
    def quote(identifier: str) -> str:
        return '"' + str(identifier).replace('"', '""') + '"'

    @staticmethod
    def escape_literal(value: str) -> str:
        """Escape a value for safe embedding as a single-quoted SQL string
        literal (used for file paths / format strings, never for
        user-controlled column data)."""
        return str(value).replace("'", "''")

    @staticmethod
    def sanitize_columns(columns: list[Any]) -> list[str]:
        """
        Returns a list of safe, unique names aligned *positionally* with
        `columns` (never keyed by the original label, since DataFrames can
        legally contain duplicate column labels - a dict keyed by label
        would silently collide in that case). Safe = non-empty, string,
        unique. Original values are preserved verbatim (including
        unicode) whenever possible.
        """
        seen: set[str] = set()
        result: list[str] = []

        for idx, col in enumerate(columns):
            name = str(col).strip() if col is not None and str(col).strip() else f"column_{idx}"

            base = name
            final = base
            counter = 0
            while final in seen:
                counter += 1
                final = f"{base}_{counter}"
            seen.add(final)

            result.append(final)

        return result


# =====================================================================
# 3. Column profiling
# =====================================================================

@dataclass
class ColumnProfile:
    """Everything the engine learned about one column."""

    original_name: Any
    safe_name: str
    pandas_dtype: str
    row_count: int
    null_count: int
    null_ratio: float
    distinct_count: int
    distinct_ratio: float
    inferred_type: DuckDBType
    type_params: dict[str, int] = field(default_factory=dict)
    cast_expression: Optional[str] = None      # filled by CastExpressionBuilder
    notes: list[str] = field(default_factory=list)

    def duckdb_type_sql(self) -> str:
        if self.inferred_type == DuckDBType.DECIMAL:
            p = self.type_params.get("precision", 18)
            s = self.type_params.get("scale", 4)
            return f"DECIMAL({p},{s})"
        return self.inferred_type.value


class ColumnProfiler:
    """
    Pure-pandas heuristics. Never touches the database. Given a Series,
    proposes a DuckDBType + parameters. This is intentionally decoupled
    from SQL generation so it can be unit tested with plain pandas.
    """

    def __init__(self, sample_size: int = _SAMPLE_SIZE_DEFAULT, random_state: int = 42):
        self.sample_size = sample_size
        self.random_state = random_state

    # -- public entry point -------------------------------------------------

    def profile(self, series: pd.Series, original_name: Any, safe_name: str) -> ColumnProfile:
        row_count = len(series)
        null_count = int(series.isna().sum())
        null_ratio = (null_count / row_count) if row_count else 0.0

        non_null = series.dropna()
        distinct_count = int(non_null.nunique(dropna=True)) if not non_null.empty else 0
        distinct_ratio = (distinct_count / len(non_null)) if len(non_null) else 0.0

        sample = self._sample(non_null)

        profile = ColumnProfile(
            original_name=original_name,
            safe_name=safe_name,
            pandas_dtype=str(series.dtype),
            row_count=row_count,
            null_count=null_count,
            null_ratio=null_ratio,
            distinct_count=distinct_count,
            distinct_ratio=distinct_ratio,
            inferred_type=DuckDBType.VARCHAR,
        )

        if sample.empty:
            profile.notes.append("all values null/empty -> defaulting to VARCHAR")
            return profile

        try:
            self._infer(series, sample, profile)
        except Exception as exc:  # never let a single pathological column abort the whole load
            profile.inferred_type = DuckDBType.VARCHAR
            profile.type_params = {}
            profile.notes.append(f"type inference raised {type(exc).__name__}: {exc}; defaulting to VARCHAR")

        return profile

    # -- sampling -------------------------------------------------------

    def _sample(self, non_null: pd.Series) -> pd.Series:
        if len(non_null) <= self.sample_size:
            return non_null
        return non_null.sample(n=self.sample_size, random_state=self.random_state)

    # -- dispatch ---------------------------------------------------------

    def _infer(self, series: pd.Series, sample: pd.Series, profile: ColumnProfile) -> None:
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            profile.inferred_type = DuckDBType.BOOLEAN
            return

        if pd.api.types.is_datetime64_any_dtype(dtype):
            if getattr(dtype, "tz", None) is not None:
                profile.inferred_type = DuckDBType.TIMESTAMPTZ
            elif self._all_midnight(sample):
                profile.inferred_type = DuckDBType.DATE
            else:
                profile.inferred_type = DuckDBType.TIMESTAMP
            return

        if pd.api.types.is_timedelta64_dtype(dtype):
            profile.inferred_type = DuckDBType.INTERVAL
            return

        if pd.api.types.is_integer_dtype(dtype):
            self._infer_integer(sample, profile)
            return

        if pd.api.types.is_float_dtype(dtype):
            self._infer_float(sample, profile)
            return

        if isinstance(sample.iloc[0], (bytes, bytearray)):
            profile.inferred_type = DuckDBType.BLOB
            return

        # Generic object/string/categorical/extension-string column: run
        # the full string-based detector cascade. Order matters - most
        # specific / cheapest checks first. `.astype(str)` below works
        # uniformly for pandas 'object', nullable 'string', pandas>=3
        # default 'str', Arrow-backed string, and 'category' dtypes.
        self._infer_object_column(sample, profile)

    # -- numeric dtypes -----------------------------------------------------

    def _infer_integer(self, sample: pd.Series, profile: ColumnProfile) -> None:
        lo, hi = int(sample.min()), int(sample.max())
        profile.inferred_type = self._smallest_int_type(lo, hi, profile)

    def _infer_float(self, sample: pd.Series, profile: ColumnProfile) -> None:
        finite = sample[np.isfinite(sample)]
        if finite.empty:
            profile.inferred_type = DuckDBType.DOUBLE
            profile.notes.append("non-finite float column")
            return

        # Whole-number floats (e.g. produced by an upstream NaN-forced
        # float64 column) collapse back down to the smallest int type.
        if (finite == finite.round()).all():
            lo, hi = int(finite.min()), int(finite.max())
            profile.inferred_type = self._smallest_int_type(lo, hi, profile)
            profile.notes.append("float column with only integral values -> integer type")
            return

        # Currency-like precision (exactly 2 decimals) is extremely
        # common and a strong DECIMAL signal.
        scaled = finite * 100
        if np.allclose(scaled, scaled.round(), atol=1e-6):
            precision = max(len(str(int(abs(finite).max()))) + 2, 4)
            profile.inferred_type = DuckDBType.DECIMAL
            profile.type_params = {"precision": min(precision, 38), "scale": 2}
            profile.notes.append("2-decimal precision detected -> DECIMAL")
            return

        profile.inferred_type = DuckDBType.DOUBLE

    @staticmethod
    def _smallest_int_type(lo: int, hi: int, profile: Optional[ColumnProfile] = None) -> DuckDBType:
        if -128 <= lo and hi <= 127:
            return DuckDBType.TINYINT
        if -32768 <= lo and hi <= 32767:
            return DuckDBType.SMALLINT
        if -2147483648 <= lo and hi <= 2147483647:
            return DuckDBType.INTEGER
        if -9223372036854775808 <= lo and hi <= 9223372036854775807:
            return DuckDBType.BIGINT
        if _HUGEINT_MIN <= lo and hi <= _HUGEINT_MAX:
            return DuckDBType.HUGEINT
        if profile is not None:
            profile.notes.append(
                f"integer values ({lo}..{hi}) exceed DuckDB's HUGEINT range -> keeping VARCHAR"
            )
        return DuckDBType.VARCHAR

    # -- object/string dtype --------------------------------------------

    def _infer_object_column(self, sample: pd.Series, profile: ColumnProfile) -> None:
        str_sample = sample.astype(str).str.strip()
        # drop empty strings the same way we'd drop nulls
        str_sample = str_sample[str_sample != ""]
        if str_sample.empty:
            profile.notes.append("only empty strings -> VARCHAR")
            return

        n = len(str_sample)
        lowered = str_sample.str.lower()

        # --- boolean -----------------------------------------------------
        # Any column whose entire distinct value set is drawn from
        # recognized boolean tokens (true/false, yes/no, y/n, t/f, 1/0,
        # on/off, enabled/disabled, active/inactive - possibly several
        # representations mixed together) is boolean.
        uniq_tokens = set(lowered.unique())
        if uniq_tokens and uniq_tokens.issubset(_BOOL_ALL_TOKENS):
            profile.inferred_type = DuckDBType.BOOLEAN
            return

        # --- UUID ----------------------------------------------------------
        uuid_hits = str_sample.str.fullmatch(_UUID_RE).sum()
        if uuid_hits / n >= 0.98:
            profile.inferred_type = DuckDBType.UUID
            return

        # --- JSON ------------------------------------------------------
        json_like = str_sample[str_sample.str.startswith(("{", "["))]
        if len(json_like) / n >= 0.9 and self._json_parse_ratio(json_like) >= 0.95:
            profile.inferred_type = DuckDBType.JSON
            return

        # --- identifier-like strings that merely *look* numeric ---------
        # Values such as "00123", "0042" carry meaningful leading zeros
        # (product codes, zip codes, ticket numbers, ...); casting them to
        # an integer type would silently destroy that formatting. This is
        # a pure data-shape heuristic - it never inspects the column name.
        leading_zero_ratio = str_sample.str.match(_LEADING_ZERO_RE).sum() / n
        if leading_zero_ratio > _LEADING_ZERO_TOLERANCE:
            profile.notes.append(
                f"{leading_zero_ratio:.1%} of sampled values have significant leading "
                "zeros -> keeping VARCHAR to preserve formatting"
            )
            return

        # --- numeric (int / decimal / double), incl. currency/%/sci -----
        cleaned = str_sample.str.replace(_NUMERIC_CLEAN_RE, "", regex=True)
        cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123
        numeric_mask = cleaned.str.fullmatch(r"[+-]?\d+(\.\d+)?") | cleaned.str.fullmatch(_SCI_NOTATION_RE)
        numeric_ratio = numeric_mask.sum() / n
        if numeric_ratio >= 0.95:
            self._infer_numeric_strings(cleaned[numeric_mask], str_sample, profile)
            return

        # --- date / timestamp --------------------------------------------
        fmt, is_date_only, hit_ratio = self._detect_datetime_format(str_sample)
        if fmt and hit_ratio >= 0.9:
            profile.inferred_type = DuckDBType.DATE if is_date_only else DuckDBType.TIMESTAMP
            profile.type_params["strptime_format"] = fmt
            return

        # Also try pandas' own flexible parser as a fallback for ISO-ish
        # / RFC-ish timestamps that don't match a single fixed format.
        parsed = pd.to_datetime(str_sample, errors="coerce", utc=False, format="mixed")
        if parsed.notna().sum() / n >= 0.9:
            profile.inferred_type = (
                DuckDBType.DATE if self._all_midnight(parsed.dropna()) else DuckDBType.TIMESTAMP
            )
            profile.notes.append("mixed/flexible datetime parse (no single strptime format)")
            return

        profile.inferred_type = DuckDBType.VARCHAR

    def _infer_numeric_strings(self, cleaned_numeric: pd.Series, original: pd.Series, profile: ColumnProfile) -> None:
        values = pd.to_numeric(cleaned_numeric, errors="coerce")
        values = values.dropna()
        if values.empty:
            profile.inferred_type = DuckDBType.VARCHAR
            return

        # Currency / percentage detection is recorded as a note (the cast
        # itself strips the symbols regardless of the note).
        if original.str.contains("[" + re.escape(_CURRENCY_CHARS) + "]", regex=True).any():
            profile.notes.append("currency symbols detected and stripped")
        if original.str.endswith("%").any():
            profile.notes.append("percentage symbol detected and stripped (value kept as-is, not divided by 100)")

        is_whole = (values == values.round()).all()
        if is_whole and (values.abs() < 1e15).all():
            lo, hi = int(values.min()), int(values.max())
            profile.inferred_type = self._smallest_int_type(lo, hi, profile)
            return

        # decimal-place consistency -> DECIMAL, else DOUBLE
        decimals = cleaned_numeric.str.extract(r"\.(\d+)$")[0].dropna()
        if not decimals.empty:
            max_scale = int(decimals.str.len().max())
            max_int_len = int(cleaned_numeric.str.extract(r"^-?(\d+)")[0].str.len().max() or 1)
            precision = min(max_int_len + max_scale, 38)
            if max_scale <= 12 and precision <= 38 and (decimals.str.len().nunique() <= 3):
                profile.inferred_type = DuckDBType.DECIMAL
                profile.type_params = {"precision": max(precision, max_scale + 1), "scale": max_scale}
                return

        profile.inferred_type = DuckDBType.DOUBLE

    @staticmethod
    def _json_parse_ratio(candidates: pd.Series) -> float:
        if candidates.empty:
            return 0.0
        ok = 0
        for v in candidates:
            try:
                parsed = json.loads(v)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, (dict, list)):
                ok += 1
        return ok / len(candidates)

    @staticmethod
    def _all_midnight(values: pd.Series) -> bool:
        try:
            times = pd.to_datetime(values).dt.time
        except Exception:
            return False
        return bool((times == pd.Timestamp("00:00:00").time()).all())

    @staticmethod
    def _detect_datetime_format(str_sample: pd.Series) -> tuple[Optional[str], bool, float]:
        n = len(str_sample)
        probe = str_sample if n <= 500 else str_sample.sample(n=500, random_state=42)
        for fmt, is_date_only in _DATETIME_FORMATS:
            hits = 0
            for v in probe:
                try:
                    _datetime.strptime(v, fmt)
                    hits += 1
                except (ValueError, TypeError):
                    continue
            ratio = hits / len(probe)
            if ratio >= 0.9:
                return fmt, is_date_only, ratio
        return None, False, 0.0


# =====================================================================
# 4. SQL cast expression construction
# =====================================================================

class CastExpressionBuilder:
    """
    Converts a ColumnProfile into a concrete, safe DuckDB SQL expression.
    All expressions use TRY_CAST (never CAST) so a bad value degrades to
    NULL instead of aborting the whole load; the SchemaInferenceEngine
    separately checks whether that degradation rate is acceptable.
    """

    @staticmethod
    def build(profile: ColumnProfile) -> str:
        # Always reference the sanitized, guaranteed-unique name - the
        # DataFrame registered with DuckDB has already been renamed to use
        # these names, so this is never ambiguous even when the original
        # dataset had duplicate/blank/reserved-word column labels.
        col = IdentifierSanitizer.quote(profile.safe_name)
        t = profile.inferred_type

        if t == DuckDBType.VARCHAR:
            return f"CAST({col} AS VARCHAR)"

        if t == DuckDBType.BOOLEAN:
            true_list = ",".join(f"'{tok}'" for tok in _BOOL_TRUE_TOKENS)
            false_list = ",".join(f"'{tok}'" for tok in _BOOL_FALSE_TOKENS)
            return (
                "CASE "
                f"WHEN LOWER(TRIM(CAST({col} AS VARCHAR))) IN ({true_list}) THEN TRUE "
                f"WHEN LOWER(TRIM(CAST({col} AS VARCHAR))) IN ({false_list}) THEN FALSE "
                "ELSE NULL END"
            )

        if t == DuckDBType.UUID:
            return f"TRY_CAST(CAST({col} AS VARCHAR) AS UUID)"

        if t == DuckDBType.JSON:
            return f"TRY_CAST(CAST({col} AS VARCHAR) AS JSON)"

        if t in (DuckDBType.DATE, DuckDBType.TIMESTAMP, DuckDBType.TIMESTAMPTZ):
            fmt = profile.type_params.get("strptime_format")
            target = t.value
            if fmt:
                inner = f"TRY_STRPTIME(TRIM(CAST({col} AS VARCHAR)), '{fmt}')"
                return f"TRY_CAST({inner} AS {target})"
            return f"TRY_CAST({col} AS {target})"

        if t == DuckDBType.INTERVAL:
            return f"TRY_CAST(CAST({col} AS VARCHAR) AS INTERVAL)"

        if t in (
            DuckDBType.TINYINT, DuckDBType.SMALLINT, DuckDBType.INTEGER,
            DuckDBType.BIGINT, DuckDBType.HUGEINT, DuckDBType.DOUBLE, DuckDBType.FLOAT,
        ):
            cleaned = CastExpressionBuilder._numeric_cleanup_expr(col, profile)
            return f"TRY_CAST({cleaned} AS {t.value})"

        if t == DuckDBType.DECIMAL:
            p = profile.type_params.get("precision", 18)
            s = profile.type_params.get("scale", 4)
            cleaned = CastExpressionBuilder._numeric_cleanup_expr(col, profile)
            return f"TRY_CAST({cleaned} AS DECIMAL({p},{s}))"

        if t == DuckDBType.BLOB:
            return f"TRY_CAST({col} AS BLOB)"

        return f"CAST({col} AS VARCHAR)"

    @staticmethod
    def _numeric_cleanup_expr(col: str, profile: ColumnProfile) -> str:
        # Only string-origin columns need symbol stripping; genuinely
        # numeric pandas dtypes can be cast directly. Covers classic
        # 'object', pandas' nullable 'string', and pandas >= 3's default
        # 'str' dtype (plus its parametrized 'string[...]' variants), as
        # well as 'category' dtype (whose categories are strings here).
        dtype_lower = profile.pandas_dtype.lower()
        is_stringy = (
            dtype_lower in ("object", "string", "str", "category")
            or dtype_lower.startswith("string")
        )
        if not is_stringy:
            return col
        currency_class = "[" + re.escape(_CURRENCY_CHARS) + r",\s%()]"
        stripped = f"REGEXP_REPLACE(TRIM(CAST({col} AS VARCHAR)), '{currency_class}', '', 'g')"
        # handle accounting-style negatives: "(123.45)" - the char-class
        # strip above removes the parens, so we special-case the leading
        # '(' on the *original* text and negate the result.
        return (
            f"CASE WHEN TRIM(CAST({col} AS VARCHAR)) LIKE '(%' "
            f"THEN -1 * TRY_CAST({stripped} AS DOUBLE) "
            f"ELSE TRY_CAST({stripped} AS DOUBLE) END"
        )


# =====================================================================
# 5. Schema inference engine (profiling + DB-side validation)
# =====================================================================

class SchemaInferenceEngine:
    """
    Ties together ColumnProfiler + CastExpressionBuilder, and validates
    every candidate cast against DuckDB itself before committing to it -
    all columns are checked in a *single* combined SQL query for
    efficiency, regardless of how many columns the dataset has.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        sample_size: int = _SAMPLE_SIZE_DEFAULT,
        mismatch_tolerance: float = _MISMATCH_TOLERANCE,
    ):
        self.conn = conn
        self.profiler = ColumnProfiler(sample_size=sample_size)
        self.sample_size = sample_size
        self.mismatch_tolerance = mismatch_tolerance

    def infer(self, df: pd.DataFrame, view_name: str, original_labels: list[Any]) -> list[ColumnProfile]:
        """
        `df` must already have unique, SQL-safe column labels (the caller
        renames it before registering with DuckDB). `original_labels` is
        the pre-rename label list, positionally aligned, kept only for
        human-readable reporting. `view_name` must refer to a DuckDB
        relation (table or view) that also exposes these same safe names,
        used for the validation query below.
        """
        profiles: list[ColumnProfile] = []
        for i, safe_col in enumerate(df.columns):
            profile = self.profiler.profile(df.iloc[:, i], original_labels[i], safe_col)
            profile.cast_expression = CastExpressionBuilder.build(profile)
            profiles.append(profile)

        self._validate_against_duckdb(view_name, profiles)
        return profiles

    # -- validation --------------------------------------------------------

    def _validate_against_duckdb(self, view_name: str, profiles: list[ColumnProfile]) -> None:
        candidates = [p for p in profiles if p.inferred_type != DuckDBType.VARCHAR]
        if not candidates:
            return

        # Aliases are synthetic and positional (col0_total, col0_failed, ...)
        # rather than derived from the (arbitrary, possibly quote-laden)
        # safe_name, so this query is correct regardless of what characters
        # appear in real column names.
        metrics = []
        for i, p in enumerate(candidates):
            safe_col = IdentifierSanitizer.quote(p.safe_name)
            metrics.append(
                f'SUM(CASE WHEN {safe_col} IS NOT NULL THEN 1 ELSE 0 END) AS col{i}_total,'
                f'SUM(CASE WHEN {safe_col} IS NOT NULL AND ({p.cast_expression}) IS NULL '
                f'THEN 1 ELSE 0 END) AS col{i}_failed'
            )

        query = (
            f"SELECT {', '.join(metrics)} "
            f"FROM (SELECT * FROM {view_name} USING SAMPLE {self.sample_size} ROWS)"
        )

        try:
            result = self.conn.execute(query).fetchone()
        except duckdb.Error as exc:
            # If validation itself fails (extreme edge case), fail safe by
            # downgrading every candidate to VARCHAR rather than risking a
            # broken load.
            for p in candidates:
                p.inferred_type = DuckDBType.VARCHAR
                p.type_params = {}
                p.cast_expression = CastExpressionBuilder.build(p)
                p.notes.append(f"validation query failed ({exc}); downgraded to VARCHAR")
            return

        for i, p in enumerate(candidates):
            total = result[i * 2] or 0
            failed = result[i * 2 + 1] or 0
            rate = (failed / total) if total else 0.0
            if rate > self.mismatch_tolerance:
                p.notes.append(
                    f"cast to {p.inferred_type.value} failed on {rate:.1%} of sampled "
                    "non-null values -> downgraded to VARCHAR"
                )
                p.inferred_type = DuckDBType.VARCHAR
                p.type_params = {}
                p.cast_expression = CastExpressionBuilder.build(p)


# =====================================================================
# 6. DuckDBLoader - public API
# =====================================================================

class DuckDBLoader:
    """
    Generic, dataset-agnostic DuckDB ingestion engine.

    Usage
    -----
        loader = DuckDBLoader()
        loader.load_dataframe(any_df, "any_table_name")
        loader.load_structured(structured_df)
        loader.load_metadata(metadata_df)
        report = loader.get_last_schema_report()   # inspect inferred types

    Thread safety
    -------------
    A single `duckdb.DuckDBPyConnection` is not safe for concurrent use
    from multiple threads (DuckDB's own recommendation is one connection
    per thread, or `conn.cursor()` per thread). Since this class is
    typically shared as a long-lived singleton in ingestion pipelines, all
    public methods that touch `self.conn` serialize access via an
    internal `threading.RLock`, so a single `DuckDBLoader` instance can be
    safely shared across threads without callers needing to reason about
    DuckDB's own concurrency model.
    """

    def __init__(self):
        self.db_file = str(DUCKDB_FILE)
        Path(self.db_file).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_file)
        self._last_schema_report: list[ColumnProfile] = []
        self._lock = threading.RLock()

    # ==================================================================
    # CORE GENERIC LOADER
    # ==================================================================

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str = "replace",
        sample_size: int = _SAMPLE_SIZE_DEFAULT,
        chunk_size: Optional[int] = None,
        validate: bool = True,
        mismatch_tolerance: float = _MISMATCH_TOLERANCE,
    ) -> list[ColumnProfile]:
        """
        Infers the best DuckDB schema for `df` and materializes it as
        `table_name`, using zero-copy registration + vectorized SQL casts
        (no per-cell Python conversion, regardless of row count).

        mode:
          - "replace": CREATE OR REPLACE TABLE (default)
          - "append":  INSERT INTO an existing table (created if absent).
                       Column-set compatibility with the existing table is
                       validated *before* any data is inserted; a mismatch
                       raises `ValueError` with the specific missing /
                       unexpected columns.
          - "fail":    raise if the table already exists
          - "upsert":  design-ready, not yet implemented (see docstring
                       below on how to extend it)

        chunk_size: if set, the DataFrame is written in batches of this
        many rows. Schema inference always runs once, against a sample of
        the *full* frame, so the resulting types are consistent across
        chunks. Useful for capping peak memory on very large frames.

        mismatch_tolerance: fraction (0-1) of sampled non-null values that
        may fail a candidate cast before that column is downgraded to
        VARCHAR. Defaults to 2%.
        """
        with self._lock:
            if df is None or df.empty:
                print(f"Skipped {table_name} (empty)")
                return []

            if mode not in ("replace", "append", "fail", "upsert"):
                raise ValueError(
                    f"Unknown load mode '{mode}'. Expected one of: replace, append, fail, upsert."
                )

            if mode == "fail" and self.table_exists(table_name):
                raise ValueError(f"Table '{table_name}' already exists (mode='fail').")

            if mode == "upsert":
                raise NotImplementedError(
                    "Upsert mode is design-ready but not implemented: it requires a "
                    "caller-supplied primary/unique key. Implement via "
                    "`INSERT INTO table ... ON CONFLICT (<key>) DO UPDATE SET ...` "
                    "once a key column (or composite key) is specified."
                )

            original_labels = list(df.columns)
            safe_names = IdentifierSanitizer.sanitize_columns(original_labels)

            # Work on a relabeled *view* of the data (cheap - just swaps the
            # column Index, no data copy) so every downstream step - pandas
            # profiling, DuckDB registration, and generated SQL - operates on
            # guaranteed-unique, SQL-safe names. This is what makes duplicate
            # labels, blank labels, and reserved-word labels safe to ingest.
            df_safe = df.copy(deep=False)
            df_safe.columns = safe_names

            view_name = f"__stage_{re.sub(r'[^0-9a-zA-Z_]', '_', table_name)}_{uuid_module.uuid4().hex[:8]}"

            try:
                self.conn.register(view_name, df_safe)
            except duckdb.Error as exc:
                raise RuntimeError(
                    f"Failed to register DataFrame for table '{table_name}' with DuckDB: {exc}"
                ) from exc

            try:
                engine = SchemaInferenceEngine(
                    self.conn, sample_size=sample_size, mismatch_tolerance=mismatch_tolerance
                )
                if validate:
                    profiles = engine.infer(df_safe, view_name, original_labels)
                else:
                    profiler = ColumnProfiler(sample_size=sample_size)
                    profiles = []
                    for i, safe_col in enumerate(df_safe.columns):
                        p = profiler.profile(df_safe.iloc[:, i], original_labels[i], safe_col)
                        p.cast_expression = CastExpressionBuilder.build(p)
                        profiles.append(p)

                if mode == "append" and self.table_exists(table_name):
                    self._validate_append_compatibility(table_name, profiles)

                select_clause = self._build_select_clause(profiles)

                if chunk_size and len(df_safe) > chunk_size:
                    self._load_chunked(df_safe, table_name, select_clause, mode, chunk_size)
                else:
                    self._load_single_shot(view_name, table_name, select_clause, mode)

                self._last_schema_report = profiles
                self._print_schema_summary(table_name, len(df), profiles)
                return profiles
            finally:
                try:
                    self.conn.unregister(view_name)
                except duckdb.Error as exc:
                    print(f"Warning: failed to unregister staging view '{view_name}': {exc}")

    # -- internal loading strategies ---------------------------------------

    def _build_select_clause(self, profiles: list[ColumnProfile]) -> str:
        parts = [
            f'{p.cast_expression} AS {IdentifierSanitizer.quote(p.safe_name)}'
            for p in profiles
        ]
        return ", ".join(parts)

    def _validate_append_compatibility(self, table_name: str, profiles: list[ColumnProfile]) -> None:
        """
        Before appending, verify the incoming column set exactly matches
        the existing table's column set (order-insensitive). This is a
        pure structural check (names only, not types) - DuckDB's own
        INSERT already enforces type-level compatibility per column, but
        without this check a column-count or column-name mismatch would
        either fail with a cryptic binder error or, worse, silently
        insert values into the wrong columns by position.
        """
        existing_cols = [
            row[0] for row in self.conn.execute(
                f"DESCRIBE {IdentifierSanitizer.quote(table_name)}"
            ).fetchall()
        ]
        incoming_cols = [p.safe_name for p in profiles]

        if set(existing_cols) != set(incoming_cols):
            missing = sorted(set(existing_cols) - set(incoming_cols))
            unexpected = sorted(set(incoming_cols) - set(existing_cols))
            raise ValueError(
                f"Cannot append to table '{table_name}': incoming column set does not "
                f"match the existing table. Missing columns: {missing or 'none'}; "
                f"unexpected columns: {unexpected or 'none'}."
            )

    def _load_single_shot(self, view_name: str, table_name: str, select_clause: str, mode: str) -> None:
        table_ident = IdentifierSanitizer.quote(table_name)
        try:
            if mode == "append" and self.table_exists(table_name):
                self.conn.execute(f"INSERT INTO {table_ident} SELECT {select_clause} FROM {view_name}")
            else:
                self.conn.execute(
                    f"CREATE OR REPLACE TABLE {table_ident} AS SELECT {select_clause} FROM {view_name}"
                )
        except duckdb.Error as exc:
            raise RuntimeError(f"Failed to materialize table '{table_name}': {exc}") from exc

    def _load_chunked(
        self,
        df: pd.DataFrame,
        table_name: str,
        select_clause: str,
        mode: str,
        chunk_size: int,
    ) -> None:
        table_ident = IdentifierSanitizer.quote(table_name)
        table_created = mode == "append" and self.table_exists(table_name)

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            chunk_view = f"__chunk_{uuid_module.uuid4().hex[:8]}"
            try:
                self.conn.register(chunk_view, chunk)
            except duckdb.Error as exc:
                raise RuntimeError(
                    f"Failed to register chunk [{start}:{start + chunk_size}] for "
                    f"table '{table_name}': {exc}"
                ) from exc
            try:
                query = f"SELECT {select_clause} FROM {chunk_view}"
                if not table_created:
                    self.conn.execute(f"CREATE OR REPLACE TABLE {table_ident} AS {query}")
                    table_created = True
                else:
                    self.conn.execute(f"INSERT INTO {table_ident} {query}")
            except duckdb.Error as exc:
                raise RuntimeError(
                    f"Failed to load chunk [{start}:{start + chunk_size}] into "
                    f"table '{table_name}': {exc}"
                ) from exc
            finally:
                try:
                    self.conn.unregister(chunk_view)
                except duckdb.Error as exc:
                    print(f"Warning: failed to unregister chunk view '{chunk_view}': {exc}")

    def _print_schema_summary(self, table_name: str, row_count: int, profiles: list[ColumnProfile]) -> None:
        downgraded = sum(1 for p in profiles if p.inferred_type == DuckDBType.VARCHAR)
        print(f"Loaded {row_count:,} rows into {table_name} ({len(profiles)} columns, {downgraded} kept as VARCHAR)")

    def get_last_schema_report(self) -> list[ColumnProfile]:
        """Returns the ColumnProfile list produced by the most recent load_dataframe/load_csv call."""
        return self._last_schema_report

    # ==================================================================
    # CONVENIENCE WRAPPERS (kept for backward compatibility)
    # ==================================================================

    def load_structured(self, structured_df: pd.DataFrame, **kwargs) -> list[ColumnProfile]:
        return self.load_dataframe(structured_df, STRUCTURED_TABLE, **kwargs)

    def load_metadata(self, metadata_df: pd.DataFrame, **kwargs) -> list[ColumnProfile]:
        return self.load_dataframe(metadata_df, METADATA_TABLE, **kwargs)

    def load_intelligence(self, intelligence_df: pd.DataFrame, **kwargs) -> list[ColumnProfile]:
        return self.load_dataframe(intelligence_df, INTELLIGENCE_TABLE, **kwargs)

    # ==================================================================
    # READ TABLE
    # ==================================================================

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read an entire table from DuckDB and return it as a pandas DataFrame."""
        with self._lock:
            df = self.conn.execute(f"SELECT * FROM {IdentifierSanitizer.quote(table_name)}").fetchdf()
            print(f"  Read {len(df):,} rows from {table_name}")
            return df

    # ==================================================================
    # LOAD CSV DIRECTLY
    #
    # Schema inference is performed against a DuckDB-side sample (never
    # the full file), and the final load runs as SQL entirely inside
    # DuckDB (CREATE TABLE ... AS SELECT <casts> FROM read_csv_auto(...)).
    # This means arbitrarily large CSV files (10M / 50M / 100M+ rows) are
    # never fully materialized as a pandas DataFrame - the only pandas
    # object created is the small profiling sample.
    # ==================================================================

    def load_csv(
        self,
        csv_path: str,
        table_name: str,
        mode: str = "replace",
        sample_size: int = _SAMPLE_SIZE_DEFAULT,
        validate: bool = True,
        mismatch_tolerance: float = _MISMATCH_TOLERANCE,
    ) -> list[ColumnProfile]:
        with self._lock:
            safe_path = IdentifierSanitizer.escape_literal(str(csv_path))
            raw_view = f"__csvraw_{uuid_module.uuid4().hex[:8]}"

            try:
                self.conn.execute(
                    f"CREATE OR REPLACE TEMP VIEW {raw_view} AS "
                    f"SELECT * FROM read_csv_auto('{safe_path}', ALL_VARCHAR=TRUE)"
                )
            except duckdb.Error as exc:
                raise RuntimeError(
                    f"Failed to read CSV '{csv_path}' (invalid UTF-8, malformed rows, or "
                    f"missing file): {exc}"
                ) from exc

            try:
                total_rows = self.conn.execute(f"SELECT COUNT(*) FROM {raw_view}").fetchone()[0]
                if total_rows == 0:
                    print(f"Skipped {table_name} (CSV '{csv_path}' has 0 rows)")
                    return []

                sample_rows = min(sample_size, total_rows)
                sample_df = self.conn.execute(
                    f"SELECT * FROM {raw_view} USING SAMPLE {sample_rows} ROWS"
                ).fetchdf()

                original_labels = list(sample_df.columns)
                safe_names = IdentifierSanitizer.sanitize_columns(original_labels)
                sample_df.columns = safe_names

                # Renamed view over the *full* CSV relation (still lazy -
                # DuckDB does not materialize this), used both for cast
                # validation and for the final full-file load below.
                renamed_view = f"__csvview_{uuid_module.uuid4().hex[:8]}"
                alias_clause = ", ".join(
                    f"{IdentifierSanitizer.quote(orig)} AS {IdentifierSanitizer.quote(safe)}"
                    for orig, safe in zip(original_labels, safe_names)
                )
                self.conn.execute(
                    f"CREATE OR REPLACE TEMP VIEW {renamed_view} AS "
                    f"SELECT {alias_clause} FROM {raw_view}"
                )

                try:
                    engine = SchemaInferenceEngine(
                        self.conn, sample_size=sample_size, mismatch_tolerance=mismatch_tolerance
                    )
                    if validate:
                        profiles = engine.infer(sample_df, renamed_view, original_labels)
                    else:
                        profiler = ColumnProfiler(sample_size=sample_size)
                        profiles = []
                        for i, safe_col in enumerate(sample_df.columns):
                            p = profiler.profile(sample_df.iloc[:, i], original_labels[i], safe_col)
                            p.cast_expression = CastExpressionBuilder.build(p)
                            profiles.append(p)

                    if mode == "fail" and self.table_exists(table_name):
                        raise ValueError(f"Table '{table_name}' already exists (mode='fail').")
                    if mode == "upsert":
                        raise NotImplementedError(
                            "Upsert mode is design-ready but not implemented for CSV loads; "
                            "see load_dataframe() docstring."
                        )
                    if mode == "append" and self.table_exists(table_name):
                        self._validate_append_compatibility(table_name, profiles)

                    select_clause = self._build_select_clause(profiles)
                    self._load_single_shot(renamed_view, table_name, select_clause, mode)

                    self._last_schema_report = profiles
                    self._print_schema_summary(table_name, total_rows, profiles)
                    return profiles
                finally:
                    self.conn.execute(f"DROP VIEW IF EXISTS {renamed_view}")
            finally:
                self.conn.execute(f"DROP VIEW IF EXISTS {raw_view}")

    # ==================================================================
    # TABLE EXISTS
    # ==================================================================

    def table_exists(self, table_name: str) -> bool:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        return result > 0

    # ==================================================================
    # ROW COUNT
    # ==================================================================

    def row_count(self, table_name: str) -> int:
        with self._lock:
            return self.conn.execute(
                f"SELECT COUNT(*) FROM {IdentifierSanitizer.quote(table_name)}"
            ).fetchone()[0]

    # ==================================================================
    # LIST TABLES
    # ==================================================================

    def show_tables(self) -> None:
        with self._lock:
            tables = self.conn.execute("SHOW TABLES").fetchall()
            print("\nDuckDB Tables:\n")
            for table in tables:
                print(f"  • {table[0]}")

    # ==================================================================
    # TABLE INFO
    # ==================================================================

    def describe_table(self, table_name: str) -> pd.DataFrame:
        with self._lock:
            result = self.conn.execute(f"DESCRIBE {IdentifierSanitizer.quote(table_name)}").fetchdf()
            print(result)
            return result

    # ==================================================================
    # QUERY
    # ==================================================================

    def query(self, sql: str) -> pd.DataFrame:
        with self._lock:
            return self.conn.execute(sql).fetchdf()

    # ==================================================================
    # CLOSE
    # ==================================================================

    def close(self) -> None:
        with self._lock:
            self.conn.close()
            print("\nDuckDB Connection Closed")
    if __name__ == "__main__":
        print("Running DuckDB Loader...")