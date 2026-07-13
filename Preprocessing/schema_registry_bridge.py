from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from Preprocessing.schema_analyzer import SchemaAnalyzer
from Agents.retrieval_plan_builder import (
    ColumnMeta,
    ColumnRole,
    RelationshipMeta,
    StaticSchemaRegistry,
    TableMeta,
    VectorCollectionMeta,
)

_NUMERIC_DTYPES = {
    "int64", "int32", "int16", "int8",
    "float64", "float32", "float16",
    "Int64", "UInt64", "UInt32", "UInt16",
}
_BOOLEAN_DTYPES = {"bool"}


def _role_for_structured(dtype: str) -> ColumnRole:
    """SchemaAnalyzer's "structured" bucket = numeric OR boolean OR
    datetime, all under one label. Re-split using the dtype string it
    already recorded per column."""
    if dtype in _BOOLEAN_DTYPES:
        return ColumnRole.BOOLEAN
    if "datetime" in dtype:
        return ColumnRole.TEMPORAL
    if dtype in _NUMERIC_DTYPES:
        return ColumnRole.NUMERIC
    # Structured-but-unrecognized dtype (e.g. a pandas extension type
    # SchemaAnalyzer's _is_structured_dtype didn't list) -- degrade to
    # UNKNOWN rather than guessing.
    return ColumnRole.UNKNOWN


def _role_for_metadata(
    stats: Dict,
    high_uniqueness_threshold: float,
    low_uniqueness_threshold: float,
) -> ColumnRole:
    """SchemaAnalyzer's "metadata" bucket collapses three distinct
    signals it already computes (high-uniqueness/short = ID-like,
    low-uniqueness = categorical, everything else = an unlabeled
    default) into one flat label. Re-derive the split here from the
    same stats fields SchemaAnalyzer itself used, since the original
    sub-classification isn't returned."""
    unique_ratio = stats.get("unique_ratio", 0)
    p75_length = stats.get("p75_length", 0)

    if unique_ratio >= high_uniqueness_threshold and p75_length <= 40:
        return ColumnRole.IDENTIFIER
    if unique_ratio <= low_uniqueness_threshold:
        return ColumnRole.CATEGORICAL
    # SchemaAnalyzer's own default fallback for this bucket: some
    # short-to-medium string column that's neither clearly an ID nor
    # clearly low-cardinality. Categorical is the closer fit of the
    # two available roles (bounded-ish label set) versus UNKNOWN,
    # which would make the column invisible to filtering entirely.
    return ColumnRole.CATEGORICAL


def _build_column_meta(
    col_name: str,
    bucket: str,
    stats: Dict,
    analyzer: SchemaAnalyzer,
) -> ColumnMeta:
    if bucket == "structured":
        role = _role_for_structured(stats["dtype"])
    elif bucket == "unstructured":
        role = ColumnRole.TEXTUAL
    else:  # "metadata"
        role = _role_for_metadata(
            stats,
            analyzer.high_uniqueness_threshold,
            analyzer.low_uniqueness_threshold,
        )

    return ColumnMeta(
        name=col_name,
        role=role,
        searchable=(role == ColumnRole.TEXTUAL),
        aggregatable=(role == ColumnRole.NUMERIC),
        filterable=True,
        sortable=True,
    )


def _infer_primary_key(columns: Sequence[ColumnMeta], stats_by_col: Dict[str, Dict]) -> Optional[str]:
    """Best-effort: the IDENTIFIER column with the highest uniqueness
    ratio (ties broken by column order). Returns None if no IDENTIFIER
    column exists -- RetrievalPlanBuilder treats primary_key as
    optional throughout, so this is a convenience, not a requirement."""
    identifier_cols = [c for c in columns if c.role == ColumnRole.IDENTIFIER]
    if not identifier_cols:
        return None
    return max(
        identifier_cols,
        key=lambda c: stats_by_col.get(c.name, {}).get("unique_ratio", 0),
    ).name


def _schema_fingerprint(table_name: str, columns: Sequence[ColumnMeta]) -> str:
    """Deterministic hash of (table name, column name+role pairs).
    Changes whenever the analyzed schema shape changes -- this is what
    SchemaContextCache watches via registry.version to know when to
    invalidate its caches. Deliberately NOT based on Python's built-in
    hash() since that's randomized per-process for strings; this needs
    to be reproducible."""
    payload = table_name + "|" + "|".join(f"{c.name}:{c.role.value}" for c in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_table_meta(
    df: pd.DataFrame,
    table_name: str,
    analyzer: Optional[SchemaAnalyzer] = None,
) -> Tuple[TableMeta, Dict]:
    """
    Runs SchemaAnalyzer on `df` and returns a (TableMeta, raw_analysis)
    pair -- the raw analysis dict is returned too so callers can still
    use SchemaAnalyzer.print_summary() or inspect column_statistics
    directly if needed.
    """
    analyzer = analyzer or SchemaAnalyzer()
    analysis = analyzer.analyze(df)
    stats_by_col = analysis["column_statistics"]

    bucket_by_col: Dict[str, str] = {}
    for col in analysis["structured_columns"]:
        bucket_by_col[col] = "structured"
    for col in analysis["unstructured_columns"]:
        bucket_by_col[col] = "unstructured"
    for col in analysis["metadata_columns"]:
        bucket_by_col[col] = "metadata"

    columns: List[ColumnMeta] = [
        _build_column_meta(col, bucket_by_col[col], stats_by_col[col], analyzer)
        for col in df.columns
    ]

    primary_key = _infer_primary_key(columns, stats_by_col)

    table = TableMeta(
        name=table_name,
        columns=tuple(columns),
        row_count=len(df),
        primary_key=primary_key,
    )
    return table, analysis


def build_schema_registry(
    tables: Iterable[Tuple[str, pd.DataFrame]],
    relationships: Optional[Sequence[RelationshipMeta]] = None,
    vector_collections: Optional[Sequence[VectorCollectionMeta]] = None,
    analyzer: Optional[SchemaAnalyzer] = None,
) -> StaticSchemaRegistry:
    """
    Main entry point. Pass an iterable of (table_name, dataframe) pairs
    -- one per uploaded table -- plus any relationships/vector
    collections your pipeline already knows about from elsewhere (this
    function does not discover either on its own; see module docstring).

    Example:
        registry = build_schema_registry([
            ("employees", employees_df),
            ("performance_reviews", reviews_df),
        ], relationships=[
            RelationshipMeta(from_table="performance_reviews", from_column="employee_id",
                              to_table="employees", to_column="employee_id"),
        ])
        planner = RetrievalPlanBuilder(schema_registry=registry)
    """
    analyzer = analyzer or SchemaAnalyzer()
    # Normalize optional arguments
    relationships = list(relationships or [])
    vector_collections = list(vector_collections or [])

    table_metas: List[TableMeta] = []
    fingerprints: List[str] = []

    for table_name, df in tables:
        table_meta, _raw = build_table_meta(df, table_name, analyzer)
        table_metas.append(table_meta)
        fingerprints.append(_schema_fingerprint(table_name, table_meta.columns))

    version_payload = "|".join(sorted(fingerprints)) + f"|rel={len(relationships)}|vec={len(vector_collections)}"
    version = hashlib.sha256(version_payload.encode("utf-8")).hexdigest()[:16]

    return StaticSchemaRegistry(
        tables=table_metas,
        relationships=relationships,
        vector_collections=vector_collections,
        version=version,
    )
