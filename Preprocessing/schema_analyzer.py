# preprocessing/schema_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List


class SchemaAnalyzer:
    """
    Classifies DataFrame columns into:
    - structured (numeric, boolean, datetime)
    - unstructured (long-form text)
    - metadata (ids, categories, short text, misc)
    """

    def __init__(
        self,
        text_length_threshold: int = 80,
        text_word_threshold: int = 10,
        high_uniqueness_threshold: float = 0.95,
        low_uniqueness_threshold: float = 0.05
    ):
        self.text_length_threshold = text_length_threshold
        self.text_word_threshold = text_word_threshold
        self.high_uniqueness_threshold = high_uniqueness_threshold
        self.low_uniqueness_threshold = low_uniqueness_threshold

    # -----------------------------
    # MAIN ENTRY
    # -----------------------------
    def analyze(self, df: pd.DataFrame) -> Dict:

        structured_columns = []
        unstructured_columns = []
        metadata_columns = []

        column_statistics = {}

        for col in df.columns:

            stats = self._analyze_column(df[col])
            column_statistics[col] = stats

            bucket = self._classify_column(stats)

            if bucket == "structured":
                structured_columns.append(col)
            elif bucket == "unstructured":
                unstructured_columns.append(col)
            else:
                metadata_columns.append(col)

        return {
            "structured_columns": structured_columns,
            "unstructured_columns": unstructured_columns,
            "metadata_columns": metadata_columns,
            "column_statistics": column_statistics
        }

    # -----------------------------
    # COLUMN ANALYSIS
    # -----------------------------
    def _analyze_column(self, series: pd.Series) -> Dict:

        non_null = series.dropna()

        stats = {
            "dtype": str(series.dtype),
            "total_rows": len(series),
            "non_null_count": len(non_null),
            "null_count": series.isna().sum(),
            "unique_count": non_null.nunique(),
            "unique_ratio": (non_null.nunique() / len(non_null)) if len(non_null) > 0 else 0
        }

        # -----------------------------
        # TEXT METRICS (OBJECT / STRING)
        # -----------------------------
        if self._is_textual(series):

            sample = non_null.astype(str)

            if len(sample) > 0:

                lengths = sample.str.len()
                word_counts = sample.str.split().str.len()

                stats.update({
                    "avg_length": lengths.mean(),
                    "median_length": lengths.median(),
                    "p75_length": np.percentile(lengths, 75),
                    "max_length": lengths.max(),

                    "avg_words": word_counts.mean(),
                    "median_words": word_counts.median(),
                    "p75_words": np.percentile(word_counts, 75)
                })

            else:
                stats.update(self._empty_text_stats())

        else:
            stats.update(self._empty_text_stats())

        return stats

    # -----------------------------
    # TEXT DETECTION
    # -----------------------------
    def _is_textual(self, series: pd.Series) -> bool:
        return (
            pd.api.types.is_object_dtype(series) or
            pd.api.types.is_string_dtype(series)
        )

    # -----------------------------
    # EMPTY TEXT STATS
    # -----------------------------
    def _empty_text_stats(self) -> Dict:
        return {
            "avg_length": 0,
            "median_length": 0,
            "p75_length": 0,
            "max_length": 0,
            "avg_words": 0,
            "median_words": 0,
            "p75_words": 0
        }

    # -----------------------------
    # CLASSIFICATION LOGIC
    # -----------------------------
    def _classify_column(self, stats: Dict) -> str:

        dtype = stats["dtype"]

        # -----------------------------
        # STRUCTURED
        # -----------------------------
        if self._is_structured_dtype(dtype):
            return "structured"

        # datetime detection (object-safe fallback)
        if "datetime" in dtype:
            return "structured"

        # -----------------------------
        # UNSTRUCTURED (STRONG TEXT SIGNAL)
        # -----------------------------
        if (
            stats.get("p75_length", 0) >= self.text_length_threshold
            and stats.get("p75_words", 0) >= self.text_word_threshold
        ):
            return "unstructured"

        # -----------------------------
        # ID COLUMN DETECTION (metadata)
        # -----------------------------
        if (
            stats["unique_ratio"] >= self.high_uniqueness_threshold
            and stats.get("p75_length", 0) <= 40
        ):
            return "metadata"

        # -----------------------------
        # LOW CARDINALITY CATEGORICAL (metadata)
        # -----------------------------
        if stats["unique_ratio"] <= self.low_uniqueness_threshold:
            return "metadata"

        # -----------------------------
        # DEFAULT
        # -----------------------------
        return "metadata"

    # -----------------------------
    # STRUCTURED DTYPE CHECK
    # -----------------------------
    def _is_structured_dtype(self, dtype: str) -> bool:

        numeric_types = {
            "int64", "int32", "int16", "int8",
            "float64", "float32", "float16",
            "bool",
            "Int64", "UInt64", "UInt32", "UInt16"
        }

        return dtype in numeric_types

    # -----------------------------
    # PRINT SUMMARY
    # -----------------------------
    def print_summary(self, result: Dict):

        print("\n" + "=" * 60)
        print("SCHEMA ANALYSIS RESULT")
        print("=" * 60)

        print("\nSTRUCTURED COLUMNS:")
        for col in result["structured_columns"]:
            print(f"  • {col}")

        print("\nUNSTRUCTURED COLUMNS:")
        for col in result["unstructured_columns"]:
            print(f"  • {col}")

        print("\nMETADATA COLUMNS:")
        for col in result["metadata_columns"]:
            print(f"  • {col}")

        print("\n" + "=" * 60)