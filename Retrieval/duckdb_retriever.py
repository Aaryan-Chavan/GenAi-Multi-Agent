# retrieval/duckdb_retriever.py

from __future__ import annotations

import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence
)

import duckdb
import pandas as pd

from Config.settings import (
    DUCKDB_FILE,
    INTELLIGENCE_TABLE
)


class DuckDBRetriever:
    """
    Production DuckDB Retriever

    Responsibilities
    ----------------
    1. Retrieve chunk metadata
    2. Retrieve chunk text
    3. Apply metadata filters
    4. Execute analytical queries
    5. Support hybrid retrieval

    Used By
    --------
    HybridRetriever
    StructuredAgent
    SemanticAgent
    """

    def __init__(
        self,
        db_path: Optional[str] = None
    ):

        self.db_path = str(
            db_path or DUCKDB_FILE
        )

        self.conn = duckdb.connect(
            self.db_path,
            read_only=True
        )

    # =====================================================
    # INTERNAL
    # =====================================================

    def _validate_table(
        self,
        table_name: str
    ) -> None:

        tables = self.list_tables()

        if table_name not in tables:

            raise ValueError(
                f"Table '{table_name}' "
                f"not found. Available: {tables}"
            )

    # =====================================================
    # TABLE INFO
    # =====================================================

    def list_tables(
        self
    ) -> List[str]:

        result = self.conn.execute(
            """
            SHOW TABLES
            """
        ).fetchall()

        return [r[0] for r in result]

    # =====================================================
    # CHUNK LOOKUP
    # =====================================================

    def fetch_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        table_name: str = INTELLIGENCE_TABLE,
        columns: str = "*"
    ) -> List[Dict]:

        if not chunk_ids:
            return []

        self._validate_table(
            table_name
        )

        placeholders = ",".join(
            ["?"] * len(chunk_ids)
        )

        query = f"""
        SELECT {columns}
        FROM {table_name}
        WHERE chunk_id IN ({placeholders})
        """

        df = self.conn.execute(
            query,
            list(chunk_ids)
        ).fetchdf()

        return df.to_dict(
            orient="records"
        )

    # =====================================================
    # SINGLE CHUNK
    # =====================================================

    def fetch_one(
        self,
        chunk_id: str,
        table_name: str = INTELLIGENCE_TABLE
    ) -> Optional[Dict]:

        rows = self.fetch_by_chunk_ids(
            [chunk_id],
            table_name
        )

        return rows[0] if rows else None

    # =====================================================
    # METADATA FILTERS
    # =====================================================

    def fetch_with_filters(
        self,
        table_name: str = INTELLIGENCE_TABLE,
        limit: int = 100,
        columns: str = "*",
        **filters
    ) -> List[Dict]:

        self._validate_table(
            table_name
        )

        where_clauses = []
        values = []

        for key, value in filters.items():

            if value is None:
                continue

            where_clauses.append(
                f"{key} = ?"
            )

            values.append(value)

        where_sql = ""

        if where_clauses:

            where_sql = (
                "WHERE "
                + " AND ".join(
                    where_clauses
                )
            )

        query = f"""
        SELECT {columns}
        FROM {table_name}
        {where_sql}
        LIMIT ?
        """

        values.append(limit)

        df = self.conn.execute(
            query,
            values
        ).fetchdf()

        return df.to_dict(
            orient="records"
        )

    # =====================================================
    # DATAFRAME QUERY
    # =====================================================

    def fetch_dataframe(
        self,
        query: str,
        params: Optional[List[Any]] = None
    ) -> pd.DataFrame:

        return self.conn.execute(
            query,
            params or []
        ).fetchdf()

    # =====================================================
    # HYBRID LOOKUP
    # =====================================================

    def fetch_from_qdrant_hits(
        self,
        qdrant_results: List[Dict],
        table_name: str = INTELLIGENCE_TABLE
    ) -> List[Dict]:

        if not qdrant_results:
            return []

        score_map = {

            str(r["chunk_id"]):
            r["score"]

            for r in qdrant_results
        }

        chunk_ids = list(
            score_map.keys()
        )

        rows = self.fetch_by_chunk_ids(
            chunk_ids,
            table_name
        )

        for row in rows:

            row["semantic_score"] = (
                score_map.get(
                    str(
                        row.get(
                            "chunk_id"
                        )
                    ),
                    0.0
                )
            )

        rows.sort(
            key=lambda x:
            x["semantic_score"],
            reverse=True
        )

        return rows

    # =====================================================
    # ANALYTICS
    # =====================================================

    def execute(
        self,
        query: str,
        params: Optional[List[Any]] = None
    ):

        start = time.perf_counter()

        result = self.conn.execute(
            query,
            params or []
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"DuckDB Query "
            f"({elapsed:.3f}s)"
        )

        return result

    # =====================================================
    # TABLE PREVIEW
    # =====================================================

    def preview(
        self,
        table_name: str = INTELLIGENCE_TABLE,
        limit: int = 5
    ) -> pd.DataFrame:

        self._validate_table(
            table_name
        )

        return self.conn.execute(
            f"""
            SELECT *
            FROM {table_name}
            LIMIT {limit}
            """
        ).fetchdf()

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        try:
            self.conn.close()

        except Exception:
            pass

    # =====================================================
    # CONTEXT MANAGER
    # =====================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):

        self.close()


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    retriever = DuckDBRetriever()

    print(
        "\nAvailable Tables:"
    )

    print(
        retriever.list_tables()
    )

    sample = retriever.preview()

    print(
        "\nPreview:"
    )

    print(sample.head())

    retriever.close()