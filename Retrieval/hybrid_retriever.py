from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from Retrieval.qdrant_retriever import QdrantRetriever
from Retrieval.duckdb_retriever import DuckDBRetriever

from Config.settings import (
    TOP_K_QDRANT,
    MAX_RETRIEVAL_RESULTS
)


class HybridRetriever:
    """
    Hybrid Retrieval Pipeline

    Flow:
        Query
          ↓
        Qdrant Semantic Search
          ↓
        Chunk IDs
          ↓
        DuckDB Metadata Lookup
          ↓
        Merged Results

    Output:
        [
            {
                "chunk_id": "...",
                "score": 0.91,
                "chunk_text": "...",
                ...
            }
        ]
    """

    def __init__(
        self,
        qdrant_retriever: Optional[QdrantRetriever] = None,
        duckdb_retriever: Optional[DuckDBRetriever] = None
    ) -> None:

        self.qdrant = (
            qdrant_retriever
            if qdrant_retriever
            else QdrantRetriever()
        )

        self.duckdb = (
            duckdb_retriever
            if duckdb_retriever
            else DuckDBRetriever()
        )

    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    @staticmethod
    def normalize_query(
        query: str
    ) -> str:

        if not query:
            return ""

        return " ".join(
            str(query).strip().split()
        )

    # =====================================================
    # MAIN RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_QDRANT,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        query = self.normalize_query(
            query
        )

        if not query:
            return []

        semantic_hits = self.qdrant.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not semantic_hits:
            return []

        chunk_ids = []

        seen = set()

        for hit in semantic_hits:

            chunk_id = str(
                hit.get("chunk_id")
            )

            if chunk_id not in seen:

                seen.add(chunk_id)

                chunk_ids.append(
                    chunk_id
                )

        records = self.duckdb.fetch_by_chunk_ids(
            chunk_ids=chunk_ids
        )

        if not records:
            return []

        record_map = {

            str(record["chunk_id"]): record

            for record in records

        }

        merged_results = []

        for hit in semantic_hits:

            chunk_id = str(
                hit["chunk_id"]
            )

            record = record_map.get(
                chunk_id
            )

            if not record:
                continue

            merged = {
                **record,
                "score": float(
                    hit["score"]
                )
            }

            merged_results.append(
                merged
            )

        return merged_results[
            :MAX_RETRIEVAL_RESULTS
        ]

    # =====================================================
    # FILTERED RETRIEVAL
    # =====================================================

    def retrieve_with_filters(
        self,
        query: str,
        top_k: int = TOP_K_QDRANT,
        score_threshold: float = 0.0,
        **filters: Any
    ) -> List[Dict[str, Any]]:

        results = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not filters:
            return results

        filtered_results = []

        for result in results:

            keep = True

            for key, value in filters.items():

                if value is None:
                    continue

                if str(
                    result.get(key)
                ).lower() != str(
                    value
                ).lower():

                    keep = False

                    break

            if keep:

                filtered_results.append(
                    result
                )

        return filtered_results

    # =====================================================
    # ONLY SEMANTIC IDS
    # =====================================================

    def retrieve_ids(
        self,
        query: str,
        top_k: int = TOP_K_QDRANT
    ) -> List[str]:

        hits = self.qdrant.search(
            query=query,
            top_k=top_k
        )

        return [

            str(hit["chunk_id"])

            for hit in hits

        ]

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def health_check(
        self
    ) -> Dict[str, bool]:

        status = {
            "qdrant": False,
            "duckdb": False
        }

        try:

            self.qdrant.count()

            status["qdrant"] = True

        except Exception:
            pass

        try:

            self.duckdb.fetch_dataframe(
                "SELECT 1"
            )

            status["duckdb"] = True

        except Exception:
            pass

        return status

    # =====================================================
    # CLEANUP
    # =====================================================

    def close(self) -> None:

        try:
            self.duckdb.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ):
        self.close()


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    retriever = HybridRetriever()

    results = retriever.retrieve(
        query="battery drains quickly",
        top_k=5
    )

    print("\nRESULTS\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {i}")
        print("-" * 60)

        for k, v in result.items():

            print(f"{k}: {v}")

    print(
        "\nHealth:",
        retriever.health_check()
    )

    retriever.close()