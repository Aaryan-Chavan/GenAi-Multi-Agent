from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)

from sentence_transformers import (
    SentenceTransformer
)

from Config.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    TOP_K_QDRANT
)


class QdrantRetriever:

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        canonical_mapping_path: Optional[str] = None
    ):

        self.collection_name = collection_name

        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=60
        )

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE
        )

        self.canonical_map = {}

        if canonical_mapping_path:
            self._load_canonical_mapping(
                canonical_mapping_path
            )

    # =====================================================
    # CANONICAL MAPPING
    # =====================================================

    def _load_canonical_mapping(
        self,
        file_path: str
    ):

        path = Path(file_path)

        if not path.exists():
            return

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            mapping = json.load(f)

        reverse_map = {}

        for canonical, aliases in mapping.items():

            canonical = canonical.lower()

            reverse_map[
                canonical
            ] = canonical

            for alias in aliases:

                reverse_map[
                    alias.lower()
                ] = canonical

        self.canonical_map = reverse_map

    # =====================================================
    # QUERY NORMALIZATION
    # =====================================================

    def normalize_query(
        self,
        query: str
    ) -> str:

        query = str(query).lower()

        query = re.sub(
            r"[^\w\s]",
            " ",
            query
        )

        query = re.sub(
            r"\s+",
            " ",
            query
        ).strip()

        if not self.canonical_map:
            return query

        normalized = []

        for token in query.split():

            normalized.append(
                self.canonical_map.get(
                    token,
                    token
                )
            )

        return " ".join(
            normalized
        )

    # =====================================================
    # EMBEDDING
    # =====================================================

    def _embed_query(
        self,
        query: str
    ) -> List[float]:

        query = self.normalize_query(
            query
        )

        embedding = self.embedder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.tolist()

    # =====================================================
    # FILTERS
    # =====================================================

    @staticmethod
    def build_filter(
        filters: Optional[Dict]
    ):

        if not filters:
            return None

        conditions = []

        for key, value in filters.items():

            conditions.append(

                FieldCondition(
                    key=key,
                    match=MatchValue(
                        value=value
                    )
                )

            )

        return Filter(
            must=conditions
        )

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query: str,
        top_k: int = TOP_K_QDRANT,
        filters: Optional[Dict] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:

        if not query:
            return []

        vector = self._embed_query(
            query
        )

        q_filter = self.build_filter(
            filters
        )

        response = self.client.query_points(

            collection_name=
            self.collection_name,

            query=vector,

            query_filter=
            q_filter,

            limit=top_k,

            score_threshold=
            score_threshold,

            search_params=
            SearchParams(
                hnsw_ef=256,
                exact=False
            ),

            with_payload=True,
            with_vectors=False

        )

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append({

                "id":
                    point.id,

                "score":
                    round(
                        float(point.score),
                        4
                    ),

                "chunk_id":
                    payload.get(
                        "chunk_id"
                    ),

                "record_id":
                    payload.get(
                        "record_id"
                    ),

                "chunk_text":
                    payload.get(
                        "chunk_text",
                        ""
                    ),

                "topic":
                    payload.get(
                        "topic"
                    ),

                "keywords":
                    payload.get(
                        "keywords"
                    ),

                "sentiment":
                    payload.get(
                        "sentiment_label"
                    ),

                "payload":
                    payload
            })

        return results

    # =====================================================
    # TOP MATCH
    # =====================================================

    def top_match(
        self,
        query: str
    ):

        results = self.search(
            query=query,
            top_k=1
        )

        if not results:
            return None

        return results[0]

    # =====================================================
    # COUNT
    # =====================================================

    def count(self):

        return self.client.get_collection(
            self.collection_name
        ).points_count

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def health_check(self) -> bool:

        try:

            self.client.get_collection(
                self.collection_name
            )

            return True

        except Exception:

            return False